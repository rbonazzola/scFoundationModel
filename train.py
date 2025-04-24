import os 
import time
import logging

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from data.multi_file_dataset import MultiScRNADataset as scRNADataset
from model.performer.performer import PerformerLM
from utils import save_ckpt

from torch.optim import Adam
from masking import data_mask
import mlflow
import dagshub

from tqdm import tqdm

SEED = 42
torch.set_float32_matmul_precision('high')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
dagshub.init("scFoundationModel", "rbonazzola", mlflow=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_ckpt(epoch, model, optimizer, scheduler, losses, model_name, ckpt_folder):
    """
    save checkpoint
    """
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    torch.save(
        {
            'epoch': epoch,            
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'losses': losses,
        },
        f'{ckpt_folder}/{model_name}_{epoch}.pth'
    )


class Trainer:
    def __init__(self, args):
        logging.info("Initializing Trainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half_precision = args.half_precision and self.device.type == "cuda"
        self.SEQ_LEN = args.gene_num + 1
        self.LEARNING_RATE = args.lr
        self.MAX_EPOCHS = args.max_epochs
        self.MAX_BATCHES = args.max_batches
        self.VALIDATE_EVERY = 5
        self.GRADIENT_ACCUMULATION = args.gradient_accumulation or GRADIENT_ACCUMULATION
        self.BATCH_SIZE = args.batch_size or BATCH_SIZE
        self.TOP_N_GENES = args.top_n_genes
        self.NUM_WORKERS = args.num_workers or 0
        self.USING_COMPILE = args.compile
        self.MASK_PROBABILITY = args.mask_probability

        self._load_data(args.data_path)

        self.model_args = {
            "num_tokens": self.train_dataset.N_CLASSES,
            "dim": args.embedding_dim,
            "depth": args.depth,
            "max_seq_len": self.SEQ_LEN,
            "heads": args.heads,
            "local_attn_heads": args.local_attn_heads,
            "g2v_position_emb": None
        }

        self._initialize_model()
        self._initialize_training_components()
    

    def _load_data(self, data_path):
        
        logging.info("Loading data")
        start_time = time.time()

        self.train_dataset = scRNADataset(f"{args.data_path}/train")
        self.val_dataset = scRNADataset(f"{args.data_path}/val")
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS)
        
        self.N_CLASSES = self.train_dataset.N_CLASSES
        logging.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
    

    def _initialize_model(self):
        logging.info("Initializing model")
        start_time = time.time()
        self.model = PerformerLM(**self.model_args).to(self.device)
        self.model
        
        if self.use_half_precision:
            self.model = self.model.half()
        
        if args.compile:
            self.model = torch.compile(self.model)

        logging.info(f"Model initialized in {time.time() - start_time:.2f} seconds")
    
    def _initialize_training_components(self):
        
        self.optimizer = Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=1)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.N_CLASSES - 1, reduction='mean')
        self.softmax = nn.Softmax(dim=-1)
    
    def train(self):
        logging.info("Starting training")
        print(f"{type(self.MASK_PROBABILITY)=}")
        mlflow.start_run()
        mlflow.log_params({
           "max_epochs": self.MAX_EPOCHS, 
           "learning_rate": self.LEARNING_RATE, 
           "batch_size": self.BATCH_SIZE,
           "n_batches": self.MAX_BATCHES,
           "samples_per_epoch": len(self.train_dataset) if self.MAX_BATCHES is None else min(self.MAX_BATCHES * self.BATCH_SIZE, len(self.train_dataset)),
           "embedding_dim": self.model_args["dim"],
           "depth": self.model_args["depth"],
           "heads": self.model_args["heads"],
           "local_attn_heads": self.model_args["local_attn_heads"],
           "platform": os.uname().nodename,
           "top_n_genes": self.TOP_N_GENES,
           "num_workers": self.NUM_WORKERS,
           "using_compile": self.USING_COMPILE,
           "n_parameters": count_parameters(self.model),
           "mask_probability": self.MASK_PROBABILITY
        })

        
        for epoch in range(1, self.MAX_EPOCHS + 1):
            
            epoch_start_time = time.time()
            self.model.train()
            
            batch_times, processing_times, mask_times = [], [], []
            
            total_batches = 0
            running_loss = 0.0
            
            for index, data in tqdm(enumerate(self.train_loader)):

                if self.MAX_BATCHES and index >= self.MAX_BATCHES:
                    break
            
                batch_start_time = time.time()           
                
                data = data.to(self.device)
            
                # ──────── MASKING ────────
                mask_start_time = time.time()
                data, labels = data_mask(data, mask_prob=self.MASK_PROBABILITY)
                mask_time = time.time() - mask_start_time
                torch.cuda.synchronize()
            
                # ──────── FORWARD + LOSS ────────
                processing_start_time = time.time()
                
                with torch.cuda.amp.autocast(enabled=self.use_half_precision):                                       
                    logits = self.model(data)
                    torch.cuda.synchronize()
                loss = self.loss_fn(logits.transpose(1, 2), labels) / self.GRADIENT_ACCUMULATION
            
                # ──────── BACKWARD ────────
                loss.backward()
            
                # ──────── OPTIMIZER STEP ────────
                if index % self.GRADIENT_ACCUMULATION == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
                # ──────── TIME COLLECTION ────────
                processing_time = time.time() - processing_start_time
                batch_time = time.time() - batch_start_time
                
                batch_times.append(batch_time)
                processing_times.append(processing_time)
                mask_times.append(mask_time)
                
                running_loss += loss.item()
                total_batches += 1

            
            epoch_time = time.time() - epoch_start_time
            avg_batch_time = sum(batch_times) / total_batches if total_batches > 0 else 0
            avg_processing_time = sum(processing_times) / total_batches if total_batches > 0 else 0
            data_loading_time = epoch_time - sum(batch_times)
            avg_mask_time = sum(mask_times) / total_batches if total_batches > 0 else 0

            epoch_loss = running_loss / total_batches

            logging.info(f'Epoch {epoch}: Loss = {running_loss / total_batches:.6f}')
            logging.info(f'Epoch {epoch}: Time = {epoch_time:.2f} sec, Avg batch = {avg_batch_time:.4f} sec')
            logging.info(f'Processing time: {avg_processing_time:.4f} sec, Data loading time: {data_loading_time:.4f} sec')
            
            mlflow.log_metrics({
                "epoch_time": epoch_time,
                "avg_batch_time": avg_batch_time,
                "avg_processing_time": avg_processing_time,
                "data_loading_time": data_loading_time,
                "data_loading_time_per_sample": data_loading_time / total_batches / self.BATCH_SIZE,
                "time_per_sample": epoch_time / total_batches / self.BATCH_SIZE,
                "avg_mask_time": avg_mask_time,
                "train_loss": epoch_loss}, step=epoch)
            
            self.scheduler.step()
            
            if epoch % self.VALIDATE_EVERY == 0:
              self.validate(epoch)
               
            run_id = mlflow.active_run().info.run_id            
            save_ckpt(epoch, self.model, self.optimizer, self.scheduler, epoch_loss, f"performer_model", "./checkpoints/{run_id}")
        
        mlflow.end_run()
        logging.info("Training complete")


    def validate(self, epoch):
        logging.info(f"Validating at epoch {epoch}")
        self.model.eval()
        running_loss_val = 0.0
        correct_num, val_num = 0, 0
        
        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                data, labels = data_mask(data)
                logits = self.model(data)
                loss = self.loss_fn(logits.transpose(1, 2), labels)
                running_loss_val += loss.item()
                final = self.softmax(logits)[..., 1:-1].argmax(dim=-1) + 1
                correct_num += ((labels != self.N_CLASSES - 1) * (final == labels)).sum().item()
                val_num += (labels != self.N_CLASSES - 1).sum().item()
        
        val_loss = running_loss_val / len(self.val_loader)
        val_acc = 100.0 * correct_num / val_num
        logging.info(f'Epoch {epoch}: Validation Loss = {val_loss:.6f}, Accuracy = {val_acc:.4f}%')
        mlflow.log_metrics({"val_loss": val_loss, "val_acc": val_acc}, step=epoch)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--data_path', type=str, default=f"{os.getenv('HOME')}/data/scrna/root")
    parser.add_argument('--gene_num', type=int, default=3932)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gradient_accumulation', type=int, default=1)
    parser.add_argument('--half_precision', action='store_true', default=False, help='Use FP16 for faster training')
    parser.add_argument('--max_batches', type=int, default=100000, help='Limit training to a given number of batches')
    parser.add_argument('--use-flash-attention', '--use_flash_attention', action='store_true', default=False, help='Limit training to a given number of batches')
    parser.add_argument('--mask-probability', '--mask_probability', default=0.15, help='Masking probability during training', type=float)
    parser.add_argument('--embedding_dim', type=int, default=200, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=10, help='Number of attention heads')
    parser.add_argument('--local_attn_heads', type=int, default=0, help='Number of local attention heads')
    parser.add_argument('--highly_variable_genes_file', type=str, default="data/highly_variable_genes.csv", help='File with highly variable genes')
    parser.add_argument('--top_n_genes', type=int, default=None, help="Number of genes to use")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers to use for data loading.")

    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()