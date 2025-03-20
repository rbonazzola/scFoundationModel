import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import scRNADataset
from model.performer.performer import PerformerLM
from utils import save_ckpt
import scanpy as sc

from torch.optim import Adam
from masking import data_mask
from sklearn.model_selection import train_test_split

import mlflow
from tqdm import tqdm

SEED = 42
BATCH_SIZE = 2

dagshub.init("your-repo-name", "your-username", mlflow=True)

torch.set_float32_matmul_precision('high')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    logging.info("Starting training script")
    start_time = time.time()
    
    SEQ_LEN = args.gene_num + 1
    LEARNING_RATE = args.lr
    
    logging.info("Loading data")
    data_load_start = time.time()
    if args.data_path.endswith("gz"):
        import gzip
        with gzip.open(args.data_path) as f:
            data = sc.read_h5ad(f).X
    else:
        data = sc.read_h5ad(args.data_path).X
    data_train, data_val = train_test_split(data, test_size=0.05, random_state=SEED)
    data_load_end = time.time()
    logging.info(f"Data loaded in {data_load_end - data_load_start:.2f} seconds")
    
    train_dataset = scRNADataset(data_train)
    val_dataset = scRNADataset(data_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    N_CLASSES = train_dataset.N_CLASSES
    
    logging.info("Initializing model")
    model_init_start = time.time()
    model = PerformerLM(
        num_tokens=N_CLASSES,
        dim=200,
        depth=2,
        max_seq_len=SEQ_LEN,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.compile(model)
    model_init_end = time.time()
    logging.info(f"Model initialized in {model_init_end - model_init_start:.2f} seconds")
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    PAD_TOKEN_ID = N_CLASSES - 1
    EPOCHS = args.epochs
    VALIDATE_EVERY = 5
    GRADIENT_ACCUMULATION = 10
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='mean')
    softmax = nn.Softmax(dim=-1)
    
    mlflow.start_run()
    mlflow.log_params({"epochs": EPOCHS, "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE})
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        cum_acc = 0.0
        epoch_start = time.time()
    
        for index, data in tqdm(enumerate(train_loader)):
            data = data.to(device)
            data, labels = data_mask(data)
    
            # Forward pass
            logits = model(data)
            loss = loss_fn(logits.transpose(1, 2), labels) / GRADIENT_ACCUMULATION
    
            # Backward pass
            loss.backward()
    
            if index % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e2)
                optimizer.step()
                optimizer.zero_grad()
    
            running_loss += loss.item()
    
            final = softmax(logits)[..., 1:-1]
            final = final.argmax(dim=-1) + 1
            pred_num = (labels != PAD_TOKEN_ID).sum(dim=-1)
            correct_num = ((labels != PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
    
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * (cum_acc / len(train_loader))
    
        logging.info(f'Epoch {epoch}: Training Loss = {epoch_loss:.6f}, Accuracy = {epoch_acc:.4f}%')
        mlflow.log_metrics({"train_loss": epoch_loss, "train_acc": epoch_acc}, step=epoch)
    
        scheduler.step()
    
        if epoch % VALIDATE_EVERY == 0:
            model.eval()
            running_loss_val = 0.0
            correct_num, val_num = 0, 0
    
            with torch.no_grad():
                for index, data in enumerate(val_loader):
                    data = data.to(device)
                    data, labels = data_mask(data)
    
                    logits = model(data)
                    loss = loss_fn(logits.transpose(1, 2), labels)
                    running_loss_val += loss.item()
    
                    final = softmax(logits)[..., 1:-1]
                    final = final.argmax(dim=-1) + 1
    
                    correct_num += ((labels != PAD_TOKEN_ID) * (final == labels)).sum().item()
                    val_num += (labels != PAD_TOKEN_ID).sum().item()
    
            val_loss = running_loss_val / len(val_loader)
            val_acc = 100.0 * correct_num / val_num
    
            logging.info(f'Epoch {epoch}: Validation Loss = {val_loss:.6f}, Accuracy = {val_acc:.4f}%')
            mlflow.log_metrics({"val_loss": val_loss, "val_acc": val_acc}, step=epoch)
    
        save_ckpt(epoch, model, optimizer, scheduler, epoch_loss, "performer_model", "./checkpoints")
    
    mlflow.end_run()
    total_time = time.time() - start_time
    logging.info(f'Total training time: {total_time:.2f} seconds')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--data_path', type=str, default="./data/transforms/CRA004476_transformed.h5ad")
    parser.add_argument('--gene_list', type=str, default="./data/gene_list.txt")
    
    parser.add_argument('--gene_num', type=int, default=40214)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--compile', action='store_true', default=False)
    
    args = parser.parse_args()

    main(args)
