import transformers
import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

import data
from data.dataset import scRNADataset

from model.performer.performer import PerformerLM
from utils import save_ckpt
import scanpy as sc

from torch.optim import Adam

from masking import data_mask

from sklearn.model_selection import train_test_split
SEED = 42
BATCH_SIZE = 1


def main(args):

    SEQ_LEN = args.gene_num + 1
    LEARNING_RATE = args.lr

    if args.data_path.endswith("gz"):
        import gzip
        with gzip.open(args.data_path) as f:
            data = sc.read_h5ad(f).X
    else:
        data = sc.read_h5ad(args.data_path).X
    data_train, data_val = train_test_split(data, test_size=0.05,random_state=SEED)
    
    train_dataset = scRNADataset(data_train)
    val_dataset = scRNADataset(data_val)
    
    # train_sampler = DistributedSampler(train_dataset)
    # val_sampler = SequentialDistributedSampler(val_dataset, batch_size=BATCH_SIZE, world_size=world_size)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    N_CLASSES = train_dataset.N_CLASSES
    
    model = PerformerLM(
        num_tokens = N_CLASSES,
        dim = 200,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        local_attn_heads = 0,
        g2v_position_emb = None
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.compile(model)
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    #### TRAINING LOOP
    ######### model, train_loader, val_loader, data_mask, 
    ######### GRADIENT_ACCUMULATION, optimizer,
    
    PAD_TOKEN_ID = N_CLASSES - 1
    EPOCHS = args.epochs
    VALIDATE_EVERY = 5
    GRADIENT_ACCUMULATION = 10
    
    # scheduler, etc.
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='mean')
    softmax = nn.Softmax(dim=-1)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        cum_acc = 0.0
    
        for index, data in enumerate(train_loader):
            index += 1
            data = data.to(device)
            data, labels = data_mask(data)
    
            # Forward
            logits = model(data)
            loss = loss_fn(logits.transpose(1, 2), labels) / GRADIENT_ACCUMULATION
    
            # Backward
            loss.backward()
    
            # Acumulación de gradientes
            if index % GRADIENT_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e2)
                optimizer.step()
                optimizer.zero_grad()
    
            running_loss += loss.item()
    
            # Cálculo de accuracy “token a token”
            final = softmax(logits)[..., 1:-1]  # quitamos primera y última col.
            final = final.argmax(dim=-1) + 1    # desplazamos para ajustar IDs
            pred_num = (labels != PAD_TOKEN_ID).sum(dim=-1)
            correct_num = ((labels != PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
    
        epoch_loss = running_loss / index
        epoch_acc  = 100 * (cum_acc / index)
    
        print(f'==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f} | '
              f'Accuracy: {epoch_acc:.4f}%  ==')
    
        scheduler.step()
    
        # Proceso de validación
        if epoch % VALIDATE_EVERY == 0:
            model.eval()
            running_loss_val = 0.0
            predictions = []
            truths = []
    
            with torch.no_grad():
                for index, data in enumerate(val_loader):
                    index += 1
                    data = data.to(device)
                    data, labels = data_mask(data)
    
                    logits = model(data)
                    loss = loss_fn(logits.transpose(1, 2), labels)
                    running_loss_val += loss.item()
    
                    final = softmax(logits)[..., 1:-1]
                    final = final.argmax(dim=-1) + 1
                    predictions.append(final)
                    truths.append(labels)
    
            # Concatenamos los tensores para medir la exactitud global
            predictions = torch.cat(predictions, dim=0)
            truths      = torch.cat(truths, dim=0)
    
            correct_num = ((truths != PAD_TOKEN_ID) * (predictions == truths)).sum().item()
            val_num     = (truths != PAD_TOKEN_ID).sum().item()
    
            val_loss = running_loss_val / index
            val_acc  = 100.0 * correct_num / val_num
    
            print(f'==  Epoch: {epoch} | Validation Loss: {val_loss:.6f} | '
                  f'Accuracy: {val_acc:.4f}%  ==')
    
        save_ckpt(epoch, model, optimizer, scheduler, epoch_loss, model_name, ckpt_dir)




if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--data_path', type=str, help='Path to the data', default="./data/CRA004476_transformed.h5ad")
    # parser.add_argument('--model', type=str, help='Model to use')
    parser.add_argument('--gene_num', type=int, default=40214)
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=100)
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate', default=3e-4)
    parser.add_argument('--seed', type=int, help='Random seed')
    args = parser.parse_args()

    main(args)