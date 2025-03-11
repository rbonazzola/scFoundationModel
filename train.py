import transformers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from model import Model
from data.dataset import scRNADataset


if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--data', type=str, help='Path to the data')
    parser.add_argument('--model', type=str, help='Model to use')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    args = parser.parse_args()

    
