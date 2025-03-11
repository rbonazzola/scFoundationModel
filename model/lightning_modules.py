import pytorch_lightning as pl
from typing import List, Tuple, Union, Dict, Any, Optional, Callable

class LitSCModel(pl.LightningModule):

    def __init__(self, model, optimizer, loss):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def forward(self, x):    
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.model.loss(y_hat, y)
        return loss