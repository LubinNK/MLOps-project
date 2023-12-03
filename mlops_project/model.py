from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
from conf.config import Config


# MODEL FOR TRAIN
class CNN_new(pl.LightningModule):
    def __init__(self, conf: Config):
        super(CNN_new, self).__init__()
        k = conf.parameters.k

        self.loss_fn = nn.CrossEntropyLoss()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 6 * k, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 28 -> 14
            nn.Conv2d(6 * k, 16 * k, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(1, -1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * k, 10),
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.fc(x)
        return x

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        X, y = batch
        y_preds = self(X)
        loss = self.loss_fn(y_preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        labels_hat = torch.argmax(y_pred, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val_acc", val_acc, on_step=True, on_epoch=True, prog_bar=False)
        return {"val_loss": loss, "val_acc": val_acc}

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        labels_hat = torch.argmax(y_pred, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("test_acc", val_acc, on_step=True, on_epoch=True, prog_bar=False)
        return {"val_loss": loss, "val_acc": val_acc}

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters())
        return [optimizer]

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
