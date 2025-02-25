import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from model.spike import SpikeModel  # Import SPiKE Model
from utils.metrics import joint_accuracy  # Import evaluation metric

class SpikeLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = SpikeModel(config["model_args"])  # Initialize SPiKE model
        self.criterion = torch.nn.MSELoss()  # Use MSE loss for joint regression

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        clips, targets, _ = batch
        outputs = self(clips)  # Forward pass
        loss = self.criterion(outputs, targets)

        pck, _ = joint_accuracy(outputs, targets, self.hparams["threshold"])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_pck", pck.mean(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        clips, targets, _ = batch
        outputs = self(clips)
        loss = self.criterion(outputs, targets)

        pck, map_score = joint_accuracy(outputs, targets, self.hparams["threshold"])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_pck", pck.mean(), prog_bar=True)
        self.log("val_map", map_score.mean(), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        clips, targets, _ = batch
        outputs = self(clips)
        loss = self.criterion(outputs, targets)

        pck, map_score = joint_accuracy(outputs, targets, self.hparams["threshold"])
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_pck", pck.mean(), prog_bar=True)
        self.log("test_map", map_score.mean(), prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}