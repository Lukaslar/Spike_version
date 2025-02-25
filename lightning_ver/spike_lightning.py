import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from model.spike import SPiKE  # Import SPiKE model
from utils.metrics import joint_accuracy  # Import evaluation metric

class SpikeLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)  

        # Unpack model_args correctly
        self.model = SPiKE(
            radius=config["model_args"]["radius"],
            nsamples=config["model_args"]["nsamples"],
            spatial_stride=config["model_args"]["spatial_stride"],
            dim=config["model_args"]["dim"],
            depth=config["model_args"]["depth"],
            heads=config["model_args"]["heads"],
            dim_head=config["model_args"]["dim_head"],
            mlp_dim=config["model_args"]["mlp_dim"],
            num_coord_joints=config["num_coord_joints"],  # This is usually separate
            dropout1=config["model_args"].get("dropout1", 0.0),  # Default to 0.0 if missing
            dropout2=config["model_args"].get("dropout2", 0.0),
        )
        self.criterion = torch.nn.MSELoss()  # Mean Squared Error loss for joint regression

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step for SPiKE model."""
        clips, targets, _ = batch
        outputs = self(clips)  # Forward pass
        loss = self.criterion(outputs, targets)

        pck, _ = joint_accuracy(outputs, targets, self.hparams["threshold"])
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_pck", pck.mean(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step to evaluate model performance."""
        clips, targets, _ = batch
        outputs = self(clips)
        loss = self.criterion(outputs, targets)

        pck, map_score = joint_accuracy(outputs, targets, self.hparams["threshold"])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_pck", pck.mean(), prog_bar=True)
        self.log("val_map", map_score.mean(), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step for evaluating SPiKE model."""
        clips, targets, _ = batch
        outputs = self(clips)
        loss = self.criterion(outputs, targets)

        pck, map_score = joint_accuracy(outputs, targets, self.hparams["threshold"])
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_pck", pck.mean(), prog_bar=True)
        self.log("test_map", map_score.mean(), prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Define optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
