from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets.itop import ITOP  # Import ITOP dataset

class ITOPDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        """Load datasets for training and validation."""
        self.train_dataset = ITOP(self.config["dataset_path"], train=True, use_valid_only=True)
        self.val_dataset = ITOP(self.config["dataset_path"], train=False, use_valid_only=True)

    def train_dataloader(self):
        print(f"🔍 Training dataset size: {len(self.train_dataset)}")
        return DataLoader(self.train_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=4)


    def val_dataloader(self):
        """Create DataLoader for validation."""
        return DataLoader(self.val_dataset, batch_size=self.config["batch_size"], shuffle=False, num_workers=4)

    def test_dataloader(self):
        """Create DataLoader for testing."""
        return DataLoader(self.val_dataset, batch_size=self.config["batch_size"], shuffle=False, num_workers=4)
