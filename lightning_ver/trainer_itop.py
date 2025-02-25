import pytorch_lightning as pl
import torch
from spike_lightning import SpikeLightning
from itop_data_module import ITOPDataModule
from utils.config_utils import load_config

def main():
    """Train SPiKE using PyTorch Lightning."""
    # Load config file
    config = load_config("ITOP-SIDE/1")  

    # Initialize model & data
    model = SpikeLightning(config)
    data_module = ITOPDataModule(config)

    # Define trainer with GPU support
    trainer = pl.Trainer(max_epochs=config["epochs"], gpus=1, log_every_n_steps=10)

    # Train the model
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
