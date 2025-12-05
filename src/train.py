import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from hydra.utils import instantiate
import torch
import os

from cgn.system import CGNSystem

@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)

    print(f"Working directory : {os.getcwd()}")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Instantiate DataModule
    print("Instantiating DataModule...")
    datamodule = instantiate(cfg.data)

    # Instantiate Model
    print("Instantiating Model...")
    model_arch = instantiate(cfg.model)

    # Wrap in Lightning System
    system = CGNSystem(model=model_arch)

    # Instantiate Trainer
    print("Instantiating Trainer...")
    trainer = instantiate(cfg.trainer)

    # Train
    print("Starting training...")
    trainer.fit(model=system, datamodule=datamodule)

    # Test
    print("Starting testing...")
    trainer.test(model=system, datamodule=datamodule)

if __name__ == "__main__":
    main()
