import pytorch_lightning as pl
import numpy as np # Added import
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger # Correct import for TensorBoardLogger
from src.connectome_analysis.training.baseline_trainer import BaselineLightningModule
from src.connectome_analysis.training.gnn_trainer import GNNLightningModule
from src.connectome_analysis.training.transformer_trainer import TransformerLightningModule
from src.connectome_analysis.training.lightning_modules import BaseLightningModule # For type hinting

class ExperimentManager:
    def __init__(self, config):
        self.config = config
        self.model_type = config.get("model_type", "baseline")
        self.model_name = config.get("model_name", "default_model")
        self.num_classes = config.get("num_classes", 2)
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.epochs = config.get("epochs", 10)
        self.batch_size = config.get("batch_size", 32)
        self.gpus = config.get("gpus", 0)
        self.log_dir = config.get("log_dir", "lightning_logs")
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")

    def _get_lightning_module(self, model) -> BaseLightningModule:
        if self.model_type == "baseline":
            return BaselineLightningModule(model, self.num_classes, self.learning_rate)
        elif self.model_type == "gnn":
            return GNNLightningModule(model, self.num_classes, self.learning_rate)
        elif self.model_type == "transformer":
            return TransformerLightningModule(model, self.num_classes, self.learning_rate)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _get_callbacks(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename=f"{self.model_name}-{{epoch:02d}}-{{val_loss:.2f}}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=self.config.get("early_stopping_patience", 5),
            mode="min",
        )
        return [checkpoint_callback, early_stopping_callback]

    def run_experiment(self, model, datamodule):
        lightning_module = self._get_lightning_module(model)
        callbacks = self._get_callbacks()

        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator="gpu" if self.gpus > 0 else "cpu",
            devices=self.gpus if self.gpus > 0 else "auto", # "auto" lets PL determine devices for CPU
            logger=TensorBoardLogger(self.log_dir, name=self.model_name),
            callbacks=callbacks,
            enable_checkpointing=True,
            log_every_n_steps=1,
            num_sanity_val_steps=0 # Disable sanity check for sklearn models
        )

        # For baseline models, fit the scikit-learn model before the trainer.fit call
        if self.model_type == "baseline":
            print(f"Fitting scikit-learn model {type(lightning_module.model).__name__} outside PL Trainer...")
            all_train_features = []
            all_train_labels = []
            for batch in datamodule.train_dataloader():
                features, labels = batch
                all_train_features.append(features.cpu().numpy())
                all_train_labels.append(labels.cpu().numpy())
            
            if all_train_features and all_train_labels:
                all_train_features_np = np.concatenate(all_train_features, axis=0)
                all_train_labels_np = np.concatenate(all_train_labels, axis=0)
                lightning_module.model.train(all_train_features_np, all_train_labels_np) # Changed .fit to .train
                print("Scikit-learn model fitting complete.")
            else:
                print("Warning: No training data found for scikit-learn model fitting.")

        # For baseline models, trainer.fit is essentially just running validation/test loops
        # as the model is already fitted. For other models, it performs actual training.
        trainer.fit(lightning_module, datamodule)
        test_results_metrics = trainer.test(lightning_module, datamodule)
        # The lightning_module instance now contains populated all_test_preds, all_test_targets, all_test_logits
        return test_results_metrics, lightning_module
