import optuna
import pytorch_lightning as pl
from typing import Dict, Any, Callable, Optional
from src.connectome_analysis.data.data_modules import ConnectomeDataModule
from src.connectome_analysis.training.experiment_manager import ExperimentManager
from src.connectome_analysis.optimization.pruning_callbacks import PyTorchLightningPruningCallback # Assuming this will be created
from pytorch_lightning.loggers import TensorBoardLogger # Correct import for TensorBoardLogger

class OptunaOptimizer:
    """
    Manages hyperparameter optimization using Optuna for PyTorch Lightning models.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the OptunaOptimizer.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - 'study_name' (str): Name of the Optuna study.
                - 'n_trials' (int): Number of trials for optimization.
                - 'model_type' (str): Type of model ('baseline', 'gnn', 'transformer').
                - 'model_name' (str): Base name for the model.
                - 'num_classes' (int): Number of output classes.
                - 'data_module_params' (Dict): Parameters for the ConnectomeDataModule.
                - 'trainer_params' (Dict): Base parameters for the PyTorch Lightning Trainer.
                - 'model_builder' (callable): Function that builds and returns a model instance.
                - 'search_space' (Dict): Dictionary defining the hyperparameter search space.
        """
        self.config = config
        self.study_name = config.get('study_name', 'optuna_study')
        self.n_trials = config.get('n_trials', 50)
        self.model_type = config.get('model_type', 'baseline')
        self.model_name = config.get('model_name', 'default_model')
        self.num_classes = config.get('num_classes', 2)
        self.data_module_params = config.get('data_module_params', {})
        self.trainer_params = config.get('trainer_params', {})
        self.model_builder = config.get('model_builder')
        self.search_space = config.get('search_space', {})

        if self.model_builder is None:
            raise ValueError("model_builder function must be provided in the config.")
        # Type assertion to help Pylance understand that model_builder is callable
        assert callable(self.model_builder), "model_builder must be a callable function."
        if not self.search_space:
            raise ValueError("search_space must be defined in the config.")

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggests hyperparameters based on the defined search space.
        """
        hparams = {}
        for param_name, param_info in self.search_space.items():
            param_type = param_info.get("type")
            if param_type == "float":
                hparams[param_name] = trial.suggest_float(
                    param_name,
                    param_info["low"],
                    param_info["high"],
                    log=param_info.get("log", False)
                )
            elif param_type == "int":
                hparams[param_name] = trial.suggest_int(
                    param_name,
                    param_info["low"],
                    param_info["high"],
                    log=param_info.get("log", False)
                )
            elif param_type == "categorical":
                hparams[param_name] = trial.suggest_categorical(
                    param_name,
                    param_info["choices"]
                )
            # Add more types as needed (e.g., uniform, discrete_uniform)
        return hparams

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function to train and evaluate a model.
        """
        # Suggest hyperparameters for the current trial
        trial_hparams = self._suggest_hyperparameters(trial)

        # Update model and trainer parameters with trial-specific hyperparameters
        model_params = self.config.get('model_params', {}).copy()
        trainer_params = self.trainer_params.copy()
        
        # Apply trial hyperparameters. This assumes a flat structure for now.
        # More complex mapping might be needed for nested parameters.
        for k, v in trial_hparams.items():
            if k in model_params:
                model_params[k] = v
            elif k in trainer_params:
                trainer_params[k] = v
            elif k == "learning_rate": # Special handling for learning rate
                trainer_params["learning_rate"] = v
            elif k == "batch_size": # Special handling for batch size
                self.data_module_params["batch_size"] = v
            # Add more specific mappings if needed

        # Build model for the current trial
        # Ensure model_builder is callable, reinforcing __init__ checks for Pylance
        current_model_builder = self.model_builder
        if not callable(current_model_builder):
            # This path should ideally not be reached if __init__ completed successfully.
            raise TypeError(
                f"self.model_builder is not callable in objective method. Type: {type(current_model_builder)}. "
                "This indicates an issue with OptunaOptimizer initialization or state."
            )
        model_instance = current_model_builder(**model_params) # Pass suggested model params

        # Initialize DataModule (assuming it's already set up with full data)
        # For Optuna, we typically use a fixed train/validation split or a single fold of CV
        # Here, we'll assume the ConnectomeDataModule is initialized with the full dataset
        # and its setup() method will handle the train/val split for the trial.
        # For simplicity, we'll use the data_module_params from config.
        # In a real scenario, you might pass specific train/val indices to the DataModule.
        
        # Create a new DataModule instance for each trial to ensure isolation
        # This assumes ConnectomeDataModule can handle internal train/val splits
        # or that we pass pre-split data to it. For now, we'll use the config.
        data_module = ConnectomeDataModule(
            features=self.config['features'], # Assuming features, labels, site_labels are passed in config
            labels=self.config['labels'],
            site_labels=self.config.get('site_labels'),
            **self.data_module_params
        )
        data_module.setup() # Call setup to create train/val/test datasets

        # Initialize ExperimentManager for the current trial
        experiment_config = {
            "model_type": self.model_type,
            "model_name": f"{self.model_name}_trial_{trial.number}",
            "num_classes": self.num_classes,
            "learning_rate": trainer_params.get("learning_rate", 1e-3), # Use trial LR if suggested
            "epochs": trainer_params.get("epochs", 10),
            "batch_size": self.data_module_params.get("batch_size", 32), # Use trial batch size if suggested
            "gpus": trainer_params.get("gpus", 0),
            "log_dir": trainer_params.get("log_dir", "lightning_logs"),
            "checkpoint_dir": trainer_params.get("checkpoint_dir", "checkpoints"),
            **trainer_params # Override with specific trainer params
        }
        experiment_manager = ExperimentManager(experiment_config)

        # Add Optuna pruning callback
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        
        # Run training and evaluation
        try:
            # Modify ExperimentManager to accept callbacks
            # For now, we'll directly create a Trainer and pass callbacks
            trainer = pl.Trainer(
                max_epochs=experiment_config["epochs"],
                accelerator="gpu" if experiment_config["gpus"] > 0 else "cpu",
                devices=experiment_config["gpus"] if experiment_config["gpus"] > 0 else "auto",
                logger=TensorBoardLogger(experiment_config["log_dir"], name=experiment_config["model_name"]),
                callbacks=[pruning_callback], # Pass pruning callback
                enable_checkpointing=False, # Disable checkpointing for trials to save space
                log_every_n_steps=1,
            )
            
            lightning_module = experiment_manager._get_lightning_module(model_instance)
            trainer.fit(lightning_module, data_module)

            # Evaluate on validation set
            eval_results = trainer.validate(lightning_module, data_module.val_dataloader())
            
            # Assuming validation_epoch_end logs 'val_auroc'
            # Optuna typically minimizes, so return 1 - AUROC for maximization
            val_auroc = eval_results[0].get('val_auroc', 0.0)
            return val_auroc # Optuna direction will be 'maximize'

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed due to error: {e}")
            # Return a very low value for failed trials if not pruned
            return -1.0

    def optimize(self) -> optuna.Study:
        """
        Runs the hyperparameter optimization study.
        """
        study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize', # Maximize validation AUROC
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            )
        )
        study.optimize(self.objective, n_trials=self.n_trials, callbacks=[]) # Callbacks handled by trainer
        return study
