import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class PyTorchLightningPruningCallback(Callback):
    """
    PyTorch Lightning callback for Optuna pruning.
    Inherits from PyTorch Lightning's Callback and integrates with Optuna's pruning mechanism.
    """
    def __init__(self, trial: optuna.trial.Trial, monitor: str):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Called when the validation epoch ends.
        Checks if the trial should be pruned based on the monitored metric.
        """
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)
        if current_score is None:
            return

        self.trial.report(current_score.item(), trainer.current_epoch)
        if self.trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(trainer.current_epoch)
            raise optuna.exceptions.TrialPruned(message)
