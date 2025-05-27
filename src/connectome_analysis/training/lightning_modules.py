import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score, Precision, Recall, ConfusionMatrix
from typing import Any, Optional # Added Any, Optional

class BaseLightningModule(pl.LightningModule):
    def __init__(self, model, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.save_hyperparameters(ignore=['model'])

        # Metrics
        self.accuracy = Accuracy(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.auroc = AUROC(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.f1_score = F1Score(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.precision = Precision(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.recall = Recall(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(task="binary" if num_classes == 2 else "multiclass", num_classes=num_classes)

        # Store predictions and targets for aggregation in test_epoch_end
        self.all_test_preds = []
        self.all_test_targets = []
        self.all_test_logits = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y, logits

    def training_step(self, batch, batch_idx) -> Optional[torch.Tensor]: # Added return type hint
        loss, preds, targets, _ = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.accuracy(preds, targets)
        self.log('train_acc', self.accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets, logits = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.accuracy(preds, targets)
        self.log('val_acc', self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        if self.num_classes == 2:
            self.auroc(logits[:, 1], targets)
        else:
            self.auroc(logits, targets)
        self.log('val_auroc', self.auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, targets, logits = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        self.all_test_preds.append(preds)
        self.all_test_targets.append(targets)
        self.all_test_logits.append(logits) # Store logits for AUC and prob calculation

        return loss

    def on_test_epoch_end(self):
        # Concatenate all predictions, targets, and logits from all test steps
        all_preds = torch.cat(self.all_test_preds)
        all_targets = torch.cat(self.all_test_targets)
        all_logits = torch.cat(self.all_test_logits)

        # Compute metrics on the concatenated tensors
        self.accuracy(all_preds, all_targets)
        self.log('test_acc', self.accuracy, on_step=False, on_epoch=True)
        
        if self.num_classes == 2:
            self.auroc(all_logits[:, 1], all_targets)
        else:
            self.auroc(all_logits, all_targets)
        self.log('test_auroc', self.auroc, on_step=False, on_epoch=True)
        
        self.f1_score(all_preds, all_targets)
        self.log('test_f1', self.f1_score, on_step=False, on_epoch=True)
        
        self.precision(all_preds, all_targets)
        self.log('test_precision', self.precision, on_step=False, on_epoch=True)
        
        self.recall(all_preds, all_targets)
        self.log('test_recall', self.recall, on_step=False, on_epoch=True)
        
        # cm = self.confusion_matrix(all_preds, all_targets) # This updates the metric state
        # self.log('test_confusion_matrix', cm) # Logging the metric object might not be what's intended.
        # Typically, you'd compute and log the confusion matrix values if needed, or log the metric object if it's supported.
        # For now, let's assume logging the metric object is fine or handle it if it causes issues later.
        # If direct logging of the ConfusionMatrix object is problematic, one might log its .compute() result.
        # For simplicity, let's keep it as is unless it errors.

        # Clear the stored lists for the next test run
        self.all_test_preds.clear()
        self.all_test_targets.clear()
        self.all_test_logits.clear()
        
        # Reset metrics
        self.accuracy.reset()
        self.auroc.reset()
        self.f1_score.reset()
        self.precision.reset()
        self.recall.reset()
        self.confusion_matrix.reset()

    def configure_optimizers(self) -> Any: # Changed return type hint to Any
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
