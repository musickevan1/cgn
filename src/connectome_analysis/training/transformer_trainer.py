import torch
from src.connectome_analysis.training.lightning_modules import BaseLightningModule

class TransformerLightningModule(BaseLightningModule):
    def __init__(self, model, num_classes=2, learning_rate=1e-3):
        super().__init__(model, num_classes, learning_rate)
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        # Transformer models typically take a sequence of features
        # and potentially attention masks.
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        # Assuming batch contains (input_sequence, labels)
        input_sequence, labels = batch
        logits = self.forward(input_sequence)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels, logits
