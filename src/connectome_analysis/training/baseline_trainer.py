import torch
from src.connectome_analysis.training.lightning_modules import BaseLightningModule

class BaselineLightningModule(BaseLightningModule):
    def __init__(self, model, num_classes=2, learning_rate=1e-3):
        super().__init__(model, num_classes, learning_rate)
        self.save_hyperparameters(ignore=['model'])
        # For scikit-learn models, we handle training manually (via .fit())
        # and don't use PyTorch Lightning's automatic optimization.
        self.automatic_optimization = False

    def configure_optimizers(self):
        # Scikit-learn models have their own training logic (fit method)
        # Return an empty list, which is valid for PyTorch Lightning.
        return []

    # Remove on_train_epoch_start, training_step, on_train_epoch_end
    # The scikit-learn model will be fitted directly in ExperimentManager.

    def forward(self, x):
        # Ensure x is a PyTorch tensor for consistency if it comes from dataloader
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Scikit-learn's predict_proba expects numpy array
        x_np = x.cpu().numpy()
        
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(x_np)
            # Convert probabilities to PyTorch tensor, ensure it's on the correct device
            return torch.tensor(probas, dtype=torch.float32, device=self.device)
        elif hasattr(self.model, "decision_function"): # For SVMs without probability=True
            decision_values = self.model.decision_function(x_np)
            if decision_values.ndim == 1: # Binary classification
                 # Convert decision_values to "probabilities" (e.g., by sigmoid or just stack)
                 # For simplicity, let's create a two-column output [P(class_0), P(class_1)]
                 # This is a placeholder; proper calibration might be needed.
                 # A common trick is to use sigmoid on decision_function output for "probabilities"
                 # Or, if it's already somewhat like a logit, use it.
                 # For now, let's assume it's a score for the positive class.
                 # We need two columns for cross_entropy if num_classes is 2.
                 # This part is tricky and depends on the specific sklearn model.
                 # Let's assume positive class score and derive negative from it.
                 # This is a simplification.
                positive_scores = torch.tensor(decision_values, dtype=torch.float32, device=self.device).unsqueeze(-1)
                negative_scores = -positive_scores # Simplistic assumption
                return torch.cat([negative_scores, positive_scores], dim=1) # Returns "logits" like array
            else: # Multiclass
                return torch.tensor(decision_values, dtype=torch.float32, device=self.device)

        else: # Fallback for models like LogisticRegression that might just have predict
            # This case is problematic for cross_entropy loss.
            # For now, raise an error or return something that _common_step can handle.
            # Ideally, all classifiers should support predict_proba.
            raise NotImplementedError(f"Model {type(self.model).__name__} does not have predict_proba or decision_function.")

    def _common_step(self, batch, batch_idx):
        features, labels = batch
        # `self.forward` is designed to return probabilities (for models with predict_proba)
        # or logit-like scores (for models with decision_function).
        
        output = self.forward(features) # Shape: (batch_size, num_classes)

        # Check if the output looks like probabilities (e.g., from predict_proba)
        # A simple check: values are between 0 and 1.
        # More robustly, we assume if predict_proba was called, these are probabilities.
        # For LogisticRegression, self.forward calls predict_proba.
        
        if hasattr(self.model, "predict_proba"): # If output is probabilities
            # Add a small epsilon to prevent log(0)
            log_probas = torch.log(output + 1e-9)
            loss = torch.nn.functional.nll_loss(log_probas, labels)
            preds = torch.argmax(output, dim=1) # Argmax on probabilities is fine for predictions
            # For AUROC, BaseLightningModule expects logits. Here, 'output' is probabilities.
            # TorchMetrics AUROC can handle probabilities directly for binary cases (if passed to `preds` or `target` arg based on its API).
            # The BaseLightningModule's validation_step calls: self.auroc(logits[:, 1], targets)
            # This means it expects `logits` to be [N, C] and takes the prob of positive class.
            # So, returning `output` (probabilities) here is fine for AUROC.
            return loss, preds, labels, output 
        else: # If output is from decision_function (logit-like scores)
            # Assume `output` can be treated as logits.
            loss = torch.nn.functional.cross_entropy(output, labels)
            preds = torch.argmax(output, dim=1)
            return loss, preds, labels, output
