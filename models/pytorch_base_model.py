import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, Type
from abc import abstractmethod
from .base_model import BaseModel, NpArray

class PyTorchBaseModel(BaseModel):
    """
    A robust abstract base class for PyTorch time-series models.

    This class provides a complete, standardized training and prediction pipeline,
    including data loading, early stopping, and model persistence. Subclasses
    only need to implement the `build_model` method.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        device_str = self.config.get('device', 'cpu')
        if device_str == 'auto':
            device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if str(device_str).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use --device cpu or set experiment.device to auto.")
        self.device = torch.device(device_str)
        self.model: Optional[nn.Module] = None

        # Training hyperparameters
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.weight_decay = self.config.get('weight_decay', 1e-4)
        self.patience = self.config.get('patience', 10)
        self.num_workers = self.config.get('num_workers', 0)
        
        # Loss weighting for spike prediction
        self.weight_alpha = float(self.config.get('weight_alpha', 0))
        if not math.isfinite(self.weight_alpha) or abs(self.weight_alpha) > 20:
            raise ValueError('weight_alpha must be finite and between -20 and 20.')
        self.target_mean = None
        self.target_std = None

        print(f"PyTorch model '{self.name}' initialized on device: {self.device}")

    @abstractmethod
    def build_model(self, static_features: Optional[NpArray] = None) -> nn.Module:
        """
        Build and return the PyTorch model architecture.
        This method is called internally by the `train` method.

        Args:
            static_features (Optional[NpArray]): Static features of shape
                                                 (cells, static_features) that can be
                                                 used to construct the model.

        Returns:
            nn.Module: The constructed PyTorch model.
        """
        pass

    def _create_dataloader(self, X: NpArray, Y: Optional[NpArray] = None, shuffle: bool = False) -> DataLoader:
        """Creates a PyTorch DataLoader from NumPy arrays."""
        if len(X) == 0 or (Y is not None and len(X) != len(Y)):
            raise ValueError('Data must be nonempty and X/Y sample counts must match.')
        X_tensor = torch.from_numpy(X).float()
        
        if Y is not None:
            Y_tensor = torch.from_numpy(Y).float()
            dataset = TensorDataset(X_tensor, Y_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.config.get('random_seed', 42)),
            num_workers=self.num_workers,
            pin_memory=(self.device.type == 'cuda')
        )

    def _loss_terms(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Return error and weight sums, using statistics fitted on training targets."""
        if predictions.shape != targets.shape:
            raise RuntimeError(f'Shape mismatch: predictions {predictions.shape}, targets {targets.shape}')
        if self.weight_alpha == 0:
            weights = torch.ones_like(targets)
        else:
            if self.target_mean is None or self.target_std is None:
                raise RuntimeError('Fit loss statistics on training targets before using weighted loss.')
            normalized = ((targets - self.target_mean) / self.target_std).clamp(-3, 3)
            weights = torch.exp(self.weight_alpha * normalized)
        return (((predictions - targets) ** 2) * weights).sum(), weights.sum()

    def _weighted_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        errors, weights = self._loss_terms(predictions, targets)
        return errors / weights

    def train(
        self,
        X_train: NpArray, Y_train: NpArray,
        X_val: Optional[NpArray] = None, Y_val: Optional[NpArray] = None,
        static_features: Optional[NpArray] = None
    ) -> None:
        """Standard PyTorch training loop with early stopping."""
        print(f"🚀 Starting training for {self.name}...")
        
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError('epochs, batch_size, and patience must be positive.')
        if (X_val is None) != (Y_val is None):
            raise ValueError('Provide both X_val and Y_val, or neither.')
        if not np.isfinite(Y_train).all():
            raise ValueError('Training targets contain nonfinite values.')
        axes = tuple(range(Y_train.ndim - 1))
        self.target_mean = torch.as_tensor(Y_train.mean(axis=axes), dtype=torch.float32, device=self.device)
        self.target_std = torch.as_tensor(Y_train.std(axis=axes), dtype=torch.float32, device=self.device).clamp_min(1e-8)
        train_loader = self._create_dataloader(X_train, Y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, Y_val) if X_val is not None else None

        if self.model is None:
            self.model = self.build_model(static_features=static_features).to(self.device)
            print(f"Model built with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters.")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        training_start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            train_loss = self._train_epoch(train_loader, optimizer)
            
            val_loss_str = "N/A"
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                val_loss_str = f"{val_loss:.4f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"\n✋ Early stopping triggered at epoch {epoch + 1}.")
                        break
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1:3d}/{self.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss_str} | Time: {epoch_time:.1f}s")

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"✅ Restored best model with validation loss: {best_val_loss:.4f}")
        
        self.is_fitted = True
        total_time = time.time() - training_start_time
        print(f"🏁 Training finished in {total_time // 60:.0f}m {total_time % 60:.1f}s.")

    def predict(self, X_test: NpArray, static_features: Optional[NpArray] = None) -> NpArray:
        """Generate predictions for the test set."""
        super().predict(X_test, static_features)
        print(f"Generating predictions with {self.name}...")
        self.model.eval()
        test_loader = self._create_dataloader(X_test)
        
        all_predictions = []
        with torch.no_grad():
            for batch_X_tuple in test_loader:
                batch_X = batch_X_tuple[0].to(self.device)
                predictions = self.model(batch_X)
                all_predictions.append(predictions.cpu().numpy())
        
        return np.concatenate(all_predictions, axis=0)

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss, total_weight = 0.0, 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            optimizer.zero_grad()
            predictions = self.model(batch_X)
            
            if predictions.shape != batch_y.shape:
                raise RuntimeError(f"Shape mismatch: predictions {predictions.shape}, targets {batch_y.shape}")
            
            errors, weights = self._loss_terms(predictions, batch_y)
            loss = errors / weights
            if not torch.isfinite(loss):
                raise RuntimeError('Training loss is nonfinite.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += errors.detach().item()
            total_weight += weights.detach().item()
        return total_loss / total_weight

    def _validate_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss, total_weight = 0.0, 0.0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_X)
                errors, weights = self._loss_terms(predictions, batch_y)
                total_loss += errors.item()
                total_weight += weights.item()
        val_loss = total_loss / total_weight
        if not math.isfinite(val_loss):
            raise RuntimeError('Validation loss is nonfinite.')
        return val_loss

    def _checkpoint_extra(self) -> dict:
        return {}

    def _restore_checkpoint_extra(self, extra: dict) -> None:
        pass
    
    def save(self, file_path: Path) -> None:
        """Saves the model's state dictionary and configuration."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save a model that has not been trained.")
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'loss_target_mean': self.target_mean,
            'loss_target_std': self.target_std,
            'extra': self._checkpoint_extra(),
        }, file_path)
        print(f"💾 Model '{self.name}' saved to {file_path}")

    @classmethod
    def load(cls: Type['PyTorchBaseModel'], file_path: Path, device: str = 'cpu') -> 'PyTorchBaseModel':
        """Loads a model, re-instantiating it from its configuration."""
        checkpoint = torch.load(file_path, map_location=device, weights_only=False)
        config = {**checkpoint['config'], 'device': device, '_loading_checkpoint': True}
        
        # Re-create the model instance with the saved config
        instance = cls(config)
        instance._restore_checkpoint_extra(checkpoint.get('extra', {}))
        instance.target_mean = checkpoint.get('loss_target_mean')
        instance.target_std = checkpoint.get('loss_target_std')
        
        # Build the model architecture and load the saved weights
        instance.model = instance.build_model().to(instance.device)
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.is_fitted = True
        
        print(f"✅ Model '{instance.name}' loaded from {file_path}")
        return instance
