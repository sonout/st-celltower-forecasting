from abc import ABC, abstractmethod
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Type
from typing_extensions import TypeAlias

# Create a TypeAlias for NumPy arrays to make signatures cleaner
NpArray: TypeAlias = np.ndarray

class BaseModel(ABC):
    """
    Abstract base class for all telecom KPI prediction models.

    This class defines a consistent, robust interface for training, predicting,
    saving, and loading models that operate on pre-sequenced NumPy arrays.

    Attributes:
        config (Dict[str, Any]): The complete configuration for the model.
        is_fitted (bool): A flag indicating if the model has been trained.
        n_targets (int): The number of target variables to predict.
        output_steps (int): The number of future time steps to predict.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the model, validating the presence of essential metadata.
        """
        if not config:
            raise ValueError("A configuration dictionary must be provided.")
        self.config = config
        self.is_fitted = False

        # Store essential metadata from the config for easy access
        self.n_targets = self.config.get('n_targets')
        self.output_steps = self.config.get('output_steps')
        if self.n_targets is None or self.output_steps is None:
            raise KeyError("Config must contain 'n_targets' and 'output_steps'.")

    @property
    def name(self) -> str:
        """Returns the model's class name for reporting and logging."""
        return self.__class__.__name__

    @abstractmethod
    def train(
        self,
        X_train: NpArray,
        Y_train: NpArray,
        X_val: Optional[NpArray] = None,
        Y_val: Optional[NpArray] = None,
        static_features: Optional[NpArray] = None
    ) -> None:
        """
        Train the model on the provided sequenced data.

        Args:
            X_train (np.ndarray): Training input data of shape
                                  (sequences, input_steps, cells, features).
            Y_train (np.ndarray): Training target data of shape
                                  (sequences, output_steps, cells, targets).
            X_val (Optional[np.ndarray]): Validation input data.
            Y_val (Optional[np.ndarray]): Validation target data.
            static_features (Optional[np.ndarray]): Static features of shape
                                                   (cells, static_features).
        """
        pass

    @abstractmethod
    def predict(
        self,
        X_test: NpArray,
        static_features: Optional[NpArray] = None
    ) -> NpArray:
        """
        Predict future KPI values using the test input data.

        Args:
            X_test (np.ndarray): Test input data of shape
                                 (sequences, input_steps, cells, features).
            static_features (Optional[np.ndarray]): Static features of shape
                                                   (cells, static_features).
        
        Returns:
            np.ndarray: Predictions of shape (sequences, output_steps, cells, targets).
        """
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.name}' has not been trained yet. Call train() before predict().")
        pass

    def save(self, file_path: Path) -> None:
        """
        Saves the trained model instance to a file using pickle.

        Args:
            file_path (Path): The path to save the model file to (e.g., 'model.pkl').
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save a model that has not been trained.")
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Model '{self.name}' saved to {file_path}")

    @classmethod
    def load(cls: Type['BaseModel'], file_path: Path) -> 'BaseModel':
        """
        Loads a model instance from a file.

        Args:
            file_path (Path): The path to the saved model file.

        Returns:
            BaseModel: An instance of the loaded model.
        """
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is type {type(model)}, but expected {cls}.")
        print(f"✅ Model '{model.name}' loaded from {file_path}")
        return model