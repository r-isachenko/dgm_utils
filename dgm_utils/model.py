"""Base model class for deep generative models."""

from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for generative models.
    
    All generative models should inherit from this class and implement
    the forward, loss, and sample methods.
    
    Examples:
        >>> class MyModel(BaseModel):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.net = nn.Linear(10, 10)
        ...     
        ...     def forward(self, x):
        ...         return self.net(x)
        ...     
        ...     def loss(self, x):
        ...         return {"total_loss": ((x - self(x)) ** 2).mean()}
        ...     
        ...     def sample(self, n):
        ...         return torch.randn(n, 10).numpy()
    """
    
    def __init__(self) -> None:
        super().__init__()
        
    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor.
        
        Returns:
            Output tensor.
        """
        pass

    @abstractmethod
    def loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the loss for training.
        
        Args:
            x: Input tensor.
        
        Returns:
            Dictionary containing at least "total_loss" key with the main loss
            to backpropagate, and optionally other loss components for logging.
        """
        pass

    @abstractmethod
    @torch.no_grad()
    def sample(self, n: int) -> Union[np.ndarray, torch.Tensor]:
        """
        Generate samples from the model.
        
        Args:
            n: Number of samples to generate.
        
        Returns:
            Array or tensor of generated samples.
        """
        pass
