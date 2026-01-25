"""
dgm_utils - Utilities for Deep Generative Models course.

This package provides utilities for training, visualizing, and evaluating
deep generative models including VAEs, GANs, flow models, and diffusion models.
"""

from .training import train_model, train_adversarial
from .data import load_dataset, LabeledDataset
from .visualize import (
    plot_training_curves,
    show_samples,
    visualize_images,
    visualize_2d_data,
    visualize_2d_samples,
    visualize_2d_densities,
)
from .model import BaseModel


__all__ = [
    # Training
    "train_model",
    "train_adversarial",
    # Data
    "load_dataset",
    "LabeledDataset",
    # Visualization
    "plot_training_curves",
    "show_samples",
    "visualize_images",
    "visualize_2d_data",
    "visualize_2d_samples",
    "visualize_2d_densities",
    # Model
    "BaseModel",
]

__version__ = "0.1.1"
