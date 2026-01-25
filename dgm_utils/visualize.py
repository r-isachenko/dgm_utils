"""Visualization utilities for deep generative models."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

import torch


TICKS_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
LABEL_FONT_SIZE = 14
TITLE_FONT_SIZE = 16


def plot_training_curves(
    epochs: int,
    train_losses: Dict[str, List[float]],
    test_losses: Optional[Dict[str, List[float]]],
    logscale_y: bool = False,
    logscale_x: bool = False,
) -> None:
    """
    Plot training and test loss curves.
    
    Args:
        epochs: Number of epochs trained.
        train_losses: Dictionary mapping loss names to lists of training losses.
        test_losses: Optional dictionary mapping loss names to lists of test losses.
        logscale_y: Whether to use log scale for y-axis.
        logscale_x: Whether to use log scale for x-axis.
    """
    plt.figure()

    n_train = len(train_losses[list(train_losses.keys())[0]])
    x_train = np.linspace(0, epochs, n_train)
    for key, value in train_losses.items():
        plt.plot(x_train, value, label=key + "_train")

    if test_losses is not None:
        n_test = len(test_losses[list(test_losses.keys())[0]])
        x_test = np.arange(n_test)
        for key, value in test_losses.items():
            plt.plot(x_test, value, label=key + "_test")

    if logscale_y:
        plt.semilogy()

    if logscale_x:
        plt.semilogx()

    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.xlabel("Epoch", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Loss", fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    plt.grid()
    plt.show()


def show_samples(
    samples: Union[np.ndarray, torch.Tensor],
    title: str,
    figsize: Optional[Tuple[int, int]] = None,
    nrow: Optional[int] = None,
    normalize: bool = False
) -> None:
    """
    Display a grid of image samples.
    
    Args:
        samples: Image samples with shape (N, C, H, W).
        title: Title for the plot.
        figsize: Optional figure size as (width, height).
        nrow: Number of images per row. Defaults to sqrt(N).
        normalize: Whether to normalize images for display.
    """
    if isinstance(samples, np.ndarray):
        samples = torch.tensor(samples)
    if nrow is None:
        nrow = int(np.sqrt(len(samples)))

    grid_samples = make_grid(samples, nrow=nrow, normalize=normalize, scale_each=True)
    grid_img = grid_samples.permute(1, 2, 0)

    plt.figure(figsize=(6, 6) if figsize is None else figsize)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.imshow(grid_img)
    plt.axis("off")
    plt.show()


def visualize_images(data: np.ndarray, title: str) -> None:
    """
    Display a random sample of 100 images from a dataset.
    
    Args:
        data: Image dataset with shape (N, C, H, W).
        title: Title for the plot.
    """
    idxs = np.random.choice(len(data), replace=False, size=(100,))
    images = data[idxs]
    show_samples(images, title)


def visualize_2d_data(
    train_data: np.ndarray,
    test_data: np.ndarray,
    train_labels: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    s: int = 10
) -> None:
    """
    Visualize 2D train and test data side by side.
    
    Args:
        train_data: Training data with shape (N, 2).
        test_data: Test data with shape (N, 2).
        train_labels: Optional labels for coloring train points.
        test_labels: Optional labels for coloring test points.
        s: Marker size for scatter plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("train", fontsize=TITLE_FONT_SIZE)
    ax1.scatter(train_data[:, 0], train_data[:, 1], s=s, c=train_labels)
    ax1.tick_params(labelsize=LABEL_FONT_SIZE)
    ax2.set_title("test", fontsize=TITLE_FONT_SIZE)
    ax2.scatter(test_data[:, 0], test_data[:, 1], s=s, c=test_labels)
    ax2.tick_params(labelsize=LABEL_FONT_SIZE)
    plt.show()


def visualize_2d_samples(
    data: np.ndarray,
    title: str,
    labels: Optional[np.ndarray] = None,
    xlabel: Optional[str] = "x1",
    ylabel: Optional[str] = "x2",
    s: int = 10
) -> None:
    """
    Visualize 2D sample data.
    
    Args:
        data: Sample data with shape (N, 2).
        title: Title for the plot.
        labels: Optional labels for coloring points.
        xlabel: Label for x-axis. Set to None to hide.
        ylabel: Label for y-axis. Set to None to hide.
        s: Marker size for scatter plot.
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], s=s, c=labels)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    plt.show()


def visualize_2d_densities(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    densities: np.ndarray,
    title: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Visualize 2D density estimates as a heatmap.
    
    Args:
        x_grid: X coordinates of the grid.
        y_grid: Y coordinates of the grid.
        densities: Density values at each grid point.
        title: Title for the plot.
        xlabel: Optional label for x-axis.
        ylabel: Optional label for y-axis.
    """
    densities = densities.reshape([y_grid.shape[0], y_grid.shape[1]])
    plt.figure(figsize=(5, 5))
    plt.pcolor(x_grid, y_grid, densities)

    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    plt.show()
