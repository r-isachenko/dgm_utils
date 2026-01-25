"""Training utilities for deep generative models."""

from collections import defaultdict
from IPython.display import clear_output
from typing import Dict, List, Optional, Union
from contextlib import nullcontext
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from tqdm.auto import tqdm

from .visualize import show_samples, visualize_2d_samples, plot_training_curves
from .model import BaseModel


def train_epoch(
    epoch: int,
    model: BaseModel,
    train_loader: DataLoader,
    optimizer: Optimizer,
    conditional: bool = False,
    device: str = "cpu",
    loss_key: str = "total_loss",
    use_amp: bool = False,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, List[float]]:
    """
    Train the model for one epoch.
    
    Args:
        epoch: Current epoch number (for logging).
        model: The model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for updating model parameters.
        conditional: Whether the model is conditional (expects (x, y) batches).
        device: Device to train on ("cpu" or "cuda").
        loss_key: Key in the loss dict to use for backpropagation.
        use_amp: Whether to use automatic mixed precision.
        scaler: GradScaler for AMP training.
    
    Returns:
        Dictionary mapping loss names to lists of per-batch loss values.
    """
    model.train()

    stats: Dict[str, List[float]] = defaultdict(list)
    for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
        if conditional:
            x, y = batch
            x, y = x.to(device), y.to(device)
        else:
            x, y = batch.to(device), None
        optimizer.zero_grad()
        ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with ctx:
            losses = model.loss(x, y) if y is not None else model.loss(x)
            loss = losses[loss_key]

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        for k, v in losses.items():
            stats[k].append(v.item())

    return stats


def eval_model(
    epoch: int,
    model: BaseModel,
    data_loader: DataLoader,
    conditional: bool = False,
    device: str = "cpu",
    use_amp: bool = False,
) -> Dict[str, float]:
    """
    Evaluate the model on a dataset.
    
    Args:
        epoch: Current epoch number (for logging).
        model: The model to evaluate.
        data_loader: DataLoader for evaluation data.
        conditional: Whether the model is conditional (expects (x, y) batches).
        device: Device to evaluate on ("cpu" or "cuda").
        use_amp: Whether to use automatic mixed precision.
    
    Returns:
        Dictionary mapping loss names to average loss values over the dataset.
    """
    model.eval()
    stats: Dict[str, float] = defaultdict(float)
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating epoch {epoch}"):
            if conditional:
                x, y = batch
                x, y = x.to(device), y.to(device)
            else:
                x, y = batch.to(device), None
            ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()
            with ctx:
                losses = model.loss(x, y) if y is not None else model.loss(x)
            for k, v in losses.items():
                stats[k] += v.item() * x.shape[0]

        for k in stats.keys():
            stats[k] /= len(data_loader.dataset)  # type: ignore
    return stats


def check_samples_is_2d(samples: np.ndarray) -> bool:
    """Check if samples are 2D points (for visualization)."""
    shape = samples.shape
    return len(shape) == 2 and shape[1] == 2


def train_model(
    model: BaseModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    optimizer: Optimizer,
    scheduler: Optional[LRScheduler] = None,
    conditional: bool = False,
    loss_key: str = "total_loss",
    n_samples: int = 100,
    visualize_samples: bool = True,
    logscale_y: bool = False,
    logscale_x: bool = False,
    device: str = "cpu",
    use_amp: bool = False,
) -> None:
    """
    Train a generative model with visualization.
    
    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
        epochs: Number of training epochs.
        optimizer: Optimizer for updating model parameters.
        scheduler: Optional learning rate scheduler.
        conditional: Whether the model is conditional (expects (x, y) batches).
        loss_key: Key in the loss dict to use for backpropagation.
        n_samples: Number of samples to generate for visualization.
        visualize_samples: Whether to visualize generated samples during training.
        logscale_y: Whether to use log scale for y-axis in loss plots.
        logscale_x: Whether to use log scale for x-axis in loss plots.
        device: Device to train on ("cpu" or "cuda").
        use_amp: Whether to use automatic mixed precision.
    """
    train_losses: Dict[str, List[float]] = defaultdict(list)
    test_losses: Dict[str, List[float]] = defaultdict(list)
    model = model.to(device)
    print("Start of the training")

    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    test_loss = eval_model(0, model, test_loader, conditional, device)
    for k in test_loss.keys():
        test_losses[k].append(test_loss[k])

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            epoch, model, train_loader, optimizer, conditional, device, loss_key, use_amp, scaler
        )
        if scheduler is not None:
            scheduler.step()
        test_loss = eval_model(epoch, model, test_loader, conditional, device, use_amp)

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
        for k in test_loss.keys():
            test_losses[k].append(test_loss[k])

        epoch_loss = np.mean(train_loss[loss_key])
        if visualize_samples:
            with torch.no_grad():
                samples = model.sample(n_samples)
                if isinstance(samples, torch.Tensor):
                    samples = samples.cpu().detach().numpy()

            clear_output(wait=True)
            title = f"Samples, epoch: {epoch}, {loss_key}: {epoch_loss:.3f}"
            if check_samples_is_2d(samples):
                visualize_2d_samples(samples, title=title)
            else:
                show_samples(samples, title=title)
            plot_training_curves(epoch, train_losses, test_losses, logscale_y, logscale_x)
        else:
            clear_output(wait=True)
            print(f"Epoch: {epoch}, loss: {epoch_loss}")
            plot_training_curves(epoch, train_losses, test_losses, logscale_y, logscale_x)

    print("End of the training")



def train_epoch_adversarial(
    epoch: int,
    d_steps: int,
    gan: BaseModel,
    train_loader: DataLoader,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    device: str = "cpu",
    generator_loss_key: str = "g_total_loss",
    discriminator_loss_key: str = "d_total_loss",
) -> Dict[str, List[float]]:
    """
    Train a GAN for one epoch.
    
    Args:
        epoch: Current epoch number (for logging).
        d_steps: Number of discriminator steps per generator step.
        gan: The GAN model with loss_discriminator and loss_generator methods.
        train_loader: DataLoader for training data.
        generator_optimizer: Optimizer for the generator.
        discriminator_optimizer: Optimizer for the discriminator.
        device: Device to train on ("cpu" or "cuda").
        generator_loss_key: Key in the generator loss dict for backpropagation.
        discriminator_loss_key: Key in the discriminator loss dict for backpropagation.
    
    Returns:
        Dictionary mapping loss names to lists of per-batch loss values.
    """
    stats: Dict[str, List[float]] = defaultdict(list)
    g_losses: Dict[str, torch.Tensor] = {}

    gan.train()
    for iteration, x in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
        x = x.to(device)
        discriminator_optimizer.zero_grad()
        d_losses = gan.loss_discriminator(x)
        d_losses[discriminator_loss_key].backward()
        discriminator_optimizer.step()

        if iteration % d_steps == 0:
            generator_optimizer.zero_grad()
            g_losses = gan.loss_generator(x.shape[0])
            g_losses[generator_loss_key].backward()
            generator_optimizer.step()

        for k, v in d_losses.items():
            stats[k].append(v.item())
        for k, v in g_losses.items():
            stats[k].append(v.item())

    return stats


def train_adversarial(
    gan: BaseModel,
    train_loader: DataLoader,
    epochs: int,
    d_steps: int,
    generator_optimizer: Optimizer,
    discriminator_optimizer: Optimizer,
    device: str = "cpu",
    generator_loss_key: str = "g_total_loss",
    discriminator_loss_key: str = "d_total_loss",
    n_samples: int = 100,
    visualize_samples: bool = True,
    logscale_y: bool = False,
    logscale_x: bool = False,
) -> None:
    """
    Train a GAN with visualization.
    
    Args:
        gan: The GAN model with generator, discriminator, loss_discriminator,
            loss_generator, and sample methods.
        train_loader: DataLoader for training data.
        epochs: Number of training epochs.
        d_steps: Number of discriminator steps per generator step.
        generator_optimizer: Optimizer for the generator.
        discriminator_optimizer: Optimizer for the discriminator.
        device: Device to train on ("cpu" or "cuda").
        generator_loss_key: Key in the generator loss dict for backpropagation.
        discriminator_loss_key: Key in the discriminator loss dict for backpropagation.
        n_samples: Number of samples to generate for visualization.
        visualize_samples: Whether to visualize generated samples during training.
        logscale_y: Whether to use log scale for y-axis in loss plots.
        logscale_x: Whether to use log scale for x-axis in loss plots.
    
    Raises:
        AssertionError: If generator_loss_key equals discriminator_loss_key.
    """
    assert generator_loss_key != discriminator_loss_key, "Loss keys must be different!"
    gan.discriminator = gan.discriminator.to(device)
    gan.generator = gan.generator.to(device)

    train_losses: Dict[str, List[float]] = defaultdict(list)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_adversarial(
            epoch,
            d_steps,
            gan,
            train_loader,
            generator_optimizer,
            discriminator_optimizer,
            device,
            generator_loss_key,
            discriminator_loss_key
        )
        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
        g_epoch_loss = np.mean(train_loss[generator_loss_key])
        d_epoch_loss = np.mean(train_loss[discriminator_loss_key])

        gan.eval()
        if visualize_samples:
            with torch.no_grad():
                samples = gan.sample(n_samples)
                if isinstance(samples, torch.Tensor):
                    samples = samples.cpu().detach().numpy()

            clear_output(wait=True)
            title = f"Samples, epoch: {epoch}, {generator_loss_key}: {g_epoch_loss:.3f}, {discriminator_loss_key}: {d_epoch_loss:.3f}"
            if check_samples_is_2d(samples):
                visualize_2d_samples(samples, title=title)
            else:
                show_samples(samples, title=title)
            plot_training_curves(epoch, train_losses, None, logscale_y, logscale_x)
        else:
            clear_output(wait=True)
            print(f"Epoch: {epoch}, {generator_loss_key}: {g_epoch_loss:.3f}, {discriminator_loss_key}: {d_epoch_loss:.3f}")
            plot_training_curves(epoch, train_losses, None, logscale_y, logscale_x)
                    
    print("End of the training")
