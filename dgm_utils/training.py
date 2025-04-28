from collections import defaultdict
from IPython.display import clear_output
from typing import Dict, List, Optional

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
    loss_key: str = "total",
) -> defaultdict[str, List[float]]:
    model.train()

    stats = defaultdict(list)
    for batch in tqdm(train_loader, desc=f'Training epoch {epoch}'):
        if conditional:
            x, y = batch
            x, y = x.to(device), y.to(device)
            losses = model.loss(x, y)
        else:
            x = batch.to(device)
            losses = model.loss(x)
        optimizer.zero_grad()
        losses[loss_key].backward()
        optimizer.step()

        for k, v in losses.items():
            stats[k].append(v.item())

    return stats


def eval_model(
    epoch: int, 
    model: BaseModel, 
    data_loader: DataLoader, 
    conditional: bool = False,
    device: str = "cpu"
) -> defaultdict[str, float]:
    model.eval()
    stats = defaultdict(float)
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'Evaluating epoch {epoch}'):
            if conditional:
                x, y = batch
                x, y = x.to(device), y.to(device)
                losses = model.loss(x, y)
            else:
                x = batch.to(device)
                losses = model.loss(x)
            
            for k, v in losses.items():
                stats[k] += v.item() * x.shape[0]

        for k in stats.keys():
            stats[k] /= len(data_loader.dataset) # type: ignore
    return stats


def check_samples_is_2d(samples: np.ndarray) -> bool:
    shape = samples.shape
    if len(shape) == 2 and shape[1] == 2:
        return True
    return False


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
):

    train_losses: Dict[str, List[float]] = defaultdict(list)
    test_losses: Dict[str, List[float]] = defaultdict(list)
    model = model.to(device)
    print("Start of the training")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            epoch, model, train_loader, optimizer, conditional, device, loss_key
        )
        if scheduler is not None:
            scheduler.step()
        test_loss = eval_model(epoch, model, test_loader, conditional, device)

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
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
) -> defaultdict[str, List[float]]:
    stats = defaultdict(list)

    gan.train()
    for iteration, x in enumerate(tqdm(train_loader, desc=f'Training epoch {epoch}')):
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
):
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
