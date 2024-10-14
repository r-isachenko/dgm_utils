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
    device: str = "cpu",
    loss_key: str = "total",
) -> defaultdict[str, List[float]]:
    model.train()

    stats = defaultdict(list)
    for x in tqdm(train_loader, desc=f'Training epoch {epoch}'):
        x = x.to(device)
        losses = model.loss(x)
        optimizer.zero_grad()
        losses[loss_key].backward()
        optimizer.step()

        for k, v in losses.items():
            stats[k].append(v.item())

    return stats


def eval_model(epoch: int, model: BaseModel, data_loader: DataLoader, device: str = "cpu") -> defaultdict[str, float]:
    model.eval()
    stats = defaultdict(float)
    with torch.no_grad():
        for x in tqdm(data_loader, desc=f'Evaluating epoch {epoch}'):
            x = x.to(device)
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
    device: str = "cpu",
    loss_key: str = "total_loss",
    n_samples: int = 100,
    visualize_samples: bool = True,
    logscale_y: bool = False,
    logscale_x: bool = False,
):

    train_losses: Dict[str, List[float]] = defaultdict(list)
    test_losses: Dict[str, List[float]] = defaultdict(list)
    model = model.to(device)
    print("Start of the training")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            epoch, model, train_loader, optimizer, device, loss_key
        )
        if scheduler is not None:
            scheduler.step()
        test_loss = eval_model(epoch, model, test_loader, device)

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
            plot_training_curves(train_losses, test_losses, logscale_y, logscale_x)
        else:
            clear_output(wait=True)
            print(f"Epoch: {epoch}, loss: {epoch_loss}")
            plot_training_curves(train_losses, test_losses, logscale_y, logscale_x)
                    
    print("End of the training")
