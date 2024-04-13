from collections import defaultdict
from IPython.display import clear_output
from typing import Optional, Tuple

import numpy as np

import torch

from .visualize import show_samples, visualize_2d_samples


def train_epoch(
    model: object,
    train_loader: object,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: str = "cpu",
    loss_key: str = "total",
) -> defaultdict:
    model.train()

    stats = defaultdict(list)
    for x in train_loader:
        x = x.to(device)
        losses = model.loss(x)
        optimizer.zero_grad()
        losses[loss_key].backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for k, v in losses.items():
            stats[k].append(v.item())

    return stats


def eval_model(model: object, data_loader: object, device: str = "cpu") -> defaultdict:
    model.eval()
    stats = defaultdict(float)
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            losses = model.loss(x)
            for k, v in losses.items():
                stats[k] += v.item() * x.shape[0]

        for k in stats.keys():
            stats[k] /= len(data_loader.dataset)
    return stats


def check_samples_is_2d(samples: np.ndarray) -> bool:
    shape = samples.shape
    if len(shape) == 2 and shape[1] == 2:
        return True
    return False


def train_model(
    model: object,
    train_loader: object,
    test_loader: object,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: str = "cpu",
    loss_key: str = "total_loss",
) -> Tuple[dict, dict]:

    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, loss_key
        )
        test_loss = eval_model(model, test_loader, device)

        for k in train_loss.keys():
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])

        clear_output(wait=True)
        with torch.no_grad():
            samples = model.sample(100)
            samples = samples.cpu().detach().numpy()

        epoch_loss = np.mean(train_loss[loss_key])
        title = f"Samples, epoch: {epoch}, {loss_key}: {epoch_loss:.3f}"
        if check_samples_is_2d(samples):
            visualize_2d_samples(samples, title=title)
        else:
            show_samples(samples, title=title)

    return dict(train_losses), dict(test_losses)
