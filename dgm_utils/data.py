from typing import Literal, Optional, Tuple

import numpy as np
from sklearn.datasets import make_moons

import torchvision


TOY_DATASETS = ["moons"]
IMAGE_DATASETS = ["mnist", "cifar10"]

def prepare_images(
    train_data: np.ndarray, 
    test_data: np.ndarray, 
    flatten: bool = False, 
    binarize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if binarize:
        train_data = (train_data > 128).astype("float32")
        test_data = (test_data > 128).astype("float32")
    else:
        train_data = train_data / 255.0
        test_data = test_data / 255.0

    train_data = np.transpose(train_data, (0, 3, 1, 2))
    test_data = np.transpose(test_data, (0, 3, 1, 2))

    if flatten:
        train_data = train_data.reshape(len(train_data.shape[0]), -1)
        test_data = test_data.reshape(len(train_data.shape[0]), -1)
    
    return train_data, test_data

def load_MNIST(with_targets: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    train_dataset = torchvision.datasets.MNIST(root="./", train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root="./", train=False, download=True)
    train_data, test_data = train_dataset.data.numpy(), test_dataset.data.numpy()
    axis_index = len(train_data.shape)
    train_data = np.expand_dims(train_data, axis=axis_index)
    test_data = np.expand_dims(test_data, axis=axis_index)

    if with_targets:
        train_labels, test_labels = (
            train_dataset.targets.numpy(),
            test_dataset.targets.numpy(),
        )
        return train_data, train_labels, test_data, test_labels

    return train_data, None, test_data, None


def load_CIFAR10(with_targets: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    train_dataset = torchvision.datasets.CIFAR10(root="./", train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="./", train=False, download=True)
    train_data, test_data = train_dataset.data, test_dataset.data

    if with_targets:
        train_labels, test_labels = (
            train_dataset.targets.numpy(),
            test_dataset.targets.numpy(),
        )
        return train_data, train_labels, test_data, test_labels

    return train_data, None, test_data, None

def load_moons(
    size: int, 
    with_targets: bool
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    data, labels = make_moons(n_samples=size, noise=0.1)
    split = int(0.8 * size)
    train_data, test_data = data[:split], data[split:]

    if with_targets:
        train_labels, test_labels = labels[:split], labels[split:]
        return train_data, train_labels, test_data, test_labels
    
    return train_data, None, test_data, None


def _load_dataset(
    name: Literal["mnist", "cifar10", "moons"], 
    size: Optional[int] = None,
    with_targets: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    if name == "mnist":
        return load_MNIST(with_targets=with_targets)
    elif name == "cifar10":
        return load_CIFAR10(with_targets=with_targets)
    elif name == "moons":
        if size is None:
            raise ValueError("Size must be specifed for 'moons' dataset!")
        return load_moons(size=size, with_targets=with_targets)
    else:
        raise ValueError("The argument name must have the values 'mnist', 'cifar10' or 'moons'!")


def load_dataset(
    name: str, 
    flatten: bool = False, 
    binarize: bool = True, 
    with_targets: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:

    train_data, train_labels, test_data, test_labels = _load_dataset(name, with_targets=with_targets)
    train_data = train_data.astype("float32")
    test_data = test_data.astype("float32")

    if name in IMAGE_DATASETS:
        train_data, test_data = prepare_images(
            train_data, 
            test_data, 
            flatten=flatten, 
            binarize=binarize
        )
    
    if with_targets:
        return train_data, train_labels, test_data, test_labels

    return train_data, test_data
