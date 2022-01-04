from typing import Union, Any, List, Tuple

import numpy
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np


def _load_and_concat_all_train_data_files(path):
    train_images = []
    train_labels = []
    for i in range(1):
        train = np.load(path + f"train_{i}.npz")
        train_images.append(train["images"])
        train_labels.append(train["labels"])

    return {"images": np.array(train_images).reshape(-1, 28, 28),
            "labels": np.array(train_labels).reshape(-1, )}


def mnist(path: str = None):
    train = _load_and_concat_all_train_data_files(path)
    test = np.load(path + "test.npz")
    return train, test


def convert_mnist_to_tensor_dataset(train, test):
    train_x, train_y = torch.Tensor(train["images"]), torch.Tensor(train["labels"])
    test_x, test_y = torch.Tensor(test["images"]).view(-1, 28, 28), torch.Tensor(test["labels"])
    train = TensorDataset(train_x, train_y)
    test = TensorDataset(test_x, test_y)
    return train, test


def save_train_and_test_as_tensor_datasets(train: Any, test: Any, output_filepath: str) -> None:
    """

    :param train:
    :param test:
    :param output_filepath:
    :return:
    """
    torch.save(train, output_filepath + "train.pt")
    torch.save(test, output_filepath + "test.pt")


def load_train_and_test_tensor_datasets(filepath: str) -> Tuple[Any, Any]:
    """
    Function that loads train and test sets from saved tensor datasets

    :param filepath: the location of the datasets
    :return: train and test datasets
    """
    train = torch.load(filepath + "train.pt")
    test = torch.load(filepath + "test.pt")
    return train, test


def visualize_mnist_digits(train_array: numpy.ndarray, labels: numpy.int64,
                           digit_indices: List[int] = None):
    digit_indices = [0, 1, 2, 3, 10, 500, 4500] if digit_indices is None else digit_indices
    for i in digit_indices:
        pixels = train_array[i]
        label = labels[i]
        plt.title('Label is {label}'.format(label=label))
        plt.imshow(pixels, cmap='gray')
        plt.show()
