from typing import Any, Dict, Tuple

import numpy as np
import torch
from numpy.core.records import ndarray
from torch.utils.data import TensorDataset


def _load_and_concat_all_train_data_files(path: str) -> Dict[str, ndarray]:
    """
    Internal function used by mnist(). Loads all the training data files for the MNIST model

    :param path: the path with the data
    :return: a dictionary with the data as images and labels
    """
    train_images = []
    train_labels = []
    for i in range(5):
        train = np.load(path + f"train_{i}.npz")
        train_images.append(train["images"])
        train_labels.append(train["labels"])

    return {
        "images": np.array(train_images).reshape(-1, 28, 28),
        "labels": np.array(train_labels).reshape(
            -1,
        ),
    }


def mnist(path: str) -> Tuple[Any, Any]:
    """
    Loads all the training data and test data for the MNIST model.

    :param path: the path with the location of the data
    :return: a tuple with the training and test data
    """
    train = _load_and_concat_all_train_data_files(path)
    test = np.load(path + "test.npz")
    return train, test


def convert_mnist_to_tensor_dataset(
    train: np.array, test: ndarray
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Converts training and test data from numpy arrays to torch TensorDataset

    :param train: the training data as numpy array
    :param test: the test data as numpy array
    :return: tuple with the training and test data as TensorDataset
    """
    train_x, train_y = torch.Tensor(train["images"]), torch.Tensor(train["labels"])
    test_x, test_y = torch.Tensor(test["images"]).view(-1, 28, 28), torch.Tensor(
        test["labels"]
    )
    train = TensorDataset(train_x, train_y)
    test = TensorDataset(test_x, test_y)
    return train, test


def save_train_and_test_as_tensor_datasets(
    train: TensorDataset, test: TensorDataset, output_filepath: str
) -> None:
    """
    Saves the training and test data as TensorDataset files

    :param train: training data as TensorDataset
    :param test: test data as TensorDataset
    :param output_filepath: the location to save the data to
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
