import torch

from src.data.data import load_train_and_test_tensor_datasets
from tests import _PATH_DATA_MNIST_PROCESSED_TEST


def test_data_loading():
    train_data, test_data = load_train_and_test_tensor_datasets(
        filepath=_PATH_DATA_MNIST_PROCESSED_TEST
    )
    train_tensors = train_data.tensors[0]
    train_labels = train_data.tensors[1]
    test_tensors = test_data.tensors[0]
    test_labels = test_data.tensors[1]

    N_train, N_test = 1000, 1000

    assert len(train_data) == N_train and len(
        test_data) == N_test, "Train or test data sets not expected length. Should " \
                              "be 1000 and 1000, respectively."
    assert train_tensors.shape == torch.Size([N_train, 28, 28]), "Train tensor not the right shape, expected " \
                                                                 "(1000, 28, 28)"
    assert train_labels.shape == torch.Size([N_train, ]), "Train labels not the right shape, expected " \
                                                          "(1000,)"
    assert test_tensors.shape == torch.Size([N_test, 28, 28]), "Test tensor not the right shape, expected " \
                                                               "(1000, 28, 28)"
    assert test_labels.shape == torch.Size([N_test, ]), "Test labels not the right shape, expected " \
                                                        "(1000,)"
