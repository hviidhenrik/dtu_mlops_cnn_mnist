import torch

from src.data.data import load_train_and_test_tensor_datasets
from tests import _PATH_DATA_MNIST_PROCESSED


def test_data_loading():
    train_data, test_data = load_train_and_test_tensor_datasets(
        filepath=_PATH_DATA_MNIST_PROCESSED  # os.path.join(_PATH_DATA, "processed/")
    )
    train_tensors = train_data.tensors[0]
    train_labels = train_data.tensors[1]
    test_tensors = test_data.tensors[0]
    test_labels = test_data.tensors[1]

    assert len(train_data) == 25000 and len(test_data) == 5000, "Train or test data sets not expected length. Should " \
                                                                "be 25000 and 5000, respectively."
    assert train_tensors.shape == torch.Size([25000, 28, 28]), "Train tensor not the right shape, expected " \
                                                               "(25000, 28, 28)"
    assert train_labels.shape == torch.Size([25000, ]), "Train labels not the right shape, expected " \
                                                        "(25000,)"
    assert test_tensors.shape == torch.Size([5000, 28, 28]), "Test tensor not the right shape, expected " \
                                                             "(5000, 28, 28)"
    assert test_labels.shape == torch.Size([5000, ]), "Test labels not the right shape, expected " \
                                                      "(5000,)"
