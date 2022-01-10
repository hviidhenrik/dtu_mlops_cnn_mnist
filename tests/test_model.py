import os

import pytest
import torch

from tests import _PATH_SAVED_MODELS, _PATH_DATA_MNIST_PROCESSED, _CNN_MODEL_PATH


@pytest.mark.skipif(not os.path.isfile(_CNN_MODEL_PATH), reason=f"No saved CNN model found in {_PATH_SAVED_MODELS}")
@pytest.mark.parametrize("batch_size", [1, 5, 10, 128])
def test_CNN_model_input_output_match(batch_size):
    """
    Tests if the output from the trained model matches the expected dimensions, given the input

    :param batch_size: different batch sizes are tested for
    """
    model = torch.load(_CNN_MODEL_PATH)
    new_data = torch.load(os.path.join(_PATH_DATA_MNIST_PROCESSED, "test.pt")).tensors
    x_new = new_data[0][0:batch_size, :, :]
    y_pred = model(x_new).argmax(dim=1, keepdim=True)

    assert x_new.shape == torch.Size([batch_size, 28, 28]), "Input data, X, not the right shape, expected (5, 28, 28)"
    assert y_pred.shape == torch.Size([batch_size, 1]), "Predictions, Y, not the right shape, expected (5, 1)"
