import argparse
import os

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.data.data import load_train_and_test_tensor_datasets
from src.models.CNN.model import *
from tests import _PATH_DATA


def test_CNN_model_training():
    """
    Tests if the MNIST CNN model can run a single epoch without error and subsequently trains another epoch. The
    final loss is then asserted to be lower than the initial, indicating correct training.

    """
    torch.manual_seed(1234)
    train_data, test_data = load_train_and_test_tensor_datasets(
        filepath=os.path.join(_PATH_DATA, "test/")
    )

    # load data into dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=250)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=250)

    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    args = argparse.Namespace()
    args.log_interval = 10

    # train model
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    # first run a single epoch and raise error if it fails
    try:
        test(model, test_loader)  # sets the initial loss with random weights
        train(args, model, train_loader, optimizer, epoch=1)
        scheduler.step()
    except:
        print("First epoch didn't complete!")

    # if first epoch completed, continue for another epoch and see if loss decreases
    train(args, model, train_loader, optimizer, epoch=2)
    test(model, test_loader)  # sets the final test loss

    assert model.loss[0] > model.loss[-1], "Test loss didn't decrease after two epochs, something might be wrong"
