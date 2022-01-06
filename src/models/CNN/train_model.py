from __future__ import print_function

import argparse
import os

import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR

from src.data.data import load_train_and_test_tensor_datasets
from src.models.CNN.model import *


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch corrupted MNIST model")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_data, test_data = load_train_and_test_tensor_datasets(
        filepath="data/processed/"
    )

    # load data into dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # train model
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        os.makedirs("models/", exist_ok=True)
        torch.save(model, "models/" + "cnn_mnist.pt")

    # plot loss and accuracy curves
    plt.plot(np.arange(1, len(model.loss) + 1), model.loss)
    plt.title("Loss curve")
    plt.ylabel("NLL Loss")
    plt.xlabel("Epoch")
    plt.savefig("reports/figures/CNN/cnn_mnist_loss_curve.png")
    plt.close()

    plt.plot(np.arange(1, len(model.acc) + 1), model.acc)
    plt.title("Accuracy curve")
    plt.ylabel("Binary accuracy [%]")
    plt.xlabel("Epoch")
    plt.savefig("reports/figures/CNN/cnn_mnist_acc_curve.png")
    plt.close()


if __name__ == "__main__":
    main()

# run it with data e.g. "--epochs=1" as parameter
