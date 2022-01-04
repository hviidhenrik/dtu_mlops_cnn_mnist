import click
import numpy as np
import torch
from src.models.model import *
from src.visualization.visualize import *


def predict(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target.type(torch.long), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


@click.command()
@click.argument('model_filepath', type=click.Path())
@click.argument('data_filepath', type=click.Path())
def main(model_filepath: str, data_filepath: str):
    model = torch.load(model_filepath)
    new_data = torch.load(data_filepath).tensors
    idx = np.random.choice(len(new_data[0]), 5, replace=False)
    x_new = new_data[0][idx, :, :]
    y_true = new_data[1][idx,]
    y_pred = model(x_new).argmax(dim=1, keepdim=True)
    visualize_mnist_digits(x_new, y_pred.numpy(), savepath="reports/figures/")


if __name__ == "__main__":
    main()
