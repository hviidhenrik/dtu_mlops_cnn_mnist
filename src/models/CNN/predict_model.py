import click

from src.models.CNN.model import *
from src.visualization.visualize import *


@click.command()
@click.argument("model_filepath", type=click.Path())
@click.argument("data_filepath", type=click.Path())
def main(model_filepath: str, data_filepath: str):
    """
    Runs the predict function of the model on given data

    :param model_filepath:
    :param data_filepath:
    :return:
    """
    model = torch.load(model_filepath)
    new_data = torch.load(data_filepath).tensors
    idx = np.random.choice(len(new_data[0]), 5, replace=False)
    x_new = new_data[0][idx, :, :]
    y_pred = model(x_new).argmax(dim=1, keepdim=True)
    visualize_mnist_digits(x_new, y_pred.numpy(), savepath="reports/figures/CNN/")


if __name__ == "__main__":
    main()

# run with "models/cnn_mnist.pt" and "data/processed/test.pt" as parameters
