import click

from src.models.model import *
from src.visualization.visualize import *


@click.command()
@click.argument("model_filepath", type=click.Path())
@click.argument("data_filepath", type=click.Path())
def main(model_filepath: str, data_filepath: str):
    model = torch.load(model_filepath)
    new_data = torch.load(data_filepath).tensors
    idx = np.random.choice(len(new_data[0]), 5, replace=False)
    x_new = new_data[0][idx, :, :]
    y_pred = model(x_new).argmax(dim=1, keepdim=True)
    visualize_mnist_digits(x_new, y_pred.numpy(), savepath="reports/figures/")


if __name__ == "__main__":
    main()
