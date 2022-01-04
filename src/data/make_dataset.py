import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from typing import Union, Any, List

import numpy
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset
import numpy as np

from src.data.data import *


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train, test = mnist(input_filepath)
    train, test = convert_mnist_to_tensor_dataset(train, test)
    save_train_and_test_as_tensor_datasets(train, test, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
