import argparse
import logging

from src.data.data import *


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be used in tests (saved in ../test).
    """

    parser = argparse.ArgumentParser(
        description="Make a small dataset for unit testing"
    )
    parser.add_argument(
        "--input-filepath",
        type=str,
        default="data/raw/",
        help="path to raw input data (default: data/raw/)",
    )
    parser.add_argument(
        "--output-filepath",
        type=str,
        default="data/test/",
        help="path to save output data (default: data/test/)",
    )
    parser.add_argument(
        "--N-obs",
        type=int,
        default=1000,
        metavar="N",
        help="number of observation tensors to keep (default: 1000)",
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    train, test = mnist(args.input_filepath, num_files=1)

    train_x, train_y = torch.Tensor(train["images"]), torch.Tensor(train["labels"])
    test_x, test_y = torch.Tensor(test["images"]).view(-1, 28, 28), torch.Tensor(
        test["labels"]
    )

    train_x = train_x[0 : args.N_obs, :, :].clone()
    train_y = train_y[
        0 : args.N_obs,
    ].clone()
    test_x = test_x[0 : args.N_obs, :, :].clone()
    test_y = test_y[
        0 : args.N_obs,
    ].clone()

    train = TensorDataset(train_x, train_y)
    test = TensorDataset(test_x, test_y)

    save_train_and_test_as_tensor_datasets(train, test, args.output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

    # run it with e.g.: python src/data/make_test_dataset.py data/raw/ data/test/ 1000
