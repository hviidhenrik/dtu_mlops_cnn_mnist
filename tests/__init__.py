import os

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
_PATH_DATA_MNIST_PROCESSED = os.path.join(_PATH_DATA, "processed/")
_PATH_SAVED_MODELS = os.path.join(_PROJECT_ROOT, "models")  # saved models
_CNN_MODEL_PATH = os.path.join(_PATH_SAVED_MODELS, "cnn_mnist.pt")

