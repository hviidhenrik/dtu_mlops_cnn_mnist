import matplotlib.pyplot as plt
import numpy as np
from numpy.core.records import ndarray


def visualize_mnist_digits(
    x_array: ndarray, predicted_label: np.int64, savepath: str
) -> None:
    """
    Takes image data and corresponding labels and saves a plot of the
    digits with the corresponding predicted labels

    :param x_array: numpy array with the image data.
    Must have dimension (N, 28, 28), where N is the number of images
    :param predicted_label: a numpy array of predicted labels
    :param savepath: location to save the plots to
    """
    for i in range(len(predicted_label)):
        pixels = x_array[i]
        label = predicted_label[i]
        plt.title(f"Predicted label: {label}")
        plt.imshow(pixels, cmap="gray")
        plt.savefig(savepath + f"digit_predicted_{i}.png")
        plt.close()
