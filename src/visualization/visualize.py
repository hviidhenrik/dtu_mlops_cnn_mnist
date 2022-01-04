from typing import List

import matplotlib.pyplot as plt
import numpy as np


def visualize_mnist_digits(x_array: np.ndarray, true_labels: np.int64, savepath: str):
    for i in range(len(true_labels)):
        pixels = x_array[i]
        label = true_labels[i]
        plt.title(f'True label: {label}')
        plt.imshow(pixels, cmap='gray')
        plt.savefig(savepath + f"digit_predicted_{i}.png")
        plt.close()
