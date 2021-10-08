import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import random_split


def imshow(img):
    """Show a tensor image"""
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def split_train_validation(dataset, tr=0.8):
    n_train_samples = int(tr * len(dataset))
    n_val_samples = len(dataset) - n_train_samples
    print(n_train_samples, n_val_samples)

    generator = torch.Generator().manual_seed(42)
    training_data, validation_data = random_split(dataset, [n_train_samples, n_val_samples], generator)
    return training_data, validation_data
