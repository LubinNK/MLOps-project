import os
import sys

import numpy as np
import torch


def load_mnist(flatten=False, train=True):
    """taken from https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py"""
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source="http://yann.lecun.com/exdb/mnist/"):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return np.copy(data)

    # We can now download and read the training and test set images and labels.
    if train:
        X = load_mnist_images("data/train-images-idx3-ubyte.gz")
        y = load_mnist_labels("data/train-labels-idx1-ubyte.gz")
    else:
        X = load_mnist_images("data/t10k-images-idx3-ubyte.gz")
        y = load_mnist_labels("data/t10k-labels-idx1-ubyte.gz")

    if flatten:
        X = X.reshape([X.shape[0], -1])

    return X, y


def get_loader(X, y, batch_size=64):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(y).long()
    )
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    return train_loader
