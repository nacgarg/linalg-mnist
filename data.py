import tensorflow_datasets as tfds
import numpy as np

_data = None


def get_mnist_dataset():
    ''' Returns the MNIST dataset '''
    global _data
    if _data is None:
        mnist = tfds.image.mnist.MNIST()
        mnist.download_and_prepare()
        _data = mnist.as_dataset()
    return _data


def process_example(ex):
    ''' Scales and one-hot encodes a single training example from MNIST '''
    x_raw, y_raw = ex['image'], ex['label']

    # Scale x_raw from 0 to 1 as a float
    x = x_raw.numpy().astype(np.float16) / 256

    # Flatten x from (1, 28, 28) to (1, 784)
    x = x.reshape((1, 784))

    # One hot encode y_raw
    y = np.zeros((10))
    y[y_raw] = 1
    return x, y


def process_batch(ex):
    ''' Scales and one-hot encodes a batch of training examples from MNIST '''
    x_raw, y_raw = ex['image'], ex['label']
    batch_size = x_raw.shape[0]

    # Scale x_raw from 0 to 1 as a float
    x = x_raw.numpy().astype(np.float16) / 256

    # Flatten x from (batch_size, 1, 28, 28) to (batch_size, 1, 784)
    x = x.reshape((batch_size, 1, 784))

    # One hot encode y_raw
    y = np.zeros((batch_size, 10))
    for i, v in enumerate(y_raw):
        y[i, v] = 1
    return x, y
