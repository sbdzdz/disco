import numpy as np
from torch import nn
import itertools


def generate_binary_images(n=4):
    for n in itertools.product([0, 1], repeat=n**2):
        yield np.array(n).reshape(1, 4, 4)


def translate_x(data):
    for i in range(4):
        yield np.roll(data, i, axis=1)


def translate_y(data):
    for i in range(4):
        yield np.roll(data, i, axis=2)


def rotate(data):
    for i in range(4):
        yield np.rot90(data, k=i, axes=(1, 2))


def mirror(data):
    yield np.flip(data, axis=1)
