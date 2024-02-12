"""Run PCA on a mock image buffer to check if it's a viable exemplar representation."""

import numpy as np
from sklearn.decomposition import PCA
from time import time


def _main():
    start = time()
    batch_size = 10000
    images = np.random.rand(batch_size, 3, 256, 256)
    images_flattened = images.reshape(batch_size, -1)
    pca = PCA(n_components=512, svd_solver="randomized")
    pca.fit_transform(images_flattened)
    end = time()

    print(f"Time to run PCA: {end - start:.2f} seconds.")


if __name__ == "__main__":
    _main()
