"""Render a few shapes with the same scale and compare their properties."""
import cv2
from itertools import islice
import imageio

import matplotlib.pyplot as plt
import numpy as np

from disco.data.infinite_dsprites import InfiniteDSprites


def main():
    """Render a few shapes with the same scale and compare their properties."""
    np.random.seed(0)
    dataset = InfiniteDSprites(
        scale_range=[2],
        orientation_range=[0],
        position_x_range=[0.5],
        position_y_range=[0.5],
        color_range=[(0.9, 0.9, 0.9)],
    )
    images = []
    for img, _ in islice(dataset, 10):
        img = np.transpose(img, (1, 2, 0))

        # get all non-black pixels across all channels
        non_black_pixels = np.argwhere(np.any(img != [0, 0, 0], axis=2))

        # find the point furthest from the image center
        center = np.array(img.shape[:2]) / 2
        cv2.circle(img, tuple(center.astype(int)), 3, (1, 0, 0), 3)

        center_of_mass = np.mean(non_black_pixels, axis=0)
        center_of_mass = center_of_mass[::-1]
        cv2.circle(img, tuple(center_of_mass.astype(int)), 3, (0, 0, 1), 3)

        distances = np.linalg.norm(non_black_pixels - center_of_mass, axis=1)
        furthest_idx = np.argmax(distances)
        furthest_pixel = non_black_pixels[furthest_idx]
        cv2.circle(img, tuple(furthest_pixel[::-1]), 3, (0, 1, 0), 3)
        cv2.line(
            img,
            tuple(center_of_mass.astype(int)),
            tuple(furthest_pixel[::-1]),
            (0, 1, 0),
            3,
        )

        print(f"Maximum distance from the center: {distances.max()}")
        print(f"Object center: {center_of_mass}")

        plt.imshow(img)
        images.append(img)

    # create a gif
    imageio.mimsave("img/scale.gif", images, fps=1)


if __name__ == "__main__":
    main()
