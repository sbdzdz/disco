from disco.data import InfiniteDSprites
import cProfile
from random import random


def _main():
    dataset = InfiniteDSprites(orientation_marker=False)
    latents = dataset.sample_latents()

    for _ in range(1000):
        latents = latents._replace(scale=random(), orientation=random())
        dataset.draw(latents)


if __name__ == "__main__":
    cProfile.run("_main()", sort="cumtime")
