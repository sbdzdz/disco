from disco.data import InfiniteDSprites
import cProfile
import pstats
from random import random
from pathlib import Path


def _main():
    dataset = InfiniteDSprites(orientation_marker=True)
    latents = dataset.sample_latents()

    for _ in range(10000):
        latents = latents._replace(
            scale=random(),
            orientation=random(),
            position_x=random(),
            position_y=random(),
        )
        dataset.draw(latents)


if __name__ == "__main__":
    path = Path("profile")
    cProfile.run("_main()", "profile", sort="cumtime")

    p = pstats.Stats("profile")
    p.sort_stats("cumulative").print_stats(10)
    path.unlink()
