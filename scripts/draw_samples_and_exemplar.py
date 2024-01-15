from pathlib import Path

from PIL import Image

from codis.data import InfiniteDSprites, RandomDSprites


def _main():
    dataset = InfiniteDSprites()
    shape = dataset.generate_shape()

    path = Path("img/samples")
    path.mkdir(parents=True, exist_ok=True)

    dataset = RandomDSprites(
        img_size=256,
        shapes=[shape],
    )
    for i in range(10):
        factors = dataset.sample_latents()
        img = dataset.draw(factors, channels_first=False)
        img = Image.fromarray((255 * img).astype("uint8"))
        img.save(path / f"sample_{i}.png")

    exemplar_factors = dataset.sample_latents()._replace(
        scale=1.0,
        orientation=0.0,
        position_x=0.5,
        position_y=0.5,
    )
    img = dataset.draw(exemplar_factors, channels_first=False)
    img = Image.fromarray((255 * img).astype("uint8"))
    img.save(path / "exemplar.png")


if __name__ == "__main__":
    _main()
