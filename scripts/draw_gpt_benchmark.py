"""Create the GPT-V benchmark."""

from argparse import ArgumentParser
from pathlib import Path

import idsprites as ids
from PIL import Image


def draw_gpt_benchmark(args):
    dataset = ids.RandomDSprites(img_size=args.img_size)
    shapes = [dataset.generate_shape() for _ in range(args.num_tasks)]
    for task, shape in enumerate(shapes):
        path = args.out_path / f"task_{task}"
        path.mkdir(parents=True, exist_ok=True)
        canonical_factors = ids.Factors(
            color=(1, 1, 1),
            shape=None,
            shape_id=None,
            scale=1.0,
            orientation=0.0,
            position_x=0.5,
            position_y=0.5,
        )
        query_image = dataset.draw(
            dataset.sample_factors()._replace(shape=shape),
            channels_first=False,
        )
        correct_answer = dataset.draw(
            canonical_factors._replace(shape=shape), channels_first=False
        )
        incorrect_answers = [
            dataset.draw(
                canonical_factors.replace(shape=dataset.generate_shape()),
                channels_first=False,
            )
            for _ in range(args.num_samples_per_task - 1)
        ]
        save_img(query_image, path / "query.png")
        save_img(correct_answer, path / "correct.png")
        for i, img in enumerate(incorrect_answers):
            save_img(img, path / f"incorrect_{i}.png")


def save_img(img, path):
    img = Image.fromarray((255 * img).astype("uint8"))
    img.save(path)


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    parser = ArgumentParser()
    parser.add_argument("--out_path", type=Path, default=root / "img/gpt_benchmark")
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_samples_per_task", type=int, default=10)
    args = parser.parse_args()
    draw_gpt_benchmark(args)
