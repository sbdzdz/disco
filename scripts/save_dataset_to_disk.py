"""Generate and save a continual dataset to disk."""
from codis.data import ContinualDataset, InfiniteDSprites, Latents
from omegaconf import DictConfig
import hydra
from pathlib import Path


@hydra.main(config_path="../configs/dataset", config_name="idsprites_disk")
def main(cfg: DictConfig):
    shapes = [
        InfiniteDSprites().generate_shape()
        for _ in range(cfg.tasks * cfg.shapes_per_task)
    ]
    exemplars = generate_exemplars(shapes, img_size=cfg.img_size)
    dataset = ContinualDataset(shapes, exemplars)

    for i, (train_dataset, val_dataset, test_dataset), exemplars in enumerate(dataset):
        print(f"Saving task {i} to disk...")
        out_dir = Path(cfg.dataset_dir) / f"task_{i}"
        out_dir.mkdir(parents=True, exist_ok=True)


def generate_exemplars(shapes, img_size: int):
    """Generate a batch of exemplars for training and visualization."""
    dataset = InfiniteDSprites(
        img_size=img_size,
    )
    return [
        dataset.draw(
            Latents(
                color=(1.0, 1.0, 1.0),
                shape=shape,
                shape_id=None,
                scale=1.0,
                orientation=0.0,
                position_x=0.5,
                position_y=0.5,
            )
        )
        for shape in shapes
    ]


if __name__ == "__main__":
    main()
