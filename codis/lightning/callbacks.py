"""Lightning callbacks."""
import torch
from pytorch_lightning.callbacks import Callback

from codis.data import InfiniteDSprites, Latents
from codis.visualization import draw_batch_and_reconstructions


class VisualizationCallback(Callback):
    """Callback for visualizing VAE reconstructions."""

    def __init__(self, shapes):
        self.vis_batch = self._build_vis_batch(shapes)

    def on_train_epoch_end(self, trainer, pl_module):
        """Visualize the images and reconstructions."""
        pl_module.eval()
        x_hat, *_ = pl_module(self.vis_batch)
        trainer.logger.experiment.log(
            {"reconstructions": draw_batch_and_reconstructions(self.vis_batch, x_hat)}
        )
        pl_module.train()

    @staticmethod
    def _build_vis_batch(shapes):
        """Prepare a data loader for visualization."""
        dataset = InfiniteDSprites()
        batch = [
            dataset.draw(
                Latents(
                    color=(0.0, 0.0, 0.0),
                    shape=shape,
                    scale=0.5,
                    orientation=0.0,
                    position_x=0.5,
                    position_y=0.5,
                )
            )
            for shape in shapes
        ]
        return torch.stack(batch)
