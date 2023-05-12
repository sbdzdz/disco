"""Lightning callbacks."""
import torch
from pytorch_lightning.callbacks import Callback

from codis.data import InfiniteDSprites, Latents
from codis.visualization import draw_batch_and_reconstructions


class VisualizationCallback(Callback):
    """Callback for visualizing the model's reconstructions."""

    def __init__(self, shapes):
        self.shapes = shapes

    def _get_vis_batch(self):
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
            for shape in self.shapes
        ]
        return torch.stack(batch)

    def on_train_epoch_end(self, trainer, pl_module):
        """Visualize the model's reconstructions."""
        x = self._get_vis_batch()
        x_hat, *_ = pl_module(x)
        vis = draw_batch_and_reconstructions(x, x_hat)
        trainer.logger.experiment.log({"reconstructions": vis})
