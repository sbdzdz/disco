"""Lightning callbacks."""
from lightning.pytorch.callbacks import Callback

from codis.visualization import draw_batch_and_reconstructions
from codis.utils import to_numpy


class VisualizationCallback(Callback):
    """Callback for visualizing VAE reconstructions."""

    def __init__(self, batch):
        self.batch = batch

    def on_train_epoch_end(self, trainer, pl_module):
        """Visualize the images and reconstructions."""
        pl_module.eval()
        x_hat, *_ = pl_module(self.batch.to(pl_module.device))
        pl_module.log_image(
            {
                "reconstructions": draw_batch_and_reconstructions(
                    to_numpy(self.batch), to_numpy(x_hat)
                )
            }
        )
        pl_module.train()
