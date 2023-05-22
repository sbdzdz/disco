"""Lightning callbacks."""
from lightning.pytorch.callbacks import Callback

from codis.visualization import draw_batch_and_reconstructions
from codis.utils import to_numpy


class VisualizationCallback(Callback):
    """Callback for visualizing VAE reconstructions."""

    def __init__(self, batch):
        super().__init__()
        self.batch = batch

    def on_train_epoch_end(self, trainer, pl_module):
        """Visualize the images and reconstructions."""
        pl_module.eval()
        x_hat, *_ = pl_module(self.batch.to(pl_module.device))
        images = draw_batch_and_reconstructions(to_numpy(self.batch), to_numpy(x_hat))
        pl_module.logger.log_image("reconstructions", images=[images])
        pl_module.train()


class LoggingCallback(Callback):
    """Callback for additional logging."""

    def on_train_epoch_start(self, trainer, pl_module):
        print(f"Starting task {pl_module.train_task_id}...")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.log("task_id", pl_module.train_task_id)
