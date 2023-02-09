"""Base VAE class. Adapted from https://github.com/AntixK/PyTorch-VAE."""
from abc import abstractmethod
from typing import Any, List

from torch import Tensor, nn


class BaseVAE(nn.Module):
    """Base class for all VAEs."""

    def encode(self, x: Tensor) -> List[Tensor]:
        """Encode the input into the latent space."""
        raise NotImplementedError

    def decode(self, x: Tensor) -> Any:
        """Decode the input from the latent space."""
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """Generate a sample from the latent space."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        """Forward pass of the model."""

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        """Loss function for the model."""
