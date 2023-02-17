"""Base VAE class. Adapted from https://github.com/AntixK/PyTorch-VAE."""
from abc import abstractmethod
from typing import Any, List

from torch import Tensor, nn


class BaseVAE(nn.Module):
    """Base class for all VAEs."""

    @abstractmethod
    def encode(self, x: Tensor) -> List[Tensor]:
        """Encode an input vector into the latent space."""

    @abstractmethod
    def decode(self, z: Tensor) -> Any:
        """Decode a latent vector."""

    @abstractmethod
    def reconstruct(self, x: Tensor, **kwargs) -> Tensor:
        """Reconstruct the input vector."""

    @abstractmethod
    def forward(self, x: Tensor) -> List[Tensor]:
        """Perform the forward pass."""

    @abstractmethod
    def loss_function(
        self, x: Tensor, x_hat: Tensor, mu: Tensor, log_var: Tensor
    ) -> dict:
        """Calculate the loss function for the model."""
