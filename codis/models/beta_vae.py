"""The β-VAE model. See https://openreview.net/forum?id=Sy2fzU9gl for details."""
from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from codis.models.base_vae import BaseVAE
from codis.models.blocks import Encoder, Decoder


class BetaVAE(BaseVAE):
    """The β-VAE model class."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        hidden_dims: Optional[List] = None,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.encoder = Encoder(hidden_dims, in_channels)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        self.decoder = Decoder(list(reversed(hidden_dims)), in_channels)

    def encode(self, x: Tensor) -> List[Tensor]:
        """Pass the input through the encoder network and return the latent code.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            List of latent codes
        """
        result = self.encoder(x).flatten(start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """Pass the latent code through the decoder network and return the reconstructed input.
        Args:
            z: Latent code tensor of shape (B x D)
        Returns:
            Reconstructed input of shape (B x C x H x W)
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Perform the reparameterization trick.
        Args:
            mu: Mean of the latent Gaussian of shape (N x D)
            logvar: Standard deviation of the latent Gaussian of shape (N x D)
        Returns:
            Sampled latent vector [N x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> List[Tensor]:
        """Perform the forward pass.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            List of tensors [reconstructed input, latent mean, latent log variance]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var]

    def loss_function(
        self,
        x: Tensor,
        x_hat: Tensor,
        mu: Tensor,
        log_var: Tensor,
    ) -> dict:
        """Compute the loss given ground truth images and their reconstructions.
        Args:
            x: Ground truth images of shape (B x C x H x W)
            x_hat: Reconstructed images of shape (B x C x H x W)
            mu: Latent mean of shape (B x D)
            log_var: Latent log variance of shape (B x D)
            kld_weight: Weight for the Kullback-Leibler divergence term
        Returns:
            Dictionary containing the loss value and the individual losses.
        """
        reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum") / x.size(0)

        kl_divergence = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = reconstruction_loss + self.beta * kl_divergence

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence": kl_divergence,
        }

    def sample(self, num_samples: int, current_device: int) -> Tensor:
        """Sample a vector in the latent space and return the corresponding image.
        Args:
            num_samples: Number of samples to generate
            current_device: Device to run the model
        Returns:
            Tensor of shape (num_samples x C x H x W)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        return self.decode(z)

    def reconstruct(self, x: Tensor, **kwargs) -> Tensor:
        """Given an input image x, returns the reconstructed image.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            Reconstructed input of shape (B x C x H x W)
        """
        return self.forward(x)[0]
