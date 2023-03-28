"""The β-VAE model. See https://openreview.net/forum?id=Sy2fzU9gl for details."""
from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from codis.models.base_vae import BaseVAE
from codis.models.blocks import Encoder, Decoder

from torch.distributions import MultivariateNormal, Normal, kl_divergence as kl


class BetaVAEV2(BaseVAE):
    """The β-VAE model class."""

    def __init__(
        self,
        in_width: int = 64,
        in_channels: int = 1,
        latent_dim: int = 64,
        num_channels: Optional[List] = None,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.sp = nn.Softplus()

        if num_channels is None:
            num_channels = [32, 64, 128, 256, 512]
        self.w = in_width // 2**len(num_channels)

        self.encoder = Encoder(num_channels, in_channels)
        self.fc_mu   = nn.Linear(num_channels[-1] * self.w**2, latent_dim)
        self.fc_std  = nn.Linear(num_channels[-1] * self.w**2, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, num_channels[-1] * self.w**2)
        self.decoder = Decoder(list(reversed(num_channels)), in_channels)

    def transform(self, log_std):
        # return self.sp(log_std) # this is actually numerically more stable
        return log_std.exp()

    def encode(self, x: Tensor) -> List[Tensor]:
        """Pass the input through the encoder network and return the latent code.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            List of latent codes
        """
        result = self.encoder(x).flatten(start_dim=1)
        mu = self.fc_mu(result)
        log_std = self.fc_std(result)
        return [mu, log_std]

    def decode(self, z: Tensor) -> Tensor:
        """Pass the latent code through the decoder network and return the reconstructed input.
        Args:
            z: Latent code tensor of shape (B x D)
        Returns:
            Reconstructed input of shape (B x C x H x W)
        """
        result = self.decoder_input(z)
        result = result.view(z.shape[0], -1, self.w, self.w)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, log_std: Tensor) -> Tensor:
        """Perform the reparameterization trick.
        Args:
            mu: Mean of the latent Gaussian of shape (N x D)
            log_std: Standard deviation of the latent Gaussian of shape (N x D)
        Returns:
            Sampled latent vector [N x D]
        """
        std = self.transform(log_std)
        eps = torch.randn_like(mu) 
        return mu + std*eps

    def forward(self, x: Tensor) -> List[Tensor]:
        """Perform the forward pass.
        Args:
            x: Input tensor of shape (B x C x H x W)
        Returns:
            List of tensors [reconstructed input, latent mean, latent std]
        """
        mu, log_std = self.encode(x)
        z = self.reparameterize(mu, log_std)
        return [self.decode(z), mu, self.transform(log_std)]

    def loss_function(
        self,
        x: Tensor,
        x_hat: Tensor,
        mu: Tensor,
        log_std: Tensor,
    ) -> dict:
        """Compute the loss given ground truth images and their reconstructions.
        Args:
            x: Ground truth images of shape (B x C x H x W)
            x_hat: Reconstructed images of shape (B x C x H x W)
            mu: Latent mean of shape (B x D)
            log_std: Latent log std of shape (B x D)
            kld_weight: Weight for the Kullback-Leibler divergence term
        Returns:
            Dictionary containing the loss value and the individual losses.
        """
        reconstruction_loss = (x_hat-x).pow(2).sum() / x.shape[0]

        std = self.transform(log_std)
        q = Normal(mu,std)
        p = Normal(torch.zeros_like(mu),torch.ones_like(std))
        kl_divergence = kl(q,p).mean()

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
