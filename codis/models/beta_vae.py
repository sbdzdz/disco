"""The β-VAE model. See https://openreview.net/forum?id=Sy2fzU9gl for details."""
from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from codis.models.base_vae import BaseVAE
from codis.models.blocks import (
    build_decoder_block,
    build_encoder_block,
    build_final_layer,
)


class BetaVAE(BaseVAE):
    """The β-VAE model class."""

    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_dims: Optional[List] = None,
        beta: int = 4,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        max_iter: int = int(1e5),
        loss_type: str = "B",
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.max_capacity = torch.Tensor([max_capacity])
        self.max_iter = max_iter

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        hidden_dims = [in_channels] + hidden_dims

        # Encoder
        encoder_modules = [
            build_encoder_block(in_channels, out_channels)
            for in_channels, out_channels in zip(hidden_dims[:-1], hidden_dims[1:])
        ]
        self.encoder = nn.Sequential(*encoder_modules)

        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        # Decoder
        decoder_modules = [
            build_decoder_block(in_channels, out_channels)
            for in_channels, out_channels in zip(hidden_dims[:-1], hidden_dims[1:])
        ]
        self.decoder = nn.Sequential(*decoder_modules)
        self.final_layer = build_final_layer(hidden_dims[-1], hidden_dims[-1])

    def encode(self, x: Tensor) -> List[Tensor]:
        """Pass the input through the encoder network and return the latent code.
        Args:
            x: Input tensor [N x C x H x W]
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
            z: Latent code tensor [N x D]
        Returns:
            Reconstructed input [N x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor) -> List[Tensor]:
        """Pass the input through the network and return the reconstructed input and latent code."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)

    def loss_function(
        self,
        x: Tensor,
        x_hat: Tensor,
        mu: Tensor,
        log_var: Tensor,
        kld_weight: float,
    ) -> dict:
        """Compute the loss given reconstructions and ground truth images."""
        self.num_iter += 1
        recons_loss = F.mse_loss(x_hat, x)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        if self.loss_type == "H":
            loss = recons_loss + self.beta * kld_weight * kld_loss

        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def reconstruct(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
