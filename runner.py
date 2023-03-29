import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from codis.data import InfiniteDSprites, Latents
from codis.visualization import draw_batch_grid

from codis.models.beta_vae_v2 import BetaVAEV2
from typing import List, Optional
from codis.models.mlp import MLP


class ClassIncrementalInfiniteDSprites(InfiniteDSprites):
    """Infinite dataset of procedurally generated shapes undergoing transformations."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__seen_shapes = []
        self.change_class()
    
    def reset_to_round(self,r):
        self.__current_shape = self.__seen_shapes[r]

    @property
    def num_seen_classes(self):
        return len(self.__seen_shapes)
    
    def change_class(self):
        self.__current_shape = self.generate_shape()
        self.__seen_shapes.append(self.__current_shape)
    
    def sample_latents_for_shape(self, shape):
        """Sample a random set of latents."""
        return self.sample_latents()._replace(shape=shape)

    def __iter__(self):
        """Generate an infinite stream of images and latent vectors.
        Args:
            None
        Returns:
            An infinite stream of (image, latents) tuples."""
        while True:
            latents = self.sample_latents_for_shape(self.__current_shape)
            image   = self.draw(latents)
            yield image, latents


class CBetaVAE(BetaVAEV2):
    """The β-VAE model class."""

    def __init__(self, in_width: int = 64, in_channels: int = 1, latent_dim: int = 64, \
        num_channels: Optional[List] = None, beta: float = 1.0, H:int=128, L:int=2, act:str='elu') -> None:
        super().__init__(in_width=in_width, in_channels=in_channels, latent_dim=latent_dim, \
            num_channels=num_channels, beta=beta)
        self.regressor = MLP(latent_dim, 4, L=L, H=H, act=act)
    
    def forward(self, X, Z):
        ''' Inputs:
                X - [N,C,W,H]
                Z - Latents
        '''
        Xhat, mu, log_std = super().forward(X) # [N,C,W,H], [N,q], [N,q]
        Z = torch.stack([Z.scale, Z.orientation, Z.position_x, Z.position_y],-1) # N,4
        latent_reg_loss = (Z-self.regressor(mu)).pow(2).sum(-1).mean(0)
        reconstruction_loss = self.reconstruction_loss(X,Xhat)
        kl_divergence = self.kl_loss(mu,log_std)
        return latent_reg_loss, reconstruction_loss, kl_divergence
    
def eval_learning_so_far(model, dataset, data_loader, num_obs=250):
    with torch.no_grad():
        Nround = dataset.num_seen_classes
        for r in range(Nround):
            dataset.reset_to_round(r)
            losses = []
            for j,(X,Z) in enumerate(data_loader):
                # break if you have tested on num_obs data points
                if j*data_loader.batch_size > num_obs:
                    break
                latent_reg_loss, reconstruction_loss, kl_divergence = model(X, Z)
                losses.append([latent_reg_loss, reconstruction_loss, kl_divergence])
            # print the loss values
            losses = [np.mean([loss[i].item() for loss in losses]) for i in range(3)]
            print('\t Model evaluated on {:d} data points from round {:d}/{:d}. The losses are {:.2f}, {:.2f}, {:.2f}'\
                .format(data_loader.batch_size*j, r, Nround, losses[0], losses[1], losses[2]))
        

def train_loop(model, Nround=10, Nobs_per_round=10000, lambda_reg=10, image_size=64, \
                batch_size=64, num_eval_obs=250, print_times=10):
    dataset = ClassIncrementalInfiniteDSprites(image_size=image_size)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    N_ITERS_PER_ROUND = Nobs_per_round // batch_size + 1
    for round_ in range(Nround):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        print(f'Round {round_}/{Nround} is starting, we will run {N_ITERS_PER_ROUND} iterations.')
        losses = []
        for j,(X,Z) in enumerate(data_loader):
            # break if you have trained enough
            if j > N_ITERS_PER_ROUND:
                print(f'Round {round_}/{Nround} ended. We evaluate the learning so far.')
                break
            opt.zero_grad()
            latent_reg_loss, reconstruction_loss, kl_divergence = model(X, Z)
            vae_loss = reconstruction_loss + model.beta*kl_divergence
            loss = lambda_reg*latent_reg_loss + vae_loss
            losses.append([latent_reg_loss.item(), reconstruction_loss.item(), kl_divergence.item()])
            loss.backward()
            opt.step()
            if j % max(1,N_ITERS_PER_ROUND//print_times)==0:
                mean_losses = [np.mean([loss[i] for loss in losses]) for i in range(3)]
                print('\t Optimization iteration {:d}/{:d}. The losses are {:.2f}, {:.2f}, {:.2f}'\
                    .format(j, N_ITERS_PER_ROUND, mean_losses[0], mean_losses[1], mean_losses[2]))
                
        # evaluate the learning so far
        eval_learning_so_far(model, dataset, data_loader, num_obs=num_eval_obs)
        # move onto a new shape
        dataset.change_class()


parser = argparse.ArgumentParser(description='Continual Beta-VAE model with supervised regression for latents')
# continual learning pars
parser.add_argument('--n_round', type=int, default=10, help="number of shapes")

# model pars
parser.add_argument('--latent_dim', type=int, default=20, help="vae latent dim")
parser.add_argument('--n_channels', type=str, default='32-64-128-256', help="number of channels in enc/dec")
parser.add_argument('--image_size', type=int, default=64, help="image width")
parser.add_argument('--lambda_reg', type=float, default=10.0, help="weight of the supervised regression loss")
parser.add_argument('--beta', type=float, default=5.0, help="beta vae")

# data pars
parser.add_argument('--batch_size', type=int, default=21, help="sgd batch size")
parser.add_argument('--n_eval_obs', type=int, default=250, help="number of test points")
parser.add_argument('--n_obs_per_round', type=int, default=10000, help="number of train points in a round")

# log pars
parser.add_argument('--print_times', type=int, default=10, help="number of train points in a round")


args = parser.parse_args()
model = CBetaVAE(latent_dim=args.latent_dim, num_channels=list(map(int, args.n_channels.split('-'))), beta=args.beta)
train_loop(model, Nround=args.n_round, Nobs_per_round=args.n_obs_per_round, image_size=args.image_size, \
    batch_size=args.batch_size, lambda_reg=args.lambda_reg, num_eval_obs=args.n_eval_obs, print_times=args.print_times)
