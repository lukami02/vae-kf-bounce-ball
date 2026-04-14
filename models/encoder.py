import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
import sys

sys.path.append("..")
from config.vae_config import VAEConfig
from config.simulation_config import SimulationConfig


class BallEncoder(nn.Module):
    """
    Encodes ball image sequence into latent distribution.
    """

    def __init__(self, vae_cfg: VAEConfig, sim_cfg: SimulationConfig):
        super().__init__()
        self.cfg = vae_cfg

        # input channels
        channels = [1] + vae_cfg.encoder_ball_channels

        # Network layers
        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1),
                vae_cfg.enc_activation(),
            ]

        self.cnn = nn.Sequential(*layers)

        # Calculate flattened size after CNN
        self._flat_size = self._get_flat_size(sim_cfg.size)

        # Output heads for distribution parameters
        self.fc_mu = nn.Linear(self._flat_size, vae_cfg.dim_a)
        self.fc_var = nn.Linear(self._flat_size, vae_cfg.dim_a)

    def _get_flat_size(self, image_size):
        dummy = torch.zeros(1, 1, image_size[0], image_size[1])
        with torch.no_grad():
            out = self.cnn(dummy)
        return out.view(1, -1).shape[1]

    def forward(self, x):
        """
        x: [B, T, H, W] — Sequence of grayscale images
        
        dist: torch.distributions.Normal — Latent distribution with shape [B, T, dim_a]
        """
        B, T, H, W = x.shape
        x_flat = x.view(B * T, 1, H, W)

        enc = self.cnn(x_flat)
        enc = enc.view(B * T, -1)

        # Compute Mean and Standard Deviation
        a_mu = self.fc_mu(enc)
        a_std = self.cfg.R_std * torch.sigmoid(self.fc_var(enc)) + 1e-6

        a_mu = a_mu.view(B, T, self.cfg.dim_a)
        a_std = a_std.view(B, T, self.cfg.dim_a)

        return D.Normal(loc=a_mu, scale=a_std)

class ObstacleEncoder(nn.Module):
    """
    Encodes static obstacle image into context vector.
    """
    def __init__(self, vae_cfg: VAEConfig, sim_cfg: SimulationConfig, ball_encoder: BallEncoder):
        super().__init__()

        # Share the CNN backbone with ball_encoder
        self.cnn = ball_encoder.cnn

        # Calculate flattened size after CNN
        self._flat_size = self._get_flat_size(sim_cfg.size)

        # Project visual features to GRU hidden dimension
        self.feature_proj = nn.Linear( self._flat_size, vae_cfg.dim_obstacle)

    def _get_flat_size(self, image_size):
        dummy = torch.zeros(1, 1, image_size[0], image_size[1])
        with torch.no_grad():
            out = self.cnn(dummy)
        return out.view(1, -1).shape[1]

    def forward(self, obs_img):
        """
        obs_img:  [B, 1, H, W] — Static image of obstacles

        features: [B, dim_obstacle] — Latent context vector
        """
        B = obs_img.shape[0]
        
        # Detach ensures CNN is only trained via BallEncoder
        features = self.cnn(obs_img).detach() # [B, C, H', W']

        # Flatten and project to hidden space
        features = features.view(B, -1)
        features = self.feature_proj(features) 

        return features