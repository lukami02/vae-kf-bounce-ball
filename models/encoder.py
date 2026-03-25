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
    Encodes ball image sequence + obstacle image into latent distribution.
    """

    def __init__(self, vae_cfg: VAEConfig, sim_cfg: SimulationConfig):
        super().__init__()
        self.cfg = vae_cfg

        # input channels
        channels = [2] + vae_cfg.encoder_ball_channels

        layers = []
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=1, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                vae_cfg.enc_activation(),
                nn.MaxPool2d(2, 2)
            ]

        self.cnn = nn.Sequential(*layers)

        # compute flat size
        self._flat_size = self._get_flat_size(sim_cfg.size)

        self.fc_mu = nn.Linear(self._flat_size, vae_cfg.dim_a)
        self.fc_var = nn.Linear(self._flat_size, vae_cfg.dim_a)

    def _get_flat_size(self, image_size):
        dummy = torch.zeros(1, 2, image_size[0], image_size[1])
        out = self.cnn(dummy)
        return out.view(1, -1).shape[1]

    def reparametrize(self, mu, var):
        if self.training:
            std = torch.sqrt(var + 1e-8)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, obs_img):
        """
        x: [B, T, H, W]
        obs_img: [B, 1, H, W]
        """
        B, T, H, W = x.shape

        obs_seq = obs_img.unsqueeze(1).expand(-1, T, -1, -1, -1)  # [B,T,1,H,W]

        # add channel dim to x
        x = x.unsqueeze(2)                  # [B,T,1,H,W]
        x = torch.cat([x, obs_seq], dim=2)  # [B,T,2,H,W]
        x_flat = x.view(B * T, 2, H, W)
        enc = self.cnn(x_flat)
        enc = enc.view(B * T, -1)

        a_mu = self.fc_mu(enc)
        a_std = F.softplus(self.fc_var(enc)) + 1e-6
        #a = self.reparametrize(a_mu, a_var)
        a_mu = a_mu.view(B, T, self.cfg.dim_a)
        a_std = a_std.view(B, T, self.cfg.dim_a)
        return D.Normal(loc=a_mu, scale=a_std)

class ObstacleEncoder(nn.Module):
    """
    Encodes static obstacle image into context vector.
    """
    def __init__(self, vae_cfg: VAEConfig, sim_cfg: SimulationConfig):
        super().__init__()
        channels = [1] + vae_cfg.encoder_obstacle_channels
        layers = []
        for i in range(len(channels)-1):
            layers += [
                nn.Conv2d(channels[i], channels[i + 1], 3, stride=1, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                vae_cfg.enc_activation(),
                nn.MaxPool2d(2, 2)
            ]
        self.cnn = nn.Sequential(*layers)
        
        self.feature_proj = nn.Conv2d(vae_cfg.encoder_obstacle_channels[-1], vae_cfg.alpha_units, 1)

    def forward(self, obs_img):
        # obs_img: [B, 1, H, W]
        features = self.cnn(obs_img) # [B, C, H', W']
        features = self.feature_proj(features) # [B, alpha_units, H', W']
        
        B, C, H, W = features.shape
        features = features.view(B, C, -1).transpose(1, 2)
        return features