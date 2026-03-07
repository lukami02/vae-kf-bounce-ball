import torch
import torch.nn as nn
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

        channels = [1] + vae_cfg.encoder_ball_channels
        layers = []
        for i in range(len(channels)-1):
            layers += [nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1), vae_cfg.enc_activation()]
        self.cnn = nn.Sequential(*layers)

        # Compute flat CNN output size dynamically
        self._flat_size = self._get_flat_size(sim_cfg.size)

        self.fc_mu  = nn.Linear(self._flat_size, vae_cfg.dim_a)
        self.fc_var = nn.Linear(self._flat_size, vae_cfg.dim_a)

    def _get_flat_size(self, image_size):
        dummy = torch.zeros(1, 1, image_size[0], image_size[1])
        out = self.cnn(dummy)
        return out.view(1, -1).shape[1]

    def reparametrize(self, mu, var):
        if self.training:
            std = torch.sqrt(var + 1e-8)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        B, T, H, W = x.shape

        x_flat = x.view(B * T, 1, H, W)
        enc = self.cnn(x_flat)
        enc = enc.view(B * T, -1)

        a_mu = self.fc_mu(enc)                                    # [B*T, dim_a]
        a_var = torch.sigmoid(self.fc_var(enc)) * self.cfg.R_std  # [B*T, dim_a]
        a = self.reparametrize(a_mu, a_var)                       # [B*T, dim_a]

        a_seq = a.view(B, T, self.cfg.dim_a)                      # [B, T, dim_a]
        return a_seq, a_mu, a_var


class ObstacleEncoder(nn.Module):
    """
    Encodes static obstacle image into context vector.
    """
    def __init__(self, vae_cfg: VAEConfig, sim_cfg: SimulationConfig):
        super().__init__()

        channels = [1] + vae_cfg.encoder_obstacle_channels
        layers = []
        for i in range(len(channels)-1):
            layers += [nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1), vae_cfg.enc_activation()]
        self.cnn = nn.Sequential(*layers)

        # Compute flat CNN output size dynamically
        self._flat_size = self._get_flat_size(sim_cfg.size)

        self.fc = nn.Sequential(
            nn.Linear(self._flat_size, vae_cfg.dim_obstacle),
            vae_cfg.enc_activation(),
        )

    def _get_flat_size(self, image_size):
        dummy = torch.zeros(1, 1, image_size[0], image_size[1])
        out = self.cnn(dummy)
        return out.view(1, -1).shape[1]

    def forward(self, obs_img):
        B = obs_img.shape[0]
        enc = self.cnn(obs_img)
        enc = enc.view(B, -1)
        return self.fc(enc)  # [B, obs_hidden_dim]