import torch
import torch.nn as nn
import sys
sys.path.append("..")
from config.vae_config import VAEConfig
from config.simulation_config import SimulationConfig
from models.encoder import BallEncoder, ObstacleEncoder
from models.decoder import BallDecoder


class BaseVAE(nn.Module):
    """
    Base class for VAE models. Contains shared encoder and decoder.
    """
    def __init__(self, cfg: VAEConfig, sim_cfg: SimulationConfig):
        super().__init__()
        self.cfg     = cfg
        self.sim_cfg = sim_cfg

        self.ball_encoder = BallEncoder(cfg, sim_cfg)
        self.obstacle_encoder = ObstacleEncoder(cfg, sim_cfg)
        self.decoder = BallDecoder(cfg, sim_cfg)

    def encode(self, ball_seq, obstacle_img):
        """
        Encode ball sequence and obstacle image.
        """
        a_dist = self.ball_encoder(ball_seq, obstacle_img.unsqueeze(1))
        h_obs = self.obstacle_encoder(obstacle_img.unsqueeze(1))  # [B, dim_obstacle]
        return a_dist, h_obs

    def decode(self, a_seq):
        """
        Decode latent sequence to ball images.
        """
        return self.decoder(a_seq)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dim_a={self.cfg.dim_a}, "
            f"dim_z={self.cfg.dim_z}, "
            f"params={self.count_parameters():,})"
        )