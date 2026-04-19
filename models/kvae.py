import torch
import torch.nn as nn
import sys
sys.path.append("..")
from config.vae_config import VAEConfig
from config.simulation_config import SimulationConfig
from config.train_config import TrainConfig
from models.base_vae import BaseVAE
from models.encoder import BallEncoder, ObstacleEncoder
from models.decoder import BallDecoder
from models.kalman_filter import KalmanFilter
from models.alphanetwork import AlphaNetwork


class KVAE(BaseVAE):
    def __init__(self, cfg: VAEConfig, sim_cfg: SimulationConfig, tcfg: TrainConfig):
        super().__init__(cfg, sim_cfg, tcfg)

        self.alpha_net = AlphaNetwork(cfg)
        self.kalman = KalmanFilter(cfg, tcfg)

        # State transition matrices [K, dim_z, dim_z]
        self.A_matrices = nn.Parameter(
            cfg.A_std * torch.randn(cfg.num_matrices, cfg.dim_z, cfg.dim_z) +
            (1 - cfg.A_std) * torch.eye(cfg.dim_z)
        )

        # Control matrices [K, dim_z, dim_u]
        if cfg.dim_u > 0:
            self.B_matrices = nn.Parameter(cfg.B_std * torch.randn(cfg.num_matrices, cfg.dim_z, cfg.dim_u))
        else:
            self.B_matrices = None

        # Observation matrices [K, dim_a, dim_z]
        self.C_matrices = nn.Parameter(
            cfg.C_std * torch.randn(cfg.num_matrices, cfg.dim_a, cfg.dim_z)   
        )

    def forward(self, ball_seq, obstacle_img, u_seq=None, mask=None, epoch=100, smoother=False):
        """
        ball_seq:      [B, T, H, W]          — sequence of ball images
        obstacle_img:  [B, H, W]             — static obstacle image
        u_seq:         [B, T, dim_u]         — control inputs
        mask:          [B, T]                — mask for valid timesteps

        a_dist:        Normal([B, T, dim_a]) — encoder distribution over latent observations
        a_seq:         [B, T, dim_a]         — sampled latent observations
        h_obs:         [B, dim_obstacle]     — obstacle context vector

        A_matrices:    [K, dim_z, dim_z]     — state transition matrices
        B_matrices:    [K, dim_z, dim_u]     — control matrices
        C_matrices:    [K, dim_a, dim_z]     — observation matrices

        z_smooth:      [B, T, dim_z]         — smoothed latent states
        P_smooth:      [B, T, dim_z, dim_z]  — smoothed covariances
        z_pred:        [B, T, dim_z]         — predicted latent states
        P_pred:        [B, T, dim_z, dim_z]  — predicted covariances
        a_smooth:      [B, T, dim_a]         — reconstructed observations (from z_smooth)
        a_pred_smooth: [B, T, dim_a]         — predicted observations
        alpha_seq:     [B, T, K]             — mixture weights over dynamics
        S_pred:        [B, T, dim_a, dim_a]  — predicted observation covariance

        x_dist_encoder:[B, T, H, W]          — reconstruction from encoder samples
        x_dist_smooth: [B, T, H, W]          — reconstruction from smoothed latents
        """

        # Encode
        a_dist = self.ball_encoder(ball_seq)                          
        h_obs = self.obstacle_encoder(obstacle_img.unsqueeze(1))  # [B, dim_obstacle]

        if self.training:
            a_seq = a_dist.rsample()  
        else:
            a_seq = a_dist.mean
        
        # Kalman filter
        z_smooth, P_smooth, z_dist, z_pred, P_pred, a_smooth, a_pred_smooth, alpha_seq = self.kalman(
            a_seq       = a_seq,
            alpha_net   = self.alpha_net,
            h_obs       = h_obs,
            A_matrices  = self.A_matrices,
            C_matrices  = self.C_matrices,
            B_matrices  = self.B_matrices,
            u_seq       = u_seq,
            mask        = mask,
            epoch       = epoch,
            smoother    = smoother
        )

        # Decode
        x_dist_smooth = self.decode(a_smooth)    # [B, T, H, W]
        x_dist_pred = self.decode(a_pred_smooth) # [B, T, H, W]

        return (
            x_dist_smooth,
            a_dist, a_seq, a_smooth, a_pred_smooth,
            z_dist, z_smooth, z_pred,
            self.kalman.R, self.kalman.Q, alpha_seq
        )

if __name__ == "__main__":


    cfg     = VAEConfig()
    sim_cfg = SimulationConfig()
    tcfg    = TrainConfig()
    model   = KVAE(cfg, sim_cfg, tcfg)

    B, T, H, W = 4, 20, 32, 32
    ball_seq     = torch.zeros(B, T, H, W)
    obstacle_img = torch.zeros(B, H, W)

    out = model(ball_seq, obstacle_img)
    print("x_hat_filt:", out[0].shape)
    print("x_hat_pred:", out[1].shape)
    print("z_filt:    ", out[5].shape)
    print("alpha_seq: ", out[9].shape)