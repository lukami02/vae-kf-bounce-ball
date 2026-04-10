import torch
import torch.nn as nn
import sys
sys.path.append("..")
from config.vae_config import VAEConfig
from config.simulation_config import SimulationConfig
from models.base_vae import BaseVAE
from models.encoder import BallEncoder, ObstacleEncoder
from models.decoder import BallDecoder
from models.kalman_filter import KalmanFilter
from models.alphanetwork import AlphaNetwork


class KVAE(BaseVAE):
    def __init__(self, cfg: VAEConfig, sim_cfg: SimulationConfig):
        super().__init__(cfg, sim_cfg)

        self.alpha_net        = AlphaNetwork(cfg, obstacle=True)
        self.kalman           = KalmanFilter(cfg)

        # State transition matrices [K, dim_z, dim_z]
        self.A_matrices = nn.Parameter(
            (0.05) * torch.randn(cfg.num_matrices, cfg.dim_z, cfg.dim_z) +
            0.95 * torch.eye(cfg.dim_z)
            #self.init_A_matrices()
        )
        
        # Observation matrices [K, dim_a, dim_z]
        self.C_matrices = nn.Parameter(
            (0.05) * torch.randn(cfg.num_matrices, cfg.dim_a, cfg.dim_z)   
            #0.1 * torch.eye(cfg.dim_a, cfg.dim_z)
           #self.init_C_matrices()
        )

        # Control matrices [K, dim_z, dim_u]
        if cfg.dim_u > 0:
            self.B_matrices = nn.Parameter(0.05 * torch.randn(cfg.num_matrices, cfg.dim_z, cfg.dim_u))
        else:
            self.B_matrices = None

    def forward(self, ball_seq, obstacle_img, u_seq=None, mask=None, epoch=100, phase=1):
       
        B, T, H, W = ball_seq.shape

        # Encode
        a_dist = self.ball_encoder(ball_seq)                            # a_seq: [B, T, dim_a]
        h_obs = self.obstacle_encoder(obstacle_img.unsqueeze(1))        # [B, dim_obstacle]

        if self.training:
            a_seq = a_dist.rsample()  
        else:
            a_seq = a_dist.mean

        if phase == 0:
            x_dist_filt = self.decode(a_seq)

            return (
                x_dist_filt, None,
                a_dist, a_seq, None, None,
                None, None, None, None, None, 
                None, None, None
            )

        # Kalman filter
        z_smooth, P_smooth, z_dist, z_pred, P_pred, a_smooth, a_pred_smooth, alpha_seq, S_pred = self.kalman(
            a_seq       = a_seq,
            alpha_net   = self.alpha_net,
            h_obs       = h_obs,
            A_matrices  = self.A_matrices,
            C_matrices  = self.C_matrices,
            B_matrices  = self.B_matrices,
            u_seq       = u_seq,
            mask        = mask,
            epoch       = epoch
        )

        # Decode
        x_dist_smooth = self.decode(a_seq)   # [B, T, H, W]
        x_dist_pred = self.decode(a_smooth)   # [B, T, H, W]

        return (
            x_dist_smooth, x_dist_pred,
            a_dist, a_seq, a_smooth, a_pred_smooth,
            z_dist, z_smooth, P_smooth, z_pred, P_pred,
            self.kalman.R, self.kalman.Q, alpha_seq
        )


if __name__ == "__main__":


    cfg     = VAEConfig()
    sim_cfg = SimulationConfig()
    model   = KVAE(cfg, sim_cfg)

    B, T, H, W = 4, 20, 32, 32
    ball_seq     = torch.zeros(B, T, H, W)
    obstacle_img = torch.zeros(B, H, W)

    out = model(ball_seq, obstacle_img)
    print("x_hat_filt:", out[0].shape)
    print("x_hat_pred:", out[1].shape)
    print("z_filt:    ", out[5].shape)
    print("alpha_seq: ", out[9].shape)