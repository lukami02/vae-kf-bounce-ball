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

        self.alpha_net        = AlphaNetwork(cfg)
        self.kalman           = KalmanFilter(cfg)

        # State transition matrices [K, dim_z, dim_z]
        self.A_matrices = nn.Parameter(
            self.init_A_matrices()
        )
        
        # Observation matrices [K, dim_a, dim_z]
        self.C_matrices = nn.Parameter(
            self.init_C_matrices()
        )

        # Control matrices [K, dim_z, dim_u]
        if cfg.dim_u > 0:
            self.B_matrices = nn.Parameter(cfg.B_std * torch.randn(cfg.num_matrices, cfg.dim_z, cfg.dim_u))
        else:
            self.B_matrices = None

    def init_A_matrices(self, dt=1.0):
        K, dim_z = self.cfg.num_matrices, self.cfg.dim_z
        A = torch.zeros(K, dim_z, dim_z)
        
        # Sve matrice krecu od identiteta
        for k in range(K):
            A[k] = torch.eye(dim_z)
        
        # Mode 0
        half = dim_z // 2
        for i in range(half):
            A[0, i, i + half] = dt  # p_i += v_i * dt

        # Mode 1
        A[1] = A[0].clone()
        quarter = max(1, half // 2)
        for i in range(quarter):
            A[1, i + half, i + half] = - dt

        # Mode 2
        if K > 2:
            A[2] = A[0].clone()
            for i in range(quarter, half):
                A[2, i + half, i + half] = - dt

        for k in range(3, K):
            A[k] = A[0].clone()
            A[k] += self.cfg.A_std * torch.randn(dim_z, dim_z)

        return A

    def init_C_matrices(self):
        C = torch.zeros(self.cfg.num_matrices, self.cfg.dim_a, self.cfg.dim_z)
        for i in range(self.cfg.num_matrices):
            for j in range(min(self.cfg.dim_a, self.cfg.dim_z)):
                C[i, j, j] = 1.0
        C += self.cfg.C_std * torch.randn_like(C)
        return C

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
                None, None, None, None, None, None, None,
                None, None, None, None, None
            )

        # Kalman filter
        z_filt, P_filt, z_dist, z_pred, P_pred, a_filt, a_pred, S_pred, alpha_seq, alpha_imm, R, Q = self.kalman(
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
        x_dist_filt = self.decode(a_filt)   # [B, T, H, W]
        x_dist_pred = self.decode(a_pred)   # [B, T, H, W]

        return (
            x_dist_filt, x_dist_pred,
            a_dist, a_seq, a_filt, a_pred,
            z_dist, self.kalman.z_0, self.kalman.P_0,
            z_filt, P_filt, z_pred, P_pred,
            S_pred, alpha_seq, alpha_imm, R, Q
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