import torch
import sys
sys.path.append("..")
from config.vae_config import VAEConfig
from config.simulation_config import SimulationConfig
from models.base_vae import BaseVAE


class CVVAE(BaseVAE):
    """
    VAE + Constant Velocity baseline.
    a_{t+1} = a_t + (a_t - a_{t-1})
    """
    def __init__(self, cfg: VAEConfig, sim_cfg: SimulationConfig):
        super().__init__(cfg, sim_cfg)

    def forward(self, ball_seq, obstacle_img, u_seq=None, mask=None):
        B, T, H, W = ball_seq.shape

        a_seq, a_mu, a_var, h_obs = self.encode(ball_seq, obstacle_img)   # [B, T, dim_a]

        if mask is None:
            mask = torch.ones(B, T, device=ball_seq.device)

        a_filt_list = []
        a_pred_list = []
        a_prev      = a_seq[:, 0, :]                                      # [B, dim_a]
        velocity    = torch.zeros_like(a_prev)                            # [B, dim_a]

        for k in range(T):
            mask_k = mask[:, k].unsqueeze(-1)                             # [B, 1]

            a_pred_k = a_prev + velocity                                  # [B, dim_a]

            a_filt_k = mask_k * a_seq[:, k, :] + (1 - mask_k) * a_pred_k  # [B, dim_a]

            velocity = mask_k * (a_filt_k - a_prev) + (1 - mask_k) * velocity

            a_prev = a_filt_k

            a_filt_list.append(a_filt_k)
            a_pred_list.append(a_pred_k)

        a_filt = torch.stack(a_filt_list, dim=1)                          # [B, T, dim_a]
        a_pred = torch.stack(a_pred_list, dim=1)                          # [B, T, dim_a]

        x_hat_filt = self.decode(a_filt)
        x_hat_pred = self.decode(a_pred)

        return (x_hat_filt, x_hat_pred, 
                a_seq, a_mu, a_var, a_filt,
                None, None, None, None, 
                None)