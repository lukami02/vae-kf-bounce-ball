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

    def forward(self, ball_seq, obstacle_img, u_seq=None, mask=None, epoch=100, phase=1):
        B, T, H, W = ball_seq.shape

        a_dist, h_obs = self.encode(ball_seq, obstacle_img)   # [B, T, dim_a]

        if self.training:
            a_seq = a_dist.rsample()  
        else:
            a_seq = a_dist.mean

        if mask is None:
            mask = torch.ones(B, T, device=ball_seq.device)

        a_filt_list = []
        a_pred_list = []
        a_pred_k = torch.zeros(B, self.cfg.dim_a, device=ball_seq.device)  # placeholder
        velocity = torch.zeros(B, self.cfg.dim_a, device=ball_seq.device)
        a_prev   = torch.zeros(B, self.cfg.dim_a, device=ball_seq.device)

        for k in range(T):
            mask_k   = mask[:, k].unsqueeze(-1)
            a_filt_k = mask_k * a_seq[:, k, :] + (1 - mask_k) * a_pred_k
            a_filt_list.append(a_filt_k)

            velocity = mask_k * (a_filt_k - a_prev) + (1 - mask_k) * velocity
            a_prev   = a_filt_k

            a_pred_k = a_prev + velocity        
            a_pred_list.append(a_pred_k)

        a_filt = torch.stack(a_filt_list, dim=1)                          # [B, T, dim_a]
        a_pred = torch.stack(a_pred_list, dim=1)                          # [B, T, dim_a]

        x_dist_filt = self.decode(a_filt)
        x_dist_pred = self.decode(a_pred)

        return (x_dist_filt, x_dist_pred, 
            a_dist, a_seq, a_filt, a_pred,
            None, None, None, None, 
            None, None, None, None)
        