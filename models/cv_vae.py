import torch
import copy
import sys
sys.path.append("..")
from config.vae_config import VAEConfig
from config.simulation_config import SimulationConfig
from models.base_vae import BaseVAE
from models.kvae import KVAE


class CVVAE(BaseVAE):
    """
    VAE + Constant Velocity baseline.
    a_{t+1} = a_t + (a_t - a_{t-1})
    """
    def __init__(self, cfg: VAEConfig, sim_cfg: SimulationConfig, kvae: KVAE = None):
        super().__init__(cfg, sim_cfg)
        if kvae:
            self.ball_encoder = copy.deepcopy(kvae.ball_encoder)
            self.decoder = copy.deepcopy(kvae.decoder)

    def forward(self, ball_seq, obstacle_img=None, u_seq=None, mask=None, epoch=100):
        """
        ball_seq:      [B, T, H, W]          — sequence of ball images
        mask:          [B, T]                — mask for valid timesteps
        """
        B, T, H, W = ball_seq.shape

        a_dist = self.ball_encoder(ball_seq)   # [B, T, dim_a]

        if self.training:
            a_seq = a_dist.rsample()  
        else:
            a_seq = a_dist.mean

        if mask is None:
            mask = torch.ones(B, T, device=ball_seq.device)

        a_list = []
        a_pred_list = []
        a_pred_k = torch.zeros(B, self.cfg.dim_a, device=ball_seq.device)  
        velocity = torch.zeros(B, self.cfg.dim_a, device=ball_seq.device)
        a_prev   = torch.zeros(B, self.cfg.dim_a, device=ball_seq.device)

        is_first = True
        for k in range(T):
            mask_k = mask[:, k].unsqueeze(-1)
            a_k = mask_k * a_seq[:, k, :] + (1 - mask_k) * a_pred_k
            a_list.append(a_k)

            if is_first:
                velocity = torch.zeros_like(a_k)
                is_first = False
            else:
                velocity = mask_k * (a_k - a_prev) + (1 - mask_k) * velocity
            a_prev = a_k

            a_pred_k = a_prev + velocity
            a_pred_list.append(a_pred_k)

        a_s = torch.stack(a_list, dim=1)            # [B, T, dim_a]
        a_pred = torch.stack(a_pred_list, dim=1)    # [B, T, dim_a]

        x_dist = self.decode(a_s)
        x_dist_pred = self.decode(a_pred)

        return (x_dist, 
            a_dist, a_seq, a_s, a_pred,
            None, None, None, 
            None, None, None)