import torch
import torch.nn as nn
import sys
sys.path.append("..")
from config.vae_config import VAEConfig
from config.simulation_config import SimulationConfig
from models.base_vae import BaseVAE


class GRUVAE(BaseVAE):
    """
    VAE + GRU prediction in latent space.
    """
    def __init__(self, cfg: VAEConfig, sim_cfg: SimulationConfig):
        super().__init__(cfg, sim_cfg)

        self.gru = nn.GRU(
            input_size  = cfg.dim_a + cfg.dim_obstacle,
            hidden_size = cfg.gru_hidden_dim,
            num_layers  = cfg.gru_layers,
            batch_first = True,
        )

        self.fc_pred = nn.Linear(cfg.gru_hidden_dim, cfg.dim_a)

    def forward(self, ball_seq, obstacle_img, u_seq=None, mask=None, phase=1):
        """
        ball_seq:     [B, T, H, W]
        obstacle_img: [B, H, W]
        """
        B, T, H, W = ball_seq.shape

        a_dist, h_obs = self.encode(ball_seq, obstacle_img)

        if self.training:
            a_seq = a_dist.rsample()  
        else:
            a_seq = a_dist.mean

        if mask is None:
            mask = torch.ones(B, T, device=ball_seq.device)

        a_filt_list = []
        a_pred_list = []
        h_state = None
        a_pred_k = torch.zeros(B, self.cfg.dim_a, device=ball_seq.device)

        for k in range(T):
            mask_k = mask[:, k].unsqueeze(-1)                              # [B, 1]
            a_prev = mask_k * a_seq[:, k, :] + (1 - mask_k) * a_pred_k
            a_filt_list.append(a_prev)

            gru_input = torch.cat([a_prev, h_obs], dim=-1).unsqueeze(1)    # [B, 1, dim_a+dim_obs]
            gru_out, h_state = self.gru(gru_input, h_state)                # [B, 1, hidden]
            a_pred_k = self.fc_pred(gru_out.squeeze(1))                    # [B, dim_a]

            a_pred_list.append(a_pred_k)                                   

        a_filt = torch.stack(a_filt_list, dim=1)    # [B, T, dim_a]
        a_pred = torch.stack(a_pred_list, dim=1)    # [B, T, dim_a]

        x_dist_filt = self.decode(a_filt)
        x_dist_pred = self.decode(a_pred)

        return (x_dist_filt, x_dist_pred, 
                a_dist, a_seq, a_filt, a_pred,
                None, None, None, None, 
                None, None, None)

        