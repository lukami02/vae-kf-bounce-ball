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

    def forward(self, ball_seq, obstacle_img, u_seq=None, mask=None):
        """
        ball_seq:     [B, T, H, W]
        obstacle_img: [B, H, W]
        """
        B, T, H, W = ball_seq.shape

        a_seq, a_mu, a_var, h_obs = self.encode(ball_seq, obstacle_img)

        if mask is None:
            mask = torch.ones(B, T, device=ball_seq.device)

        a_filt_list = []
        a_pred_list = []
        h_state = None              # GRU hidden state
        a_prev = a_seq[:, 0, :]     # [B, dim_a]

        for k in range(T):
            mask_k = mask[:, k].unsqueeze(-1)                              # [B, 1]

            gru_input = torch.cat([a_prev, h_obs], dim=-1).unsqueeze(1)    # [B, 1, dim_a+dim_obs]
            gru_out, h_state = self.gru(gru_input, h_state)                # [B, 1, hidden]
            a_pred_k = self.fc_pred(gru_out.squeeze(1))                    # [B, dim_a]

            a_filt_k = mask_k * a_seq[:, k, :] + (1 - mask_k) * a_pred_k   # [B, dim_a]

            a_prev = a_filt_k

            a_filt_list.append(a_filt_k)
            a_pred_list.append(a_pred_k)

        a_filt = torch.stack(a_filt_list, dim=1)    # [B, T, dim_a]
        a_pred = torch.stack(a_pred_list, dim=1)    # [B, T, dim_a]

        x_hat_filt = self.decode(a_filt)
        x_hat_pred = self.decode(a_pred)

        return (x_hat_filt, x_hat_pred, 
                a_seq, a_mu, a_var, a_filt,
                None, None, None, None, 
                None)