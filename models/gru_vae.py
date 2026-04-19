import torch
import torch.nn as nn
import copy
import sys
sys.path.append("..")
from config.vae_config import VAEConfig
from config.simulation_config import SimulationConfig
from config.train_config import TrainConfig
from models.base_vae import BaseVAE
from models.kvae import KVAE


class GRUVAE(BaseVAE):
    """
    VAE + GRU prediction in latent space.
    """
    def __init__(self, cfg: VAEConfig, sim_cfg: SimulationConfig, tcfg: TrainConfig, kvae: KVAE = None):
        super().__init__(cfg, sim_cfg, tcfg)
        if kvae:
            self.ball_encoder = copy.deepcopy(kvae.ball_encoder)
            self.decoder = copy.deepcopy(kvae.decoder)

        self.gru = nn.GRU(
            input_size  = cfg.dim_a + cfg.dim_obstacle + cfg.dim_u,
            hidden_size = cfg.gru_hidden_dim,
            num_layers  = cfg.gru_layers,
            batch_first = True,
        )

        self.fc_pred = nn.Linear(cfg.gru_hidden_dim, cfg.dim_a)

    def forward(self, ball_seq, obstacle_img, u_seq=None, mask=None, epoch=100, smoother=False):
        """
        ball_seq:      [B, T, H, W]          — sequence of ball images
        obstacle_img:  [B, H, W]             — static obstacle image
        u_seq:         [B, T, dim_u]         — control inputs
        mask:          [B, T]                — mask for valid timesteps
        """
        B, T, H, W = ball_seq.shape

        a_dist, h_obs = self.encode(ball_seq, obstacle_img)

        if self.training:
            a_seq = a_dist.rsample()  
        else:
            a_seq = a_dist.mean

        if mask is None:
            mask = torch.ones(B, T, device=ball_seq.device)

        if u_seq is None:
            u_seq = torch.zeros(B, T, self.cfg.dim_u, device=ball_seq.device)

        a_list = []
        a_pred_list = []
        h_state = None
        a_pred_k = torch.zeros(B, self.cfg.dim_a, device=ball_seq.device)

        for k in range(T):
            mask_k = mask[:, k].unsqueeze(-1)   # [B, 1]
            u_k    = u_seq[:, k, :]             # [B, dim_u]

            a_k = mask_k * a_seq[:, k, :] + (1 - mask_k) * a_pred_k
            a_list.append(a_k)

            gru_input = torch.cat([a_k, h_obs, u_k], dim=-1).unsqueeze(1)  # [B, 1, dim_a+dim_obs+dim_u]
            gru_out, h_state = self.gru(gru_input, h_state)                # [B, 1, hidden]

            a_pred_k = self.fc_pred(gru_out.squeeze(1))                    # [B, dim_a]
            a_pred_list.append(a_pred_k)                             

        a_s = torch.stack(a_list, dim=1)          # [B, T, dim_a]
        a_pred = torch.stack(a_pred_list, dim=1)  # [B, T, dim_a]

        x_dist = self.decode(a_s)
        
        return (x_dist, 
            a_dist, a_seq, a_s, a_pred,
            None, None, None, 
            None, None, None)

        