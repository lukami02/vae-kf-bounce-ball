import torch
import torch.nn as nn
import sys
sys.path.append("..")
from config.vae_config import VAEConfig

class AlphaNetwork(nn.Module):
    """
    Computes mixing weights alpha_k for matrix selection.
    """
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.dim_u == 0 or cfg.dim_u == cfg.dim_a, "dim_u must be 0 or equal to dim_a for consistent input"

        self.var_proj = nn.Sequential(
            nn.Linear(cfg.dim_a + cfg.dim_z, cfg.alpha_units),
            nn.Tanh()
        )

        self.gru = nn.GRUCell(cfg.alpha_units + cfg.dim_obstacle + cfg.dim_u, cfg.alpha_units)

        self.fc_alpha = nn.Sequential(
            nn.Linear(cfg.alpha_units, cfg.alpha_units // 2),
            nn.Tanh(),
            nn.Linear(cfg.alpha_units // 2, cfg.num_matrices)
        )

        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.5)))  

    def forward(self, a_k, h_obs, z_filt, state=None, u_k=None):
        if self.cfg.num_matrices == 1:
            B = a_k.shape[0]
            return torch.ones(B, 1, device=a_k.device), state

        var_input = torch.cat([a_k, z_filt], dim=-1)               # [B, dim_a + dim_z]
        var_input = self.var_proj(var_input)                       # [B, alpha_units]

        inputs = torch.cat([var_input, h_obs], dim=-1)             # [B, alpha_units + dim_obstacle]
        if self.cfg.dim_u > 0 and u_k is not None:
            inputs = torch.cat([inputs, u_k], dim=-1)

        inputs = nn.functional.dropout(inputs, p=0.1, training=self.training)

        state = self.gru(inputs, state)                             # [B, alpha_units]
        logits = self.fc_alpha(state)

        temperature = torch.clamp(self.log_temperature.exp(), 0.1, 2.0)
        
        if self.training:
            alpha = nn.functional.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            alpha = nn.functional.softmax(logits / temperature, dim=-1)
        return alpha, state

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.cfg.alpha_units, device=device)