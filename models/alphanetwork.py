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
        self.input_dim = cfg.dim_a + cfg.dim_obstacle + cfg.dim_u 

        self.gru = nn.GRUCell(self.input_dim, cfg.alpha_units)
        self.fc_alpha = nn.Linear(cfg.alpha_units, cfg.num_matrices)

    def forward(self, a_k, h_obs, state=None, u_k=None):
        if self.cfg.num_matrices == 1:
            B = a_k.shape[0]
            return torch.ones(B, 1, device=a_k.device), state

        parts = [a_k, h_obs]
        if self.cfg.dim_u > 0 and u_k is not None:
            parts.append(u_k)
        inputs = torch.cat(parts, dim=-1)                      # [B, input_dim]

        state  = self.gru(inputs, state)                       # [B, alpha_units]
        output = state

        alpha = torch.softmax(self.fc_alpha(output), dim=-1)   # [B, num_matrices]
        return alpha, state

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.cfg.alpha_units, device=device)