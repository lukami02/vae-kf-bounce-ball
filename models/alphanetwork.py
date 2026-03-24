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
        self.input_dim = cfg.dim_a + cfg.dim_obstacle + cfg.dim_u + cfg.dim_z

        self.gru = nn.GRUCell(self.input_dim, cfg.alpha_units)
        self.fc_alpha = nn.Sequential(
            nn.Linear(cfg.alpha_units, cfg.alpha_units // 2),
            nn.Tanh(),
            nn.Linear(cfg.alpha_units // 2, cfg.num_matrices)
        )

        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(self, a_k, h_obs, z_filt, state=None, u_k=None):
        if self.cfg.num_matrices == 1:
            B = a_k.shape[0]
            return torch.ones(B, 1, device=a_k.device), state

        parts = [a_k, h_obs, z_filt]
        if self.cfg.dim_u > 0 and u_k is not None:
            parts.append(u_k)
        inputs = torch.cat(parts, dim=-1)                      # [B, input_dim]
        inputs = nn.functional.dropout(inputs, p=0.1, training=self.training)
        state  = self.gru(inputs, state)                       # [B, alpha_units]
        logits = self.fc_alpha(state) 

        temperature = torch.clamp(self.log_temperature.exp(), 0.1, 2.0)
        
        if self.training:
            alpha = nn.functional.gumbel_softmax(logits, tau=temperature, hard=False)
        else:
            alpha = nn.functional.softmax(logits / temperature, dim=-1)
        return alpha, state

    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.cfg.alpha_units, device=device)