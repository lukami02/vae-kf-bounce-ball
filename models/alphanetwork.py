import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append("..")
from config.vae_config import VAEConfig

class AlphaNetwork(nn.Module):
    """
    Computes mixing weights alpha_k for matrix selection.
    """
    def __init__(self, cfg: VAEConfig, obstacle=True):
        super().__init__()
        self.cfg = cfg
        self.obstacle = obstacle

        # Calculate total input dimension
        input_dim = cfg.dim_a
        if obstacle: input_dim += cfg.gru_hidden_dim
        if cfg.dim_u > 0: input_dim += cfg.dim_u

        # Network layers
        self.var_proj = nn.Linear(input_dim, cfg.gru_hidden_dim)
        self.fc_alpha = nn.Linear(cfg.gru_hidden_dim, cfg.num_matrices) 
        self.gru = nn.GRUCell(cfg.gru_hidden_dim, cfg.gru_hidden_dim)

    def forward(self, a_k, h_obs_features, state=None, u_k=None):
        """
        a_k:            [B, dim_a]          — Current observation
        h_obs_features: [B, gru_hidden_dim] — Encoded obstacle
        state:          [B, gru_hidden_dim] — Previous GRU hidden state
        u_k:            [B, dim_u]          — Control input (optional)

        alpha:          [B, num_matrices]   — Mixing weights
        state:          [B, gru_hidden_dim] — Updated GRU hidden state
        """

        if self.cfg.num_matrices == 1:
            return torch.ones(a_k.shape[0], 1, device=a_k.device), state

        # Prepare input features
        x = a_k
        if self.obstacle:
            x = torch.cat([x, h_obs_features], dim=-1)  # [B, dim_a + hidden_dim]
        if self.cfg.dim_u:
            x = torch.cat([x, u_k], dim=-1)             # [B, dim_a + hidden_dim + dim_u]

        # Map to hidden dimension and update recurrence
        var_input = F.relu(self.var_proj(x))            # [B, hidden_dim]
        state = self.gru(var_input, state)              # [B, gru_hidden_dim]

        # Output alpha weights via softmax
        logits = self.fc_alpha(state)                   # [B, num_matrices]
        alpha = torch.softmax(logits, dim=-1)           # [B, num_matrices]
            
        return alpha, state
    
    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.cfg.gru_hidden_dim, device=device)