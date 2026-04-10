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
        
        input_dim = cfg.dim_a + (cfg.alpha_units if obstacle else 0)
        if cfg.dim_u > 0:
            input_dim += cfg.dim_u

        self.fc1 = nn.Linear(input_dim, cfg.alpha_units) 
        self.fc2 = nn.Linear(cfg.alpha_units, cfg.num_matrices) 
        
        self.gru = nn.GRUCell(cfg.alpha_units, cfg.alpha_units)

    def forward(self, a_k, h_obs_features, state=None, u_k=None, temp=1):
        # h_obs_features [B, alpha_units] 

        if self.cfg.num_matrices == 1:
            return torch.ones(a_k.shape[0], 1, device=a_k.device), state

        if self.obstacle:
            x = torch.cat([a_k, h_obs_features], dim=-1)  # [B, dim_a + alpha_units]
        else:
            x = a_k

        if u_k is not None:
            x = torch.cat([x, u_k], dim=-1)  # [B, dim_a + alpha_units + dim_u]

        x = F.relu(self.fc1(x))

        gru_out = self.gru(x, state) 

        logits = self.fc2(gru_out)
        alpha = F.softmax(logits, dim=-1)

        return alpha, gru_out
    
    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.cfg.alpha_units, device=device)