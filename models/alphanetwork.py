import torch
import torch.nn as nn
import math
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
        
        self.z_proj = nn.Linear(cfg.dim_z, cfg.alpha_units)
        
        # Attention layers
        self.query_layer = nn.Linear(cfg.alpha_units, cfg.alpha_units)
        self.key_layer = nn.Linear(cfg.alpha_units, cfg.alpha_units)
        self.value_layer = nn.Linear(cfg.alpha_units, cfg.alpha_units)

        self.var_proj = nn.Sequential(
            nn.Linear(cfg.dim_a + cfg.dim_z, cfg.alpha_units),
            nn.Tanh()
        )

        self.gru = nn.GRUCell(cfg.alpha_units * 2 + (cfg.dim_u if cfg.dim_u > 0 else 0), cfg.alpha_units)

        self.fc_alpha = nn.Sequential(
            nn.Linear(cfg.alpha_units, cfg.alpha_units // 2),
            nn.Tanh(),
            nn.Linear(cfg.alpha_units // 2, cfg.num_matrices)
        )

    def forward(self, a_k, h_obs_features, z_filt, state=None, u_k=None, temp=0.1):
        # h_obs_features [B, N, alpha_units] 

        if self.cfg.num_matrices == 1:
            return torch.ones(a_k.shape[0], 1, device=a_k.device), state

        z_q = self.z_proj(z_filt) # [B, alpha_units]
        
        Q = self.query_layer(z_q).unsqueeze(1) # [B, 1, units]
        K = self.key_layer(h_obs_features)     # [B, N, units]
        V = self.value_layer(h_obs_features)   # [B, N, units]
        
        # Scaled dot-product
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.cfg.alpha_units)
        attn_weights = torch.softmax(scores, dim=-1)    # [B, 1, N]
        context = torch.bmm(attn_weights, V).squeeze(1) # [B, alpha_units]

        var_input = self.var_proj(torch.cat([a_k, z_filt], dim=-1))
        inputs = torch.cat([var_input, context], dim=-1)
        
        if self.cfg.dim_u > 0 and u_k is not None:
            inputs = torch.cat([inputs, u_k], dim=-1)

        state = self.gru(inputs, state)
        logits = self.fc_alpha(state)

        alpha = torch.softmax(logits / temp, dim=-1)
            
        return alpha, state
    
    def init_state(self, batch_size, device):
        return torch.zeros(batch_size, self.cfg.alpha_units, device=device)