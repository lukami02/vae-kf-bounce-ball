import torch
import torch.nn as nn
import math
import sys
sys.path.append("..")
from config.vae_config import VAEConfig


class KalmanFilter(nn.Module):
    """
    Linear Kalman Filter operating in latent space.

    State space model:
        z_k = A_k @ z_{k-1} + B_k @ u_k + w_k,   w_k ~ N(0, Q)
        a_k = C_k @ z_k + v_k,                   v_k ~ N(0, R)
    """
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        # init state
        self.z_0     = nn.Parameter(torch.zeros(cfg.dim_z))                   # [dim_z]
        self.P_0_log_diag = nn.Parameter(torch.zeros(cfg.dim_z))              # [dim_z, dim_z]

        # init output
        self.a_0 = nn.Parameter(torch.zeros(cfg.dim_a))                       # [dim_a]

        # Noise covariances
        self.log_Q_diag = nn.Parameter(torch.full((cfg.dim_z,), math.log(cfg.Q_std)))

    @property
    def Q(self):
        return torch.diag(torch.exp(self.log_Q_diag))                         # [dim_z, dim_z]

    @property
    def P_0(self):
        return torch.diag(torch.exp(self.P_0_log_diag))

    def forward(self, a_seq, a_var, alpha_net, h_obs, A_matrices, C_matrices, B_matrices=None, u_seq=None, mask=None):
        """
        a_seq:       [B, T, dim_a]      — encoder means
        a_var:       [B, T, dim_a]      — encoder variances -> R_k
        alpha_net:   AlphaNetwork       
        h_obs:       [B, dim_obstacle]  — obstacle context
        A_matrices:  [K, dim_z, dim_z]  — A matrices stack
        C_matrices:  [K, dim_a, dim_z]  — C matrices stack
        B_matrices:  [K, dim_z, dim_u]  — B matrices stack
        u_seq:       [B, T, dim_u]      — control inputs
        param mask:  [B, T]

        z_filt:      [B, T, dim_z]
        P_filt:      [B, T, dim_z, dim_z]
        z_pred:      [B, T, dim_z]
        P_pred:      [B, T, dim_z, dim_z]
        a_filt:      [B, T, dim_a]   
        a_pred:      [B, T, dim_a] 
        alpha_seq:   [B, T, K]
        """

        B, T, _ = a_seq.shape
        dim_z   = self.cfg.dim_z
        device  = a_seq.device

        # init
        z = self.z_0.unsqueeze(0).expand(B, -1).clone()                       # [B, dim_z]
        P = self.P_0.unsqueeze(0).expand(B, -1, -1).clone()                   # [B, dim_z, dim_z]
        I = torch.eye(dim_z, device=device).unsqueeze(0)                      # [1, dim_z, dim_z]
        Q = self.Q.to(device)                                                 # [dim_z, dim_z]

        if mask is None:
            mask = torch.ones(B, T, device=device)

        # init alpha
        a_prev    = self.a_0.unsqueeze(0).expand(B, -1)                       # [B, dim_a]
        gru_state = alpha_net.init_state(B, device)
        alpha_k, gru_state = alpha_net(a_prev, h_obs, gru_state)

        z_filt_list = []
        P_filt_list = []
        z_pred_list = []
        P_pred_list = []
        a_filt_list = []
        a_pred_list = []
        alpha_list  = []

        for k in range(T):
            a_k    = a_seq[:, k, :]                                           # [B, dim_a]
            R_k    = torch.diag_embed(a_var[:, k, :])                         # [B, dim_a, dim_a]
            mask_k = mask[:, k].unsqueeze(-1)                                 # [B, 1]
            u_k    = u_seq[:, k, :] if u_seq is not None else None

            # Compose matrices
            w   = alpha_k.unsqueeze(-1).unsqueeze(-1)                         # [B, K, 1, 1]
            A_k = (w * A_matrices.unsqueeze(0)).sum(dim=1)                    # [B, dim_z, dim_z]
            C_k = (w * C_matrices.unsqueeze(0)).sum(dim=1)                    # [B, dim_a, dim_z]

            # Update 
            a_k_hat = torch.bmm(C_k, z.unsqueeze(-1)).squeeze(-1)             # [B, dim_a]
            r_k     = a_k - a_k_hat                                           # [B, dim_a] 

            S_k = torch.bmm(C_k, torch.bmm(P, C_k.transpose(1, 2))) + R_k     # [B, dim_a, dim_a]
            K_k = torch.linalg.solve(
                      S_k.transpose(1, 2),
                      torch.bmm(C_k, P.transpose(1, 2))
                  ).transpose(1, 2)                                           # [B, dim_z, dim_a]

            # Missing values
            K_k = K_k * mask_k.unsqueeze(-1)                                  # [B, dim_z, dim_a]

            z_filt = z + torch.bmm(K_k, r_k.unsqueeze(-1)).squeeze(-1)        # [B, dim_z]

            # Joseph form 
            IKC    = I - torch.bmm(K_k, C_k)                                  # [B, dim_z, dim_z]
            P_filt = torch.bmm(IKC, torch.bmm(P, IKC.transpose(1, 2))) \
                   + torch.bmm(K_k, torch.bmm(R_k, K_k.transpose(1, 2)))      # [B, dim_z, dim_z]

            # Filtered output
            a_filt_k = torch.bmm(C_k, z_filt.unsqueeze(-1)).squeeze(-1)       # [B, dim_a]

            # Update alpha
            a_for_alpha = mask_k * a_k + (1 - mask_k) * a_k_hat               # [B, dim_a]
            alpha_k, gru_state = alpha_net(a_for_alpha, h_obs, gru_state)
            alpha_list.append(alpha_k)

            w   = alpha_k.unsqueeze(-1).unsqueeze(-1)
            A_k = (w * A_matrices.unsqueeze(0)).sum(dim=1)                    # [B, dim_z, dim_z]
            C_k = (w * C_matrices.unsqueeze(0)).sum(dim=1)                    # [B, dim_a, dim_z]

            # State prediction
            z_pred = torch.bmm(A_k, z_filt.unsqueeze(-1)).squeeze(-1)            # [B, dim_z]
            if self.cfg.dim_u > 0 and B_matrices is not None and u_k is not None:
                B_k    = (w * B_matrices.unsqueeze(0)).sum(dim=1)                # [B, dim_z, dim_u]
                z_pred = z_pred + torch.bmm(B_k, u_k.unsqueeze(-1)).squeeze(-1)

            P_pred = torch.bmm(A_k, torch.bmm(P_filt, A_k.transpose(1, 2))) + Q  # [B, dim_z, dim_z]

            # Observation prediction
            a_pred_k = torch.bmm(C_k, z_pred.unsqueeze(-1)).squeeze(-1)       # [B, dim_a]

            z_filt_list.append(z_filt)
            P_filt_list.append(P_filt)
            z_pred_list.append(z_pred)
            P_pred_list.append(P_pred)
            a_filt_list.append(a_filt_k)
            a_pred_list.append(a_pred_k)

            z = z_pred
            P = P_pred

        z_filt    = torch.stack(z_filt_list, dim=1)   # [B, T, dim_z]
        P_filt    = torch.stack(P_filt_list, dim=1)   # [B, T, dim_z, dim_z]
        z_pred    = torch.stack(z_pred_list, dim=1)   # [B, T, dim_z]
        P_pred    = torch.stack(P_pred_list, dim=1)   # [B, T, dim_z, dim_z]
        a_filt    = torch.stack(a_filt_list, dim=1)   # [B, T, dim_a]
        a_pred    = torch.stack(a_pred_list, dim=1)   # [B, T, dim_a]
        alpha_seq = torch.stack(alpha_list,  dim=1)   # [B, T, K]

        return z_filt, P_filt, z_pred, P_pred, a_filt, a_pred, alpha_seq