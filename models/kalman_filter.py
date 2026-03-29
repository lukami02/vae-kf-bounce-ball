import torch
import torch.nn as nn
import torch.distributions as D
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
        self.z_0 = nn.Parameter(torch.zeros(cfg.dim_z))                   # [dim_z]
        self.P_0 = nn.Parameter(torch.eye(cfg.dim_z) * 5)     # [dim_z, dim_z]

        # Noise covariances
        mat_Q = cfg.Q_std * torch.eye(cfg.dim_z)
        self._mat_Q = nn.Parameter(torch.linalg.cholesky((mat_Q + mat_Q.T) / 2))
        mat_R = cfg.R_std * torch.eye(cfg.dim_a)
        self._mat_R = nn.Parameter(torch.linalg.cholesky((mat_R + mat_R.T) / 2))

    @property
    def Q(self):
        device = self._mat_Q.device
        return self._mat_Q @ self._mat_Q.T + torch.eye(self.cfg.dim_z, device=device) * self.cfg.QR_reg   # [dim_z, dim_z] 

    @property
    def R(self):
        device = self._mat_R.device
        return self._mat_R @ self._mat_R.T + torch.eye(self.cfg.dim_a, device=device) * self.cfg.QR_reg    # [dim_a, dim_a]         

    def forward(self, a_seq, alpha_net, h_obs, A_matrices, C_matrices, B_matrices=None, u_seq=None, mask=None, epoch=100):
        """
        a_seq:       [B, T, dim_a]      — encoder sequence
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

        B, T, dim_a = a_seq.shape
        dim_z   = self.cfg.dim_z
        device  = a_seq.device

        # init
        z_init = torch.zeros(B, self.cfg.dim_z, device=device)                     # [B, dim_z]

        P = self.P_0.unsqueeze(0).expand(B, -1, -1).clone()                   # [B, dim_z, dim_z]
        I = torch.eye(dim_z, device=device).unsqueeze(0)                      # [1, dim_z, dim_z]
        Q = self.Q.to(device)                                                 # [dim_z, dim_z]

        if mask is None:
            mask = torch.ones(B, T, device=device)

        # init alpha
        gru_state = alpha_net.init_state(B, device)

        z_filt_list = []
        P_filt_list = []
        z_pred_list = []
        P_pred_list = []
        a_filt_list = []
        a_pred_list = []
        alpha_list  = []
        alpha_imm_list = []
        S_list      = []
        z_dist_list = []

        z_prev = z_init
        C_k_prev = C_matrices[0].unsqueeze(0).expand(B, -1, -1).clone()       # [B, dim_a, dim_z]

        for k in range(T):
            a_k    = a_seq[:, k, :]                                           # [B, dim_a]
            mask_k = mask[:, k].unsqueeze(-1)                                 # [B, 1]
            u_k    = u_seq[:, k, :] if u_seq is not None else None

            log_likelihoods = []
            a_next_per_expert  = []

            for j in range(self.cfg.num_matrices):
                A_j = A_matrices[j].unsqueeze(0).expand(B, -1, -1)
                C_j = C_matrices[j].unsqueeze(0).expand(B, -1, -1)

                z_pred_j = torch.bmm(A_j, z_prev.unsqueeze(-1)).squeeze(-1)  # [B, dim_z]
                a_pred_j = torch.bmm(C_j, z_pred_j.unsqueeze(-1)).squeeze(-1) # [B, dim_a]
                a_next_per_expert.append(a_pred_j)

                r_j = a_k - a_pred_j
                log_likelihoods.append(-r_j.pow(2).sum(dim=-1))  # [B]

            a_next_all = torch.cat(a_next_per_expert, dim=-1)  # [B, K * dim_a]

            log_likelihoods = torch.stack(log_likelihoods, dim=-1)  # [B, K]
            alpha_imm = torch.softmax(log_likelihoods, dim=-1)  # [B, K]

            has_obs   = mask_k.squeeze(-1)  # [B]
            alpha_imm = alpha_imm * has_obs.unsqueeze(-1).float()
            
            a_prev = torch.bmm(C_k_prev, z_prev.unsqueeze(-1)).squeeze(-1).detach()

            alpha_k, gru_state = alpha_net(
                a_prev, h_obs, z_prev.detach(), gru_state,
                temp=self.cfg.get_temperature(epoch),
                a_next_all=a_next_all
            )

            alpha_list.append(alpha_k)
            alpha_imm_list.append(alpha_imm)

            # Compose matrices

            w = alpha_k.unsqueeze(-1).unsqueeze(-1)                       # [B, K, 1, 1]

            A_k = (w * A_matrices.unsqueeze(0)).sum(dim=1)                    # [B, dim_z, dim_z]
            C_k = (w * C_matrices.unsqueeze(0)).sum(dim=1)                    # [B, dim_a, dim_z]

            z = torch.bmm(A_k, z_prev.unsqueeze(-1)).squeeze(-1)              # [B, dim_z]

            # Update 
            a_k_hat = torch.bmm(C_k, z.unsqueeze(-1)).squeeze(-1)             # [B, dim_a]
            r_k = a_k - a_k_hat                                               # [B, dim_a] 

            S_k = torch.bmm(C_k, torch.bmm(P, C_k.transpose(1, 2))) + self.R.unsqueeze(0).expand(B, -1, -1)
            S_k = S_k + 1e-4 * torch.eye(dim_a, device=device).unsqueeze(0) 
            K_k = torch.linalg.solve(
                      S_k.transpose(1, 2),
                      torch.bmm(C_k, P.transpose(1, 2))
                  ).transpose(1, 2)                                           # [B, dim_z, dim_a]

            # Missing values
            K_k = K_k * mask_k.unsqueeze(-1)                                  # [B, dim_z, dim_a]

            # Joseph form 
            IKC    = I - torch.bmm(K_k, C_k)                                  # [B, dim_z, dim_z]
            P_filt = torch.bmm(IKC, torch.bmm(P, IKC.transpose(1, 2))) \
                   + torch.bmm(K_k, torch.bmm(self.R.unsqueeze(0).expand(B, -1, -1), K_k.transpose(1, 2))) 
            P_filt = (P_filt + P_filt.transpose(1, 2)) / 2.0
            P_filt = P_filt + 1e-3 * torch.eye(dim_z, device=device).unsqueeze(0)

            z_dist = z + torch.bmm(K_k, r_k.unsqueeze(-1)).squeeze(-1)        # [B, dim_z]
            z_dist = D.MultivariateNormal(loc=z_dist, scale_tril=torch.linalg.cholesky(P_filt))  # [B, dim_z]

            if self.training:
                z_filt = z_dist.rsample()
            else:
                z_filt = z_dist.mean

            p_true = max(0.0, 1.0 - epoch / 20)
            use_true = (torch.rand(B, device=device) < p_true).unsqueeze(-1)  # [B, 1]
            a_filt_kalman = torch.bmm(C_k, z_filt.unsqueeze(-1)).squeeze(-1)
            a_filt_true   = mask_k * a_k + (1 - mask_k) * a_filt_kalman
            a_filt_k = torch.where(use_true, a_filt_true, a_filt_kalman)

            # State prediction
            if self.training:
                z_pred = D.MultivariateNormal(
                    torch.bmm(A_k, z_filt.unsqueeze(-1)).squeeze(-1), Q
                ).rsample()
            else:
                z_pred = torch.bmm(A_k, z_filt.unsqueeze(-1)).squeeze(-1)

            if self.cfg.dim_u > 0 and B_matrices is not None and u_k is not None:
                B_k    = (w * B_matrices.unsqueeze(0)).sum(dim=1)
                z_pred = z_pred + torch.bmm(B_k, u_k.unsqueeze(-1)).squeeze(-1)

            P_pred = torch.bmm(A_k, torch.bmm(P_filt, A_k.transpose(1, 2))) + Q
            P_pred = (P_pred + P_pred.transpose(1, 2)) / 2.0
            P_pred = P_pred + 1e-3 * torch.eye(dim_z, device=device).unsqueeze(0)  

            # Observation prediction
            a_pred_k = torch.bmm(C_k, z_pred.unsqueeze(-1)).squeeze(-1)       # [B, dim_a]

            if k < self.cfg.burn_in and epoch < 5:
                z_filt = z_filt.detach()
                P_filt = P_filt.detach()
                z_pred = z_pred.detach()
                P_pred = P_pred.detach()

            z_filt_list.append(z_filt)
            P_filt_list.append(P_filt)
            z_pred_list.append(z_pred)
            P_pred_list.append(P_pred)
            a_filt_list.append(a_filt_k)
            a_pred_list.append(a_pred_k)
            S_list.append(S_k)
            z_dist_list.append(z_dist)

            C_k_prev = C_k.detach()
            z_prev   = z_filt
            P        = P_pred

        z_filt    = torch.stack(z_filt_list, dim=1)   # [B, T, dim_z]
        P_filt    = torch.stack(P_filt_list, dim=1)   # [B, T, dim_z, dim_z]
        z_pred    = torch.stack(z_pred_list, dim=1)   # [B, T, dim_z]
        P_pred    = torch.stack(P_pred_list, dim=1)   # [B, T, dim_z, dim_z]
        a_filt    = torch.stack(a_filt_list, dim=1)   # [B, T, dim_a]
        a_pred    = torch.stack(a_pred_list, dim=1)   # [B, T, dim_a]
        alpha_seq = torch.stack(alpha_list,  dim=1)   # [B, T, K]
        alpha_imm = torch.stack(alpha_imm_list, dim=1)# [B, T, K]
        S_pred    = torch.stack(S_list,  dim=1)       # [B, T, dim_a, dim_a]
        z_dist_locs   = torch.stack([d.loc        for d in z_dist_list], dim=1)   # [B, T, dim_z]
        z_dist_trils  = torch.stack([d.scale_tril for d in z_dist_list], dim=1)   # [B, T, dim_z, dim_z]
        z_dist        = D.MultivariateNormal(loc=z_dist_locs, scale_tril=z_dist_trils)

        return z_filt, P_filt, z_dist, z_pred, P_pred, a_filt, a_pred, S_pred, alpha_seq, alpha_imm, self.R, self.Q