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
        self.register_buffer('P_0', cfg.Q_std * torch.eye(cfg.dim_z))     # [dim_z, dim_z]

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
        z = torch.zeros(B, self.cfg.dim_z, device=device)                     # [B, dim_z]
        z[:, -dim_a:] = a_seq[:, 1, :] - a_seq[:, 0, :]
        z[:, :dim_a] = a_seq[:, 0, :] - (a_seq[:, 1, :] - a_seq[:, 0, :])
        z = z.detach()
        P = self.P_0.unsqueeze(0).expand(B, -1, -1).clone()                   # [B, dim_z, dim_z]
        I = torch.eye(dim_z, device=device).unsqueeze(0)                      # [1, dim_z, dim_z]
        Q = self.Q.to(device)                                                 # [dim_z, dim_z]

        if mask is None:
            mask = torch.ones(B, T, device=device)

        # init alpha
        a_prev = (a_seq[:, 0, :] - (a_seq[:, 1, :] - a_seq[:, 0, :])).detach()   # [B, dim_a]
        gru_state = alpha_net.init_state(B, device)
        alpha_k, gru_state = alpha_net(a_prev, h_obs, z.detach(), gru_state, temp=self.cfg.get_temperature(epoch))

        z_filt_list = []
        P_filt_list = []
        z_pred_list = []
        P_pred_list = []
        a_filt_list = []
        a_pred_list = []
        alpha_list  = []
        alpha_imm_list = []
        S_list      = []

        for k in range(T):
            a_k    = a_seq[:, k, :]                                           # [B, dim_a]
            mask_k = mask[:, k].unsqueeze(-1)                                 # [B, 1]
            u_k    = u_seq[:, k, :] if u_seq is not None else None

            log_likelihoods = []
            for j in range(self.cfg.num_matrices):
                C_j = C_matrices[j].unsqueeze(0).expand(B, -1, -1)
                a_pred_j = torch.bmm(C_j, z.unsqueeze(-1)).squeeze(-1)
                r_j = a_k - a_pred_j
                S_j = torch.bmm(C_j, torch.bmm(P, C_j.transpose(1, 2))) \
                    + self.R.unsqueeze(0).expand(B, -1, -1)
                S_j = 0.5 * (S_j + S_j.transpose(-1, -2)) \
                    + 1e-4 * torch.eye(S_j.shape[-1], device=device).unsqueeze(0)
                dist_j = D.MultivariateNormal(
                    torch.zeros_like(r_j),
                    scale_tril=torch.linalg.cholesky(S_j)
                )
                log_likelihoods.append(dist_j.log_prob(r_j))

            log_likelihoods = torch.stack(log_likelihoods, dim=-1)   # [B, K]
            logits_imm = log_likelihoods - log_likelihoods.logsumexp(dim=-1, keepdim=True)
            alpha_imm = torch.softmax(logits_imm / self.cfg.get_temperature(epoch), dim=-1)
            has_obs = mask_k.squeeze(-1)  # [B]
            alpha_imm = alpha_imm * has_obs.unsqueeze(-1).float()
            alpha_imm_list.append(alpha_imm)

            # Compose matrices
            w   = alpha_k.unsqueeze(-1).unsqueeze(-1)                         # [B, K, 1, 1]
            A_k = (w * A_matrices.unsqueeze(0)).sum(dim=1)                    # [B, dim_z, dim_z]
            C_k = (w * C_matrices.unsqueeze(0)).sum(dim=1)                    # [B, dim_a, dim_z]

            # Update 
            a_k_hat = torch.bmm(C_k, z.unsqueeze(-1)).squeeze(-1)             # [B, dim_a]
            r_k = a_k - a_k_hat                                               # [B, dim_a] 

            S_k = torch.bmm(C_k, torch.bmm(P, C_k.transpose(1, 2))) + self.R.unsqueeze(0).expand(B, -1, -1) 
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
                   + torch.bmm(K_k, torch.bmm(self.R.unsqueeze(0).expand(B, -1, -1), K_k.transpose(1, 2))) 
            P_filt = (P_filt + P_filt.transpose(1, 2)) / 2.0

            # Filtered output
            """
            if self.training:
                a_filt_k = D.MultivariateNormal(
                    torch.bmm(C_k, z_filt.unsqueeze(-1)).squeeze(-1), self.R
                ).rsample()
            else:
                a_filt_k = torch.bmm(C_k, z_filt.unsqueeze(-1)).squeeze(-1)
            """

            p_true = max(0.0, 1.0 - epoch / 20)

            use_true = (torch.rand(B, device=device) < p_true).unsqueeze(-1)  # [B, 1]

            a_filt_kalman = torch.bmm(C_k, z_filt.unsqueeze(-1)).squeeze(-1)
            a_filt_true   = mask_k * a_k + (1 - mask_k) * a_filt_kalman

            a_filt_k = torch.where(use_true, a_filt_true, a_filt_kalman)

            # Update alpha
            a_for_alpha = a_filt_k                                            # [B, dim_a]
            alpha_k, gru_state = alpha_net(a_for_alpha, h_obs, z_filt, gru_state, temp=self.cfg.get_temperature(epoch))
            alpha_list.append(alpha_k)

            w   = alpha_k.unsqueeze(-1).unsqueeze(-1)
            A_k = (w * A_matrices.unsqueeze(0)).sum(dim=1)                    # [B, dim_z, dim_z]
            C_k = (w * C_matrices.unsqueeze(0)).sum(dim=1)                    # [B, dim_a, dim_z]

            # State prediction
            if self.training:
                z_pred = D.MultivariateNormal(torch.bmm(A_k, z_filt.unsqueeze(-1)).squeeze(-1), Q).rsample()
            else:
                z_pred = torch.bmm(A_k, z_filt.unsqueeze(-1)).squeeze(-1)
                
            if self.cfg.dim_u > 0 and B_matrices is not None and u_k is not None:
                B_k    = (w * B_matrices.unsqueeze(0)).sum(dim=1)                # [B, dim_z, dim_u]
                z_pred = z_pred + torch.bmm(B_k, u_k.unsqueeze(-1)).squeeze(-1)

            P_pred = torch.bmm(A_k, torch.bmm(P_filt, A_k.transpose(1, 2))) + Q  # [B, dim_z, dim_z]
            P_pred = (P_pred + P_pred.transpose(1, 2)) / 2.0

            #S_pred = torch.bmm(C_k, torch.bmm(P_pred, C_k.transpose(1, 2))) + self.R.unsqueeze(0).expand(B, -1, -1)

            # Observation prediction
            a_pred_k = torch.bmm(C_k, z_pred.unsqueeze(-1)).squeeze(-1)       # [B, dim_a]

            if k < self.cfg.burn_in and epoch < 5:
                z_filt = z_filt.detach()
                P_filt = P_filt.detach()
                z_pred = z_pred.detach()
                P_pred = P_pred.detach()
                S_pred = S_k.detach()

            z_filt_list.append(z_filt)
            P_filt_list.append(P_filt)
            z_pred_list.append(z_pred)
            P_pred_list.append(P_pred)
            a_filt_list.append(a_filt_k)
            a_pred_list.append(a_pred_k)
            S_list.append(S_k)

            z = z_pred
            P = P_pred

        z_filt    = torch.stack(z_filt_list, dim=1)   # [B, T, dim_z]
        P_filt    = torch.stack(P_filt_list, dim=1)   # [B, T, dim_z, dim_z]
        z_pred    = torch.stack(z_pred_list, dim=1)   # [B, T, dim_z]
        P_pred    = torch.stack(P_pred_list, dim=1)   # [B, T, dim_z, dim_z]
        a_filt    = torch.stack(a_filt_list, dim=1)   # [B, T, dim_a]
        a_pred    = torch.stack(a_pred_list, dim=1)   # [B, T, dim_a]
        alpha_seq = torch.stack(alpha_list,  dim=1)   # [B, T, K]
        alpha_imm = torch.stack(alpha_imm_list, dim=1)# [B, T, K]
        S_pred    = torch.stack(S_list,  dim=1)       # [B, T, dim_a, dim_a]

        return z_filt, P_filt, z_pred, P_pred, a_filt, a_pred, S_pred, alpha_seq, alpha_imm, self.R, self.Q