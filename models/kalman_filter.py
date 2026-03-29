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

        # init state covariance
        self.P_0 = nn.Parameter(10*torch.eye(cfg.dim_z), requires_grad=False)     # [dim_z, dim_z]
        self.a_0 = nn.Parameter(torch.zeros(1, cfg.dim_a), requires_grad=True)

        # Noise covariances
        values = torch.cat([torch.full((cfg.dim_z // 2,), 0.2), torch.full((cfg.dim_z - cfg.dim_z // 2,), 0.2)])

        self.register_buffer("Q", torch.diag(values))
        self.register_buffer("R", 0.3 * torch.eye(cfg.dim_a))

    def _safe_cholesky(self, M, jitter=1e-4): 
        I = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
        # Dodaj batch dimenziju ako je potrebno
        if M.dim() == 2:
            I_batch = I
        else:
            I_batch = I.unsqueeze(0).expand_as(M)
        
        M = 0.5 * (M + M.transpose(-1, -2)) + jitter * I_batch
        
        for scale in [1.0, 10.0, 100.0, 1000.0]:
            try:
                return torch.linalg.cholesky(M + scale * jitter * I_batch)
            except torch.linalg.LinAlgError:
                continue
        
        # Fallback sa ispravnom batch dimenzijom
        return torch.linalg.cholesky(I_batch)
    
    def _kalman_filter(self, a_seq, alpha_net, h_obs, A_matrices, C_matrices, B_matrices=None, u_seq=None, mask=None, epoch=100):
        """
        Perform Kalman filtering to compute filtered state estimates.

        a_seq:       [B, T, dim_a]      — observation sequence
        alpha_net:   nn.Module          — alpha network
        h_obs:       [B, T, dim_h]      — observation embeddings
        A_matrices:  [K, dim_z, dim_z]  — A matrices stack
        C_matrices:  [K, dim_a, dim_z]  — C matrices stack
        B_matrices:  [K, dim_z, dim_u]  — B matrices stack
        u_seq:       [B, T, dim_u]      — control inputs
        mask:        [B, T]             — masking for valid observations
        epoch:       int                — current training epoch

        z_filt:      [B, T, dim_z]
        P_filt:      [B, T, dim_z, dim_z]
        """
        B, T, dim_a = a_seq.shape
        dim_z = self.cfg.dim_z
        device = a_seq.device

        I = torch.eye(dim_z, device=device).unsqueeze(0)    # [1, dim_z, dim_z]

        if mask is None:
            mask = torch.ones(B, T, device=device)

        z_filt_list, P_filt_list = [], []
        z_pred_list, P_pred_list = [], []
        a_filt_list, a_pred_list = [], []
        A_list, C_list= [], [] 
        alpha_list = []
        z_mean_list = []
        z_scale_tril_list = []
        S_pred_list = []

        z_prev = torch.zeros(B, dim_z, device=device)  # Initial state
        P = self.P_0.unsqueeze(0).expand(B, -1, -1).clone()  # Initial covariance
        C_k_prev = C_matrices[0].unsqueeze(0).expand(B, -1, -1).clone()  # Initial C

        gru_state = alpha_net.init_state(B, device)

        a_prev = self.a_0.expand(B, -1)
        alpha_k, gru_state = alpha_net(
            a_prev, h_obs, gru_state,
            temp=self.cfg.get_temperature(epoch),
        )
        if epoch < 0:
            alpha_k = alpha_k.detach()   
        w = alpha_k.unsqueeze(-1).unsqueeze(-1)                  

        A_k = (w * A_matrices.unsqueeze(0)).sum(dim=1)                # [B, dim_z, dim_z]
        C_k = (w * C_matrices.unsqueeze(0)).sum(dim=1)                # [B, dim_a, dim_z]

        for k in range(T):
            a_k    = a_seq[:, k, :]                                           # [B, dim_a]
            mask_k = mask[:, k].unsqueeze(-1)                                 # [B, 1]
            u_k    = u_seq[:, k, :] if u_seq is not None else None

            # Predict
            z = torch.bmm(A_k, z_prev.unsqueeze(-1)).squeeze(-1)              # [B, dim_z]

            # Update 
            a_k_hat = torch.bmm(C_k, z.unsqueeze(-1)).squeeze(-1)             # [B, dim_a]
            r_k = a_k - a_k_hat                                               # [B, dim_a] 

            S_k = torch.bmm(C_k, torch.bmm(P, C_k.transpose(1, 2))) + self.R.unsqueeze(0).expand(B, -1, -1)
            K_k = torch.linalg.solve(
                      S_k.transpose(1, 2),
                      torch.bmm(C_k, P.transpose(1, 2))
                  ).transpose(1, 2)     # [B, dim_z, dim_a]

            K_k = K_k * mask_k.unsqueeze(-1)                                  # [B, dim_z, dim_a]

            # Joseph form 
            IKC    = I - torch.bmm(K_k, C_k)                                  # [B, dim_z, dim_z]
            P_filt = torch.bmm(IKC, torch.bmm(P, IKC.transpose(1, 2))) \
                   + torch.bmm(K_k, torch.bmm(self.R.unsqueeze(0).expand(B, -1, -1), K_k.transpose(1, 2))) 
            P_filt = (P_filt + P_filt.transpose(1, 2)) / 2.0
            P_filt = P_filt # + 1e-4 * I

            z_filt_mean = z + torch.bmm(K_k, r_k.unsqueeze(-1)).squeeze(-1)    # [B, dim_z]
            L_k = self._safe_cholesky(P_filt)
            z_dist_k    = D.MultivariateNormal(
                loc=z_filt_mean,
                scale_tril=L_k
            )    # [B, dim_z]
            z_mean_list.append(z_filt_mean)
            z_scale_tril_list.append(L_k)

            #z_filt = z_dist_k.rsample() if self.training else z_dist_k.mean
            z_filt = z_dist_k.mean
            a_filt = torch.bmm(C_k, z_filt.unsqueeze(-1)).squeeze(-1)             # [B, dim_a]
            a_k_hat = mask_k * a_k + (1 - mask_k) * a_filt
            
            C_list.append(C_k)

            alpha_k, gru_state = alpha_net(
                a_k_hat, h_obs, gru_state,
                temp=self.cfg.get_temperature(epoch),
            )
            if epoch < 0:
                alpha_k = alpha_k.detach()   
            # Compose matrices
            w = alpha_k.unsqueeze(-1).unsqueeze(-1)                       # [B, K, 1, 1]
            A_k = (w * A_matrices.unsqueeze(0)).sum(dim=1)                # [B, dim_z, dim_z]
            C_k = (w * C_matrices.unsqueeze(0)).sum(dim=1)                # [B, dim_a, dim_z]
            A_list.append(A_k)

            # State prediction
            if False: #self.training:
                z_pred = D.MultivariateNormal(
                    torch.bmm(A_k, z_filt.unsqueeze(-1)).squeeze(-1), self.Q
                ).rsample()
            else:
                z_pred = torch.bmm(A_k, z_filt.unsqueeze(-1)).squeeze(-1)
            
            if self.cfg.dim_u > 0 and B_matrices is not None and u_k is not None:
                B_k = (w * B_matrices.unsqueeze(0)).sum(dim=1)
                z_pred = z_pred + torch.bmm(B_k, u_k.unsqueeze(-1)).squeeze(-1)

            a_pred = torch.bmm(C_k, z_pred.unsqueeze(-1)).squeeze(-1)
            P_pred = torch.bmm(A_k, torch.bmm(P_filt, A_k.transpose(1, 2))) + self.Q
            P_pred = (P_pred + P_pred.transpose(1, 2)) / 2.0# + 1e-4 * I
            
            if k < self.cfg.burn_in and epoch < 5:
                z_filt = z_filt.detach()
                P_filt = P_filt.detach()
                z_pred = z_pred.detach()
                P_pred = P_pred.detach()

            z_filt_list.append(z_filt)
            P_filt_list.append(P_filt)
            z_pred_list.append(z_pred)
            P_pred_list.append(P_pred)
            alpha_list.append(alpha_k)
            a_filt_list.append(a_filt)
            a_pred_list.append(a_pred)
            S_pred_list.append(S_k)

            C_k_prev = C_k
            z_prev   = z_filt
            P        = P_pred

        z_filt    = torch.stack(z_filt_list, dim=1)   # [B, T, dim_z]
        P_filt    = torch.stack(P_filt_list, dim=1)   # [B, T, dim_z, dim_z]
        z_pred    = torch.stack(z_pred_list, dim=1)   # [B, T, dim_z]
        P_pred    = torch.stack(P_pred_list, dim=1)   # [B, T, dim_z, dim_z]
        alpha_seq = torch.stack(alpha_list,  dim=1)   # [B, T, K]
        A_list    = torch.stack(A_list, dim=1)        # [B, T, dim_z, dim_z]
        C_list    = torch.stack(C_list, dim=1)        # [B, T, dim_a, dim_z]
        a_filt    = torch.stack(a_filt_list, dim=1)   # [B, T, dim_a]
        a_pred    = torch.stack(a_pred_list, dim=1)   # [B, T, dim_a]
        S_pred    = torch.stack(S_pred_list, dim=1)  # [B, T, dim_a, dim_a]

        z_means = torch.stack(z_mean_list, dim=1)             # [B, T, dim_z]
        z_scale_tril = torch.stack(z_scale_tril_list, dim=1)   # [B, T, dim_z, dim_z]

        z_dist = D.MultivariateNormal(loc=z_means, scale_tril=z_scale_tril)

        return z_filt, P_filt, z_pred, a_filt, a_pred, P_pred, alpha_seq, A_list, C_list, z_dist, S_pred

    def _rts_smoother(self, z_filt_list, P_filt_list, z_pred_list, P_pred_list, A_list, C_list):
        """
        Rauch-Tung-Striebel smoother for computing smoothed state estimates.

        z_filt_list:  [B, T, dim_z]
        P_filt_list:  [B, T, dim_z, dim_z]
        z_pred_list:  [B, T, dim_z]
        P_pred_list:  [B, T, dim_z, dim_z]
        A_list:       [B, T, dim_z, dim_z]

        z_smooth: [B, T, dim_z]
        P_smooth: [B, T, dim_z, dim_z]
        """

        B, T, _ = z_filt_list.shape
        device = z_filt_list[0].device
        dim_z = self.cfg.dim_z

        z_smooth = [None] * T
        P_smooth = [None] * T

        # Initialize with last filtered estimate
        z_smooth[-1] = z_filt_list[:, -1, :]  # [B, dim_z]
        P_smooth[-1] = P_filt_list[:, -1, :, :]

        for t in reversed(range(T - 1)):
            P_f = P_filt_list[:, t, :, :]        # [B, dim_z, dim_z]
            P_p = P_pred_list[:, t, :, :]        # [B, dim_z, dim_z]
            A_k = A_list[:, t, :, :]            # [B, dim_z, dim_z]
            z_f = z_filt_list[:, t, :]        # [B, dim_z]
            z_p = z_pred_list[:, t, :]        # [B, dim_z] 

            # Smoother gain: J_k = P_filt_k @ A_k^T @ P_pred_{k+1}^{-1}
            J_k = torch.linalg.solve(
                P_p.transpose(1, 2),
                torch.bmm(A_k, P_f.transpose(1, 2))
            ).transpose(1, 2)           # [B, dim_z, dim_z]

            # Smoothed mean
            delta_z = (z_smooth[t+1] - z_p).unsqueeze(-1)            # [B, dim_z, 1]
            z_smooth[t] = z_f + torch.bmm(J_k, delta_z).squeeze(-1)  # [B, dim_z]

            # Smoothed covariance
            delta_P = P_smooth[t+1] - P_p   # [B, dim_z, dim_z]
            P_smooth[t] = P_f + torch.bmm(J_k, torch.bmm(delta_P, J_k.transpose(1, 2)))  # [B, dim_z, dim_z]
            P_smooth[t] = (P_smooth[t] + P_smooth[t].transpose(1, 2)) / 2.0  

        z_smooth = torch.stack(z_smooth, dim=1)  # [B, T, dim_z]
        P_smooth = torch.stack(P_smooth, dim=1)  # [B, T, dim_z, dim_z]
        P_smooth = P_smooth# + 1e-4 * torch.eye(dim_z, device=device).unsqueeze(0).unsqueeze(0)

        z_dist = D.MultivariateNormal(
            loc=z_smooth,
            scale_tril=self._safe_cholesky(P_smooth)
        )

        if self.training:
            z_smooth_sample = z_dist.rsample()
        else:
            z_smooth_sample = z_smooth

        # Smoothed observation and prediction
        z_pred_smooth = []
        P_pred_smooth = []
        a_smooth = []
        a_pred_smooth = []

        for t in range(T):
            z_t = z_smooth_sample[:, t, :]  # [B, dim_z]
            C_t = C_list[:, t, :, :]        # [B, dim_a, dim_z]

            a_t = torch.bmm(C_t, z_t.unsqueeze(-1)).squeeze(-1)  # [B, dim_a]
            
            A_t = A_list[:, t]

            z_next = torch.bmm(A_t, z_smooth[:, t, :].unsqueeze(-1)).squeeze(-1)  # [B, dim_z]
            P_next = torch.bmm(A_t, torch.bmm(P_smooth[:, t, :, :], A_t.transpose(1, 2))) + self.Q  # [B, dim_z, dim_z]
            P_next = (P_next + P_next.transpose(1, 2)) / 2.0# + 1e-4 * torch.eye(dim_z, device=device).unsqueeze(0)

            a_pred = torch.bmm(C_list[:, t, :, :], z_next.unsqueeze(-1)).squeeze(-1)  # [B, dim_a]

            a_smooth.append(a_t)
            z_pred_smooth.append(z_next)
            P_pred_smooth.append(P_next)
            a_pred_smooth.append(a_pred)

        z_pred_smooth = torch.stack(z_pred_smooth, dim=1)
        P_pred_smooth = torch.stack(P_pred_smooth, dim=1)
        a_smooth = torch.stack(a_smooth, dim=1)
        a_pred_smooth = torch.stack(a_pred_smooth, dim=1)

        return z_smooth_sample, P_smooth, z_pred_smooth, P_pred_smooth, a_smooth, a_pred_smooth, z_dist         

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

        # Kalman filter
        z_filt, P_filt, z_pred, a_filt, a_pred, P_pred, alpha_seq, A_list, C_list, z_dist_filt, S_pred = self._kalman_filter(
            a_seq       = a_seq,
            alpha_net   = alpha_net,
            h_obs       = h_obs,
            A_matrices  = A_matrices,
            C_matrices  = C_matrices,
            B_matrices  = B_matrices,
            u_seq       = u_seq,
            mask        = mask,
            epoch       = epoch
        )
        A_list = A_list
        C_list = C_list
        # RTS smoother
        z_smooth, P_smooth, z_pred_smooth, P_pred_smooth, a_smooth, a_pred_smooth, z_dist = self._rts_smoother(
            z_filt_list  = z_filt,
            P_filt_list  = P_filt,
            z_pred_list  = z_pred,
            P_pred_list  = P_pred,
            A_list       = A_list,
            C_list       = C_list
        )
        #return z_filt, P_filt, z_dist_filt, z_pred, P_pred, a_filt, a_pred, alpha_seq, S_pred
        return z_smooth, P_smooth, z_dist, z_pred_smooth, P_pred_smooth, a_smooth, a_pred_smooth, alpha_seq, S_pred