import torch
import torch.nn as nn
import torch.distributions as D
import math
import sys
sys.path.append("..")
from config.vae_config import VAEConfig
from config.train_config import TrainConfig


class KalmanFilter(nn.Module):
    """
    Linear Kalman Filter operating in latent space.

    State space model:
        z_k = A_k @ z_{k-1} + B_k @ u_k + w_k,   w_k ~ N(0, Q)
        a_k = C_k @ z_k + v_k,                   v_k ~ N(0, R)
    """
    def __init__(self, cfg: VAEConfig, tcfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.tcfg = tcfg

        # init state
        self.P_0 = nn.Parameter(10*torch.eye(cfg.dim_z), requires_grad=False)     # [dim_z, dim_z]
        self.a_0 = nn.Parameter(torch.zeros(1, cfg.dim_a), requires_grad=True)    # [dim_a]

        # Noise covariances
        self.register_buffer("Q", cfg.Q_std * torch.eye(cfg.dim_z))
        self.register_buffer("R", cfg.R_std * torch.eye(cfg.dim_a))

    def _safe_cholesky(self, M, jitter=1e-4): 
        """
        Numerically stable Cholesky decomposition.
        """
        I = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
        I_batch = I if M.dim() == 2 else I.unsqueeze(0).expand_as(M)
        
        M = 0.5 * (M + M.transpose(-1, -2)) + jitter * I_batch
        
        # Try decomposition with progressively larger jitter
        for scale in [1.0, 10.0, 100.0, 1000.0]:
            try:
                return torch.linalg.cholesky(M + scale * jitter * I_batch)
            except torch.linalg.LinAlgError:
                continue
        
        return torch.linalg.cholesky(I_batch)
    
    def _kalman_filter(self, a_seq, alpha_net, h_obs, A_matrices, C_matrices, B_matrices=None, u_seq=None, mask=None, epoch=100):
        """
        Perform Kalman filtering to compute filtered state estimates.

        a_seq:       [B, T, dim_a]        — observation sequence (encoder outputs)
        alpha_net:   nn.Module            — mixture weight network
        h_obs:       [B, dim_obstacle]    — static obstacle context
        A_matrices:  [K, dim_z, dim_z]    — state transition matrices (one per mixture component)
        C_matrices:  [K, dim_a, dim_z]    — observation matrices
        B_matrices:  [K, dim_z, dim_u]    — control matrices (optional)
        u_seq:       [B, T, dim_u]        — control inputs (optional)
        mask:        [B, T]               — binary mask for valid timesteps
        epoch:       int                  — current training epoch (controls temperature/burn-in)

        z_filt:      [B, T, dim_z]        — filtered latent states
        P_filt:      [B, T, dim_z, dim_z] — filtered covariances
        z_pred:      [B, T, dim_z]        — predicted states (prior to update)
        a_filt:      [B, T, dim_a]        — reconstructed observations from filtered states
        a_pred:      [B, T, dim_a]        — predicted observations from predicted states
        P_pred:      [B, T, dim_z, dim_z] — predicted covariances
        alpha_seq:   [B, T, K]            — mixture weights over time
        A_list:      [B, T, dim_z, dim_z] — effective transition matrices
        B_list:      [B, T, dim_z, dim_u] — effective control matrices (or None)
        C_list:      [B, T, dim_a, dim_z] — effective observation matrices
        z_dist:      MultivariateNormal   — filtered state distribution
        S_pred:      [B, T, dim_a, dim_a] — innovation covariance
        """
        B, T, dim_a = a_seq.shape
        dim_z = self.cfg.dim_z
        device = a_seq.device

        I = torch.eye(dim_z, device=device).unsqueeze(0)    # [1, dim_z, dim_z]

        if mask is None:
            mask = torch.ones(B, T, device=device)

        # Storage for per-timestep outputs
        z_filt_list, P_filt_list = [], []
        z_pred_list, P_pred_list = [], []
        a_filt_list, a_pred_list = [], []
        A_list, B_list, C_list= [], [], [] 
        alpha_list = []
        z_mean_list = []
        z_scale_tril_list = []
        S_pred_list = []

        # Initialize state and covariance
        z_prev = torch.zeros(B, dim_z, device=device)        
        P = self.P_0.unsqueeze(0).expand(B, -1, -1).clone() 

        # Initialize GRU state for the alpha network
        gru_state = alpha_net.init_state(B, device)

        # Compute initial mixture weights
        u_0 =  u_seq[:, 0, :] if u_seq is not None else None
        a_prev = self.a_0.expand(B, -1)
        alpha_k, gru_state = alpha_net(a_prev, h_obs, gru_state, u_0)

        # Detach alpha during warmup to stabilize early training
        if epoch < self.tcfg.alpha_warmup_epochs:
            alpha_k = alpha_k.detach()   

        w = alpha_k.unsqueeze(-1).unsqueeze(-1)                  
        A_k = (w * A_matrices.unsqueeze(0)).sum(dim=1)                # [B, dim_z, dim_z]
        B_k = (w * B_matrices.unsqueeze(0)).sum(dim=1)                # [B, dim_z, dim_u]
        C_k = (w * C_matrices.unsqueeze(0)).sum(dim=1)                # [B, dim_a, dim_z]

        for k in range(T):
            a_k    = a_seq[:, k, :]                                           # [B, dim_a]
            mask_k = mask[:, k].unsqueeze(-1)                                 # [B, 1]
            u_k    = u_seq[:, k, :] if u_seq is not None else None

            # Predict
            z = torch.bmm(A_k, z_prev.unsqueeze(-1)).squeeze(-1)              # [B, dim_z]
            if B_matrices is not None and u_k is not None:
                z = z + torch.bmm(B_k, u_k.unsqueeze(-1)).squeeze(-1)

            # Update 
            a_k_hat = torch.bmm(C_k, z.unsqueeze(-1)).squeeze(-1)             # [B, dim_a]
            r_k = a_k - a_k_hat                                               # [B, dim_a] 

            # Innovation covariance
            S_k = torch.bmm(C_k, torch.bmm(P, C_k.transpose(1, 2))) + self.R.unsqueeze(0).expand(B, -1, -1)

            # Kalman gain
            K_k = torch.linalg.solve(
                      S_k.transpose(1, 2),
                      torch.bmm(C_k, P.transpose(1, 2))
                  ).transpose(1, 2)     # [B, dim_z, dim_a]

            # Zero out gain for masked (invalid) timestep
            K_k = K_k * mask_k.unsqueeze(-1)                                  # [B, dim_z, dim_a]

            # Joseph form covariance update
            IKC    = I - torch.bmm(K_k, C_k)                                  # [B, dim_z, dim_z]
            P_filt = torch.bmm(IKC, torch.bmm(P, IKC.transpose(1, 2))) \
                   + torch.bmm(K_k, torch.bmm(self.R.unsqueeze(0).expand(B, -1, -1), K_k.transpose(1, 2))) 
            P_filt = (P_filt + P_filt.transpose(1, 2)) / 2.0

            # Filtered state mean and distribution
            z_filt_mean = z + torch.bmm(K_k, r_k.unsqueeze(-1)).squeeze(-1)    # [B, dim_z]
            L_k = self._safe_cholesky(P_filt)
            z_dist_k = D.MultivariateNormal(loc=z_filt_mean, scale_tril=L_k)

            z_mean_list.append(z_filt_mean)
            z_scale_tril_list.append(L_k)
            z_filt = z_dist_k.mean

            # Reconstruct observation from filtered state
            a_filt = torch.bmm(C_k, z_filt.unsqueeze(-1)).squeeze(-1)             # [B, dim_a]

            # Use real observation where valid, filtered reconstruction where masked
            a_k_hat = mask_k * a_k + (1 - mask_k) * a_filt
            
            C_list.append(C_k)

            # Update mixture weights for next step
            alpha_k, gru_state = alpha_net(a_k_hat, h_obs, gru_state, u_k)
            if epoch < self.tcfg.alpha_warmup_epochs:
                alpha_k = alpha_k.detach()   

            # Compose matrices
            w = alpha_k.unsqueeze(-1).unsqueeze(-1)                       # [B, K, 1, 1]
            A_k = (w * A_matrices.unsqueeze(0)).sum(dim=1)                # [B, dim_z, dim_z]
            C_k = (w * C_matrices.unsqueeze(0)).sum(dim=1)                # [B, dim_a, dim_z]
            A_list.append(A_k)
            
            # One-step-ahead prediction
            z_pred = torch.bmm(A_k, z_filt.unsqueeze(-1)).squeeze(-1)
            
            if B_matrices is not None and u_k is not None:
                B_k = (w * B_matrices.unsqueeze(0)).sum(dim=1)            # [B, dim_z, dim_u]
                B_list.append(B_k)
                z_pred = z_pred + torch.bmm(B_k, u_k.unsqueeze(-1)).squeeze(-1)
            else: 
                B_list.append(None)

            a_pred = torch.bmm(C_k, z_pred.unsqueeze(-1)).squeeze(-1)
            P_pred = torch.bmm(A_k, torch.bmm(P_filt, A_k.transpose(1, 2))) + self.Q
            P_pred = (P_pred + P_pred.transpose(1, 2)) / 2.0
            
            # Detach gradients during burn-in phase
            if k < self.tcfg.burn_in and epoch < self.tcfg.epoch_burn_in:
                z_filt = z_filt.detach()
                P_filt = P_filt.detach()
                z_pred = z_pred.detach()
                P_pred = P_pred.detach()

            # Accumulate results
            z_filt_list.append(z_filt)
            P_filt_list.append(P_filt)
            z_pred_list.append(z_pred)
            P_pred_list.append(P_pred)
            alpha_list.append(alpha_k)
            a_filt_list.append(a_filt)
            a_pred_list.append(a_pred)
            S_pred_list.append(S_k)

            # Carry filtered state and predicted covariance to next iteration
            z_prev   = z_filt
            P        = P_pred

        z_filt    = torch.stack(z_filt_list, dim=1)
        P_filt    = torch.stack(P_filt_list, dim=1)
        z_pred    = torch.stack(z_pred_list, dim=1)
        P_pred    = torch.stack(P_pred_list, dim=1)
        alpha_seq = torch.stack(alpha_list,  dim=1)
        A_list    = torch.stack(A_list, dim=1)
        B_list    = torch.stack(B_list, dim=1) if B_list[0] is not None else None
        C_list    = torch.stack(C_list, dim=1)
        a_filt    = torch.stack(a_filt_list, dim=1)
        a_pred    = torch.stack(a_pred_list, dim=1)
        S_pred    = torch.stack(S_pred_list, dim=1)
 
        z_means      = torch.stack(z_mean_list, dim=1)
        z_scale_tril = torch.stack(z_scale_tril_list, dim=1)
        z_dist       = D.MultivariateNormal(loc=z_means, scale_tril=z_scale_tril)
 
        return z_filt, P_filt, z_pred, a_filt, a_pred, P_pred, alpha_seq, A_list, B_list, C_list, z_dist, S_pred
 
    
    def _rts_smoother(self, z_filt_list, P_filt_list, z_pred_list, P_pred_list, A_list, B_list, C_list, u_seq=None):
        """
        Rauch-Tung-Striebel smoother for computing smoothed state estimates.

        z_filt_list:  [B, T, dim_z]        — filtered means
        P_filt_list:  [B, T, dim_z, dim_z] — filtered covariances
        z_pred_list:  [B, T, dim_z]        — predicted means
        P_pred_list:  [B, T, dim_z, dim_z] — predicted covariances
        A_list:       [B, T, dim_z, dim_z] — effective transition matrices
        B_list:       [B, T, dim_z, dim_u] — effective control matrices (or None)
        C_list:       [B, T, dim_a, dim_z] — effective observation matrices
        u_seq:        [B, T, dim_u]        — control inputs (or None)

        z_smooth:       [B, T, dim_z]        — smoothed states (sampled if training)
        P_smooth:       [B, T, dim_z, dim_z] — smoothed covariances
        z_pred_smooth:  [B, T, dim_z]        — one-step predictions from smoothed states
        P_pred_smooth:  [B, T, dim_z, dim_z] — predicted covariances from smoothed states
        a_smooth:       [B, T, dim_a]        — reconstructed observations from smoothed states
        a_pred_smooth:  [B, T, dim_a]        — predicted observations from smoothed states
        z_dist:         MultivariateNormal   — smoothed state distribution
        """

        B, T, _ = z_filt_list.shape
        device = z_filt_list[0].device
        dim_z = self.cfg.dim_z

        z_smooth = [None] * T
        P_smooth = [None] * T

        # Initialize with last filtered estimate
        z_smooth[-1] = z_filt_list[:, -1, :] 
        P_smooth[-1] = P_filt_list[:, -1, :, :]

        for t in reversed(range(T - 1)):
            P_f = P_filt_list[:, t, :, :]     # [B, dim_z, dim_z]
            P_p = P_pred_list[:, t, :, :]     # [B, dim_z, dim_z]
            A_k = A_list[:, t, :, :]          # [B, dim_z, dim_z]
            z_f = z_filt_list[:, t, :]        # [B, dim_z]
            z_p = z_pred_list[:, t, :]        # [B, dim_z] 

            # Smoother gain
            J_k = torch.linalg.solve(
                P_p.transpose(1, 2),
                torch.bmm(A_k, P_f.transpose(1, 2))
            ).transpose(1, 2)           # [B, dim_z, dim_z]

            # Smoothed mean
            delta_z = (z_smooth[t+1] - z_p).unsqueeze(-1)            # [B, dim_z, 1]
            z_smooth[t] = z_f + torch.bmm(J_k, delta_z).squeeze(-1)  # [B, dim_z]

            # Smoothed covariance
            delta_P = P_smooth[t+1] - P_p
            P_smooth[t] = P_f + torch.bmm(J_k, torch.bmm(delta_P, J_k.transpose(1, 2)))  # [B, dim_z, dim_z]
            P_smooth[t] = (P_smooth[t] + P_smooth[t].transpose(1, 2)) / 2.0  

        z_smooth = torch.stack(z_smooth, dim=1)  # [B, T, dim_z]
        P_smooth = torch.stack(P_smooth, dim=1)  # [B, T, dim_z, dim_z]

        # Build smoothed distribution and sample (reparameterized) during training
        z_dist = D.MultivariateNormal(
            loc=z_smooth,
            scale_tril=self._safe_cholesky(P_smooth)
        )

        z_smooth_sample = z_dist.rsample() if self.training else z_smooth

        # Compute reconstructed/predicted observations and one-step-ahead predictions
        z_pred_smooth, P_pred_smooth, a_smooth, a_pred_smooth = [], [], [], []

        for t in range(T):
            z_t = z_smooth_sample[:, t, :]    # [B, dim_z]
            C_t = C_list[:, t, :, :]          # [B, dim_a, dim_z]
            A_t = A_list[:, t]                # [B, dim_z, dim_z]

            # Reconstructed observation from smoothed state
            a_t = torch.bmm(C_t, z_t.unsqueeze(-1)).squeeze(-1)
 
            # Predicted next state from smoothed mean
            z_next = torch.bmm(A_t, z_smooth[:, t, :].unsqueeze(-1)).squeeze(-1)

            if B_list is not None and u_seq is not None:
                B_t = B_list[:, t, :, :]       # [B, dim_z, dim_u]
                u_t = u_seq[:, t, :]           # [B, dim_u]
                z_next = z_next + torch.bmm(B_t, u_t.unsqueeze(-1)).squeeze(-1)
            
            # Predicted covariance from smoothed state
            P_next = torch.bmm(A_t, torch.bmm(P_smooth[:, t, :, :], A_t.transpose(1, 2))) + self.Q  # [B, dim_z, dim_z]
            P_next = (P_next + P_next.transpose(1, 2)) / 2.0

            # Predicted observation from predicted next state
            a_pred = torch.bmm(C_list[:, t, :, :], z_next.unsqueeze(-1)).squeeze(-1)  # [B, dim_a]

            a_smooth.append(a_t)
            z_pred_smooth.append(z_next)
            P_pred_smooth.append(P_next)
            a_pred_smooth.append(a_pred)

        z_pred_smooth = torch.stack(z_pred_smooth, dim=1)
        P_pred_smooth = torch.stack(P_pred_smooth, dim=1)
        a_smooth      = torch.stack(a_smooth, dim=1)
        a_pred_smooth = torch.stack(a_pred_smooth, dim=1)

        return z_smooth_sample, P_smooth, z_pred_smooth, P_pred_smooth, a_smooth, a_pred_smooth, z_dist 

    def forward(self, a_seq, alpha_net, h_obs, A_matrices, C_matrices, B_matrices=None, u_seq=None, mask=None, epoch=100):
        """
        a_seq:       [B, T, dim_a]     — encoder output sequence
        alpha_net:   AlphaNetwork      — mixture weight network
        h_obs:       [B, dim_obstacle] — static obstacle context
        A_matrices:  [K, dim_z, dim_z] — transition matrix bank
        C_matrices:  [K, dim_a, dim_z] — observation matrix bank
        B_matrices:  [K, dim_z, dim_u] — control matrix bank (optional)
        u_seq:       [B, T, dim_u]     — control input sequence (optional)
        mask:        [B, T]            — validity mask
        epoch:       int               — current training epoch

        z_smooth:       [B, T, dim_z]        — smoothed latent states
        P_smooth:       [B, T, dim_z, dim_z] — smoothed covariances
        z_dist:         MultivariateNormal   — smoothed state distribution
        z_pred_smooth:  [B, T, dim_z]        — one-step predictions from smoothed states
        P_pred_smooth:  [B, T, dim_z, dim_z] — predicted covariances
        a_smooth:       [B, T, dim_a]        — reconstructed observations
        a_pred_smooth:  [B, T, dim_a]        — predicted observations
        alpha_seq:      [B, T, K]            — mixture weights over time
        """

        # Kalman filter
        z_filt, P_filt, z_pred, a_filt, a_pred, P_pred, alpha_seq, A_list, B_list, C_list, z_dist_filt, S_pred = self._kalman_filter(
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

        # RTS smoother
        z_smooth, P_smooth, z_pred_smooth, P_pred_smooth, a_smooth, a_pred_smooth, z_dist = self._rts_smoother(
            z_filt_list  = z_filt,
            P_filt_list  = P_filt,
            z_pred_list  = z_pred,
            P_pred_list  = P_pred,
            A_list       = A_list,
            B_list       = B_list,
            C_list       = C_list,
            u_seq        = u_seq
        )

        return z_smooth, P_smooth, z_dist, z_pred_smooth, P_pred_smooth, a_smooth, a_pred_smooth, alpha_seq