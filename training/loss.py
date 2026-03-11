import torch
import torch.nn.functional as F
from config.train_config import TrainConfig
from config.vae_config import VAEConfig


def kl_divergence_gaussian(mu_q, var_q, mu_p, var_p):
    """
    Closed-form KL divergence between two diagonal Gaussians.
    KL(q || p) = 0.5 * sum(log(var_p/var_q) + var_q/var_p + (mu_q - mu_p)^2 / var_p - 1)

    mu_q, var_q:  [B*T, dim_a]  — VAE posterior (encoder output)
    mu_p, var_p:  [B*T, dim_a]  — Kalman prior (C @ z_pred)
    """
    kl = 0.5 * (torch.log(var_p / (var_q + 1e-8)) + var_q / (var_p + 1e-8) + (mu_q - mu_p) ** 2 / (var_p + 1e-8)- 1.0)
    return kl.sum(dim=-1)   # [B*T]


def innovation_loss(r_k, S_k):
    """
    Negative log likelihood of innovation.
    L = r_k^T S_k^{-1} r_k + log|S_k|

    r_k: [B, T, dim_a]          — innovation a_k - C_k @ z_pred_k
    S_k: [B, T, dim_a, dim_a]   — innovation covariance C @ P_pred @ C^T + R_k
    """
    B, T, dim_a = r_k.shape

    r = r_k.view(B * T, dim_a, 1)                                   # [B*T, dim_a, 1]
    S = S_k.view(B * T, dim_a, dim_a)                               # [B*T, dim_a, dim_a]

    # r^T S^{-1} r
    S_inv = torch.linalg.inv(S)                                    # [B*T, dim_a, dim_a]
    mahal = torch.bmm(r.transpose(1, 2), torch.bmm(S_inv, r))      # [B*T, 1, 1]
    mahal = mahal.squeeze(-1).squeeze(-1)                          # [B*T]

    # log|S|
    #log_det = torch.linalg.slogdet(S)[1]                           # [B*T]

    return mahal.mean()


def compute_loss( ball_seq, x_hat_filt, x_hat_pred, a_mu, a_var, z_pred, P_pred, a_filt, alpha_seq, C_matrices,
                cfg: VAEConfig, tcfg: TrainConfig, epoch: int, mask=None):
    """
    ball_seq,       # [B, T, H, W]     — ground truth
    x_hat_filt,     # [B, T, H, W]     — reconstruction from a_filt
    x_hat_pred,     # [B, T, H, W]     — reconstruction from a_pred
    a_mu,           # [B*T, dim_a]     — encoder mean
    a_var,          # [B*T, dim_a]     — encoder variance
    z_pred,         # [B, T, dim_z]    — Kalman predicted state
    P_pred,         # [B, T, dim_z, dim_z]
    a_filt,         # [B, T, dim_a]    — C @ z_filt
    alpha_seq,      # [B, T, K]
    C_matrices,     # [K, dim_a, dim_z]
    """
    B, T, H, W = ball_seq.shape

    # Reconstruction loss  —  E_q[log p(x | a_filt)]
    if mask is not None:
        m = mask.unsqueeze(-1).unsqueeze(-1)      
        L_recon = F.mse_loss(x_hat_filt * m, ball_seq * m)
    else:
        L_recon = F.mse_loss(x_hat_filt, ball_seq)

    # Prediction loss  —  E_q[log p(x_{k+1} | a_pred_k)]
    L_pred = F.mse_loss(x_hat_pred[:, :-1], ball_seq[:, 1:])

    L_free = torch.tensor(0.0, device=ball_seq.device)
    if mask is not None:
        free_mask = (1.0 - mask).unsqueeze(-1).unsqueeze(-1)       
        n_free = free_mask.sum().clamp(min=1)
        L_free = ((x_hat_filt - ball_seq) ** 2 * free_mask).sum() / n_free

    if z_pred is not None and P_pred is not None: # KVAE
        # KL divergence  —  KL(q(a|x) || p(a|z))
        # p(a_k|z_k) = N(C_k @ z_pred_k, R_kalman)

        # Kalman prior mean: mu_p = C_k @ z_pred_k
        w = alpha_seq.unsqueeze(-1).unsqueeze(-1)                              # [B, T, K, 1, 1]
        C_seq = (w * C_matrices.unsqueeze(0).unsqueeze(0)).sum(dim=2)          # [B, T, dim_a, dim_z]
        mu_p  = torch.bmm(
                    C_seq.view(B * T, cfg.dim_a, cfg.dim_z),
                    z_pred.view(B * T, cfg.dim_z, 1)
                ).squeeze(-1)                                                  # [B*T, dim_a]

        # Kalman prior variance: var_p = C_k @ P_pred_k @ C_k^T + a_va
        C_flat   = C_seq.view(B * T, cfg.dim_a, cfg.dim_z)                     # [B*T, dim_a, dim_z]
        P_flat   = P_pred.view(B * T, cfg.dim_z, cfg.dim_z)                    # [B*T, dim_z, dim_z]
        CP       = torch.bmm(C_flat, P_flat)                                   # [B*T, dim_a, dim_z]
        CPCt     = torch.bmm(CP, C_flat.transpose(1, 2))                       # [B*T, dim_a, dim_a]
        var_p    = torch.diagonal(CPCt, dim1=-2, dim2=-1) + a_var + 1e-8       # [B*T, dim_a]

        if mask is not None:
            obs_mask_bt = mask.contiguous().reshape(B * T)   
            L_kl = (kl_divergence_gaussian(a_mu, a_var, mu_p, var_p) * obs_mask_bt).mean()
        else:
            L_kl = kl_divergence_gaussian(a_mu, a_var, mu_p, var_p).mean()

        # Innovation loss 
        # r_k = a_k - C_k @ z_pred_k
        # S_k = C_k @ P_pred @ C_k^T + R_k

        # Innovation
        a_seq_bt = a_mu.view(B, T, cfg.dim_a)                                   # [B, T, dim_a]
        r_k = a_seq_bt - a_filt                                                 # [B, T, dim_a]

        # S_k = CPCt + R_k
        R_k  = torch.diag_embed(a_var.view(B, T, cfg.dim_a))                    # [B, T, dim_a, dim_a]
        S_k  = CPCt.view(B, T, cfg.dim_a, cfg.dim_a) + R_k                      # [B, T, dim_a, dim_a]

        if mask is not None:
            obs_mask = mask.unsqueeze(-1)                                       # [B, T, 1]
            r_k = r_k * obs_mask                                                # [B, T, dim_a]

        L_innov = innovation_loss(r_k, S_k)
    else:
        L_kl    = torch.tensor(0.0)
        L_innov = torch.tensor(0.0)

    # Weighted sum sa annealing
    lam_kl   = tcfg.get_lambda_kl(epoch)
    lam_pred = tcfg.get_lambda_pred(epoch)
    lam_free = tcfg.get_lambda_free(epoch)

    loss = (tcfg.lambda_recon * L_recon + lam_pred * L_pred + lam_kl * L_kl + tcfg.lambda_innov * L_innov + lam_free * L_free)

    terms = {
        "loss": loss.item(),
        "recon": L_recon.item(),
        "pred": L_pred.item(),
        "kl": L_kl.item(),
        "innov": L_innov.item(),
        "free":  L_free.item(),
    }

    return loss, terms