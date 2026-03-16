import torch
import torch.nn.functional as F
import torch.distributions as D
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


def innovation_loss(r_k, R):
    """
    Negative log likelihood of innovation.
    L = r_k^T S_k^{-1} r_k + log|S_k|

    r_k: [B, T, dim_a]          — innovation a_k - C_k @ z_pred_k
    S_k: [B, T, dim_a, dim_a]   — innovation covariance C @ P_pred @ C^T + R_k
    """
    B, T, dim_a = r_k.shape

    r = r_k.view(B * T, dim_a, 1)         

    R_inv = torch.linalg.inv(R)        
    R_inv = R_inv.unsqueeze(0)                                    

    mahal = torch.bmm(r.transpose(1,2), torch.bmm(R_inv.expand(r.shape[0],-1,-1), r))
    mahal = mahal.squeeze()

    # log|S|
    log_det = torch.linalg.slogdet(R)[1]    

    return (mahal + log_det).mean()

def transition_loss(z_seq, z_pred, Q):
    """
    Negative log-likelihood of Kalman state transitions.

    z_seq : [B, T, dim_z]
    z_pred : [B, T, dim_z, dim_z]
    Q     : [dim_z, dim_z]
    """
    B, T, dim_z = z_seq.shape

    r = z_seq - z_pred                        
    r = r.reshape(-1, dim_z, 1)                  

    Q_inv = torch.linalg.inv(Q)
    Q_inv = Q_inv.unsqueeze(0)

    mahal = torch.bmm(r.transpose(1,2), torch.bmm(Q_inv.expand(r.shape[0],-1,-1), r))
    mahal = mahal.squeeze()

    log_det = torch.linalg.slogdet(Q)[1]

    return (mahal + log_det).mean()

def compute_loss( ball_seq, x_dist_filt, a_dist, a_seq, z_pred, P_pred, z_filt, P_filt, alpha_seq, C_matrices, R, Q, 
                cfg: VAEConfig, tcfg: TrainConfig):
    """
    ball_seq,       # [B, T, H, W]     — ground truth
    x_dist_filt,     # [B, T, H, W]    — reconstruction distribution from a_filt
    x_dist_pred,     # [B, T, H, W]    — reconstruction distribution from a_pred
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
    L_recon = -x_dist_filt.log_prob(ball_seq).sum(dim=(2,3)).mean()

    # Regularization loss  —  E_q[q(a | x)]
    L_regularization = a_dist.log_prob(a_seq).sum(-1).mean()

    if z_pred is not None and P_pred is not None: # KVAE
        # Inovation loss — E_q[p(a | z)]
        w = alpha_seq[:, 1:].unsqueeze(-1).unsqueeze(-1)                # [B, T-1, K, 1, 1]
        C_seq = (w * C_matrices.unsqueeze(0).unsqueeze(0)).sum(dim=2)    # [B, T-1, dim_a, dim_z]

        z_pred_trim = z_pred[:, :-1, :]                                  # [B, T-1, dim_z]
        a_pred = torch.bmm(
            C_seq.reshape(-1, cfg.dim_a, cfg.dim_z),                     # [B*(T-1), dim_a, dim_z]
            z_pred_trim.reshape(-1, cfg.dim_z, 1)                        # [B*(T-1), dim_z, 1]
        ).squeeze(-1)                                                    # [B*(T-1), dim_a]
        a_target = a_seq[:, 1:, :].reshape(-1, cfg.dim_a)                # [B*(T-1), dim_a]

        kalman_observation_distrib = D.MultivariateNormal(a_pred, R)
        kalman_observation_log_likelihood = kalman_observation_distrib.log_prob(a_target)  # [B*(T-1)]  za ovaj loss da pomnozi
        L_innov = -kalman_observation_log_likelihood.mean()                           

        # Prior loss E_q[ln p(z_t | z_{t-1})]

        # z_0 ~ N(0, I)
        z0_prior = D.MultivariateNormal(
            loc=torch.zeros(cfg.dim_z, device=z_pred.device, dtype=z_pred.dtype),
            covariance_matrix=torch.eye(cfg.dim_z, device=z_pred.device, dtype=z_pred.dtype)
        )
        L_prior_z0 = -z0_prior.log_prob(z_pred[:, 0, :]).mean()  # [B]

        # z_t ~ N(z_pred_{t-1}, Q)  for t=1..T-1
        z_prior_trans = D.MultivariateNormal(
            loc=z_pred[:, :-1, :].reshape(-1, cfg.dim_z),                        # [B*(T-1), dim_z]
            covariance_matrix=Q.unsqueeze(0).expand(B * (T - 1), -1, -1)         # [B*(T-1), dim_z, dim_z]
        )
        L_prior_trans = -z_prior_trans.log_prob(z_pred[:, 1:, :].reshape(-1, cfg.dim_z)).mean()  # [B*(T-1)]

        L_prior = L_prior_z0 + L_prior_trans

        # Posterior loss E_q[p(z | a)]
        B, T, dim_z = z_pred.shape
        
        posterior_distrib = D.MultivariateNormal(loc=z_filt.reshape(-1, dim_z), covariance_matrix=P_filt.reshape(-1, dim_z, dim_z))
        posterior_log_prob = posterior_distrib.log_prob(z_pred.reshape(-1, dim_z))  
        L_posterior = posterior_log_prob.mean()

    else:
        L_innov = torch.tensor(0.0)
        L_posterior = torch.tensor(0.0)
        L_prior = torch.tensor(0.0)

    loss = (tcfg.lambda_recon * L_recon + tcfg.lambda_reg * L_regularization + tcfg.lambda_kalman * (L_innov + L_prior + L_posterior))

    terms = {
        "loss": loss.item(),
        "recon": L_recon.item(),
        "reg": L_regularization.item(),
        "kalman": (L_innov + L_prior + L_posterior).item(),
    }

    return loss, terms