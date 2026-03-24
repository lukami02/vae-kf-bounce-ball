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

def alpha_entropy_loss(alpha_seq):
    eps = 1e-8
    entropy = - (alpha_seq * (alpha_seq + eps).log()).sum(dim=-1)
    return -entropy.mean() 

def compute_loss( ball_seq, x_dist_filt, a_dist, a_seq, a_pred, a_filt, z_pred, z_filt, P_pred, P_filt, R, Q, alpha_seq,
                cfg: VAEConfig, tcfg: TrainConfig, epoch):
    """
    Computes ELBO loss:
        F = log p(x|a) + log p(a|z) + log p(z|u) - log q(a|x) - log p(z|a,u)

        ball_seq:    [B, T, H, W]            — ground truth frames
        x_dist_filt: Bernoulli [B, T, H, W]  — reconstruction from filtered state
        a_dist:      Normal [B, T, dim_a]    — encoder distribution q(a|x)
        a_seq:       [B, T, dim_a]           — sample from a_dist
        a_pred:      [B, T, dim_a]           — predicted observation from z_pred
        a_filt:      [B, T, dim_a]           — filtered observation C @ z_filt
        z_pred:      [B, T, dim_z]           — predicted latent state (rsample)
        z_filt:      [B, T, dim_z]           — filtered latent state
        P_pred:      [B, T, dim_z, dim_z]    — predicted state covariance
        P_filt:      [B, T, dim_z, dim_z]    — filtered state covariance
        R:           [dim_a, dim_a]          — observation noise covariance
        Q:           [dim_z, dim_z]          — transition noise covariance
    """
    B, T, dim_z = z_filt.shape if z_filt is not None else (*a_seq.shape[:2], 0)
    device = ball_seq.device

    # log p(x | a) — reconstruction
    logits = x_dist_filt.logits
    pos_weight = torch.tensor(tcfg.pos_weight, device=device)
    L_recon = F.binary_cross_entropy_with_logits(
        logits,
        ball_seq,
        pos_weight=pos_weight,
        reduction='none'
    ).sum(dim=(2, 3)).mean()

    if z_pred is not None and P_filt is not None:  # KVAE
        # log p(a | z) — innovation
        L_innov = - D.MultivariateNormal(a_filt, R).log_prob(a_seq).mean()

        # log p(z | u) — state prior
        L_prior_z0 = - D.MultivariateNormal(
            torch.zeros(dim_z, device=device, dtype=z_filt.dtype),
            torch.eye(dim_z, device=device, dtype=z_filt.dtype)
        ).log_prob(z_filt[:, 0, :]).mean()

        L_prior_trans = - D.MultivariateNormal(
            z_pred[:, :-1, :].reshape(-1, dim_z),
            Q.unsqueeze(0).expand(B * (T - 1), -1, -1)
        ).log_prob(z_filt[:, 1:, :].reshape(-1, dim_z)).mean()

        L_prior = L_prior_z0 + L_prior_trans

        # log q(a | x) — encoder entropy
        L_entropy = a_dist.log_prob(a_seq).sum(-1).mean()

        # log p(z | a, u) — Kalman posterior 
        P_filt_reg = P_filt[:, 1:, :, :].reshape(-1, dim_z, dim_z)
        P_filt_reg = P_filt_reg + cfg.QR_reg * torch.eye(dim_z, device=device).unsqueeze(0)

        L_posterior = - D.MultivariateNormal(
            z_filt[:, 1:, :].reshape(-1, dim_z),
            P_filt_reg
        ).log_prob(z_pred[:, :-1, :].reshape(-1, dim_z)).mean()

        # alpha entropy
        L_alpha =  alpha_entropy_loss(alpha_seq)

        loss = (tcfg.lambda_recon * L_recon +
                tcfg.lambda_innov * L_innov +
                tcfg.lambda_posterior * L_posterior +
                tcfg.lambda_prior * L_prior -
                tcfg.lambda_entropy * L_entropy +
                tcfg.lambda_alpha * L_alpha
        )

        terms = {
            "loss":      loss.item(),
            "recon":     L_recon.item(),
            "innov":     L_innov.item(),
            "prior":     L_prior.item(),
            "entropy":   L_entropy.item(),
            "posterior": L_posterior.item(),
        }

    else:  # CV_VAE / GRU_VAE
        # KL(q(a|x) || N(0,I)) — encoder regularization
        mu  = a_dist.loc
        var = a_dist.scale ** 2
        L_regularization = 0.5 * (mu**2 + var - var.log() - 1).sum(-1).mean()

        # log p(a_{t+1} | z_pred_t) — prediction
        a_dist_next = D.Normal(a_dist.loc[:, 1:, :], a_dist.scale[:, 1:, :])
        L_prediction = -a_dist_next.log_prob(a_pred[:, :-1, :]).sum(-1).mean()

        # log q(a | x) — encoder entropy
        L_entropy = a_dist.log_prob(a_seq).sum(-1).mean()

        loss = (tcfg.lambda_recon * L_recon
              + tcfg.lambda_pred * L_prediction
              + tcfg.get_lambda_kl(epoch) * L_regularization
              - tcfg.lambda_entropy * L_entropy)

        terms = {
            "loss":    loss.item(),
            "recon":   L_recon.item(),
            "reg":     L_regularization.item(),
            "pred":    L_prediction.item(),
            "entropy": L_entropy.item()
        }

    return loss, terms