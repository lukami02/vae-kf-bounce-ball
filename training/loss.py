import torch
import torch.nn.functional as F
import torch.distributions as D
import math
from config.train_config import TrainConfig
from config.vae_config import VAEConfig

def innovation_loss(a_filt, a_seq, R):
    """
    Stable innovation negative log-likelihood.

    a_filt : [B, T, dim_a]  — predicted observation 
    a_seq  : [B, T, dim_a]  — true observation
    R : [B, dim_a, dim_a]   — observation covariance
    """
    B, T, dim_a = a_filt.shape

    mu = a_filt.reshape(-1, dim_a)   # [B*T, dim_a]
    x = a_seq.reshape(-1, dim_a)     # [B, T, dim_a]

    S = R.unsqueeze(0).expand(B * T, dim_a, dim_a)  # [B*T, dim_a, dim_a]
    L = torch.linalg.cholesky(S)

    # compute log likelihood
    dist = D.MultivariateNormal(mu, scale_tril=L)
    log_prob = dist.log_prob(x)    

    return - log_prob.mean()

def alpha_bounce_loss(a_seq, alpha_seq, u_seq, mask):
    """
    Bounce-aware regularization loss for alpha dynamics.
    """
    # valid transitions
    mask_da = mask[:, 1:] * mask[:, :-1]
    mask_bounce = mask_da[:, :-1] * mask_da[:, 1:]

    # detect gravity-free episodes
    has_gravity = (u_seq.abs().sum(dim=[1, 2]) > 0)  # [B]
    no_gravity_mask = (~has_gravity).float()         # [B]

    # velocity approximation
    da = a_seq[:, 1:] - a_seq[:, :-1]
    da_prev = da[:, :-1]
    da_next = da[:, 1:]

    # bounce signal from direction change
    cos_sim = F.cosine_similarity(da_prev, da_next, dim=-1)
    bounce_signal = (1 - cos_sim).clamp(0, 1)
    bounce_signal = bounce_signal / (bounce_signal.sum() + 1e-8)

    # alpha smoothness / change measure
    alpha_cos = F.cosine_similarity(
        alpha_seq[:, 2:],
        alpha_seq[:, 1:-1],
        dim=-1
    )

    loss_per_example = (bounce_signal.detach() * (alpha_cos ** 2) * mask_bounce).sum(dim=1)

    # only apply on no-gravity episodes
    loss = (loss_per_example * no_gravity_mask).sum()

    # normalize
    denom = no_gravity_mask.sum()
    if denom > 0:
        loss = loss / denom
    else:
        loss = loss * 0.0

    return loss

def kvae_compute_loss(ball_seq, x_dist_smooth,
                a_dist, a_seq, a_smooth,
                z_dist, z_smooth, z_pred, 
                R, Q, mask, alpha_seq,
                tcfg: TrainConfig, u_seq=None):
    """
    Computes ELBO loss:
        F = log p(x|a) + log p(a|z) + log p(z|u) - log q(a|x) - log p(z|a,u)

        ball_seq:      [B, T, H, W]          — sequence of ball images
        x_dist_smooth: [B, T, H, W]          — reconstruction from smoothed latents
        x_dist_pred:   [B, T, H, W]          — reconstruction from predicted latents
        a_dist:        Normal([B, T, dim_a]) — encoder distribution over latent observations
        a_seq:         [B, T, dim_a]         — sampled latent observations
        a_smooth:      [B, T, dim_a]         — reconstructed observations (from z_smooth)
        z_dist:        MultivariateNormal    — smoothed state distribution
        z_smooth:      [B, T, dim_z]         — smoothed latent states
        P_smooth:      [B, T, dim_z, dim_z]  — smoothed covariances
        z_pred:        [B, T, dim_z]         — predicted latent states
        P_pred:        [B, T, dim_z, dim_z]  — predicted covariances
        R:             [dim_a, dim_a]        — observation noise covariance
        Q:             [dim_z, dim_z]        — transition noise covariance
        mask:          [B, T]                — mask for valid timesteps
        alpha_seq:     [B, T, K]             — mixture weights over dynamics
        epoch:         int                   — current training epoch
        phase: 
        u_seq:         [B, T, dim_u]         — control inputs
    """

    B, T, dim_z = z_smooth.shape if z_smooth is not None else (*a_seq.shape[:2], 0)
    device = ball_seq.device

    if mask is None:
        mask = torch.ones(ball_seq.shape[:2], device=ball_seq.device)
    if u_seq in None:
        u_seq = torch.zeros(ball_seq.shape[:2], device=ball_seq.device)

    # log p(x | a) — reconstruction
    logits = x_dist_smooth.logits
    pos_weight = torch.tensor(tcfg.pos_weight, device=device)
    L_recon = F.binary_cross_entropy_with_logits(
        logits,
        ball_seq,
        pos_weight=pos_weight,
        reduction='none'
    ).sum(dim=(2, 3)).mean()

    # log p(a | z) — innovation
    L_innov = innovation_loss(a_smooth, a_seq, R)

    # log p(z | u) — state prior
    L_prior = - D.MultivariateNormal(
        loc=z_pred[:, :-1, :].reshape(-1, dim_z),
        scale_tril=torch.linalg.cholesky(Q)
    ).log_prob(z_smooth[:, 1:, :].reshape(-1, dim_z))
    L_prior = L_prior.mean()

    # log q(a | x) — encoder entropy
    L_entropy = - a_dist.log_prob(a_seq).sum(-1).mean()

    # log p(z | a, u) — Kalman posterior 
    L_posterior = - z_dist.log_prob(z_smooth)
    L_posterior = L_posterior.mean()

    # Bounce loss
    L_alpha_bounce = alpha_bounce_loss(a_seq, alpha_seq, u_seq, mask)
    
    loss = (tcfg.lambda_recon * L_recon +
        tcfg.lambda_innov * L_innov +
        tcfg.lambda_prior * L_prior -
        tcfg.lambda_entropy * L_entropy  -
        tcfg.lambda_posterior * L_posterior + 
        tcfg.lambda_alpha * L_alpha_bounce
    )

    terms = {
        "loss":      loss.item(),
        "recon":     L_recon.item(),
        "innov":     L_innov.item(),
        "prior":     L_prior.item(),
        "entropy":   L_entropy.item(),
        "posterior": L_posterior.item(),
        "bounce":    L_alpha_bounce.item()
    }
    return loss, terms

def vae_compute_loss(ball_seq, x_dist_smooth, a_dist, a_seq, a_pred, tcfg, epoch, mask=None):
    """
    Computes VAE loss:
        F = log p(x|a) - KL(q(a|x) || p(a)) - L_pred

        ball_seq:      [B, T, H, W]          — sequence of ball images
        x_dist_smooth:                       — reconstruction from smoothed latents
        a_dist:        Normal([B, T, dim_a]) — encoder distribution over latent observations
        a_seq:         [B, T, dim_a]         — sampled latent observations
        a_pred:        [B, T, dim_a]         — constant velocity predicted latents
        tcfg:          TrainConfig           — training configuration
        epoch:         int                   — current training epoch (used for KL annealing)
        mask:          [B, T]                — mask for valid timesteps (1=observed, 0=missing)
    """
    device = ball_seq.device
    B, T, H, W = ball_seq.shape

    if mask is None:
        mask = torch.ones(B, T, device=device)

    # log p(x | a) — reconstruction loss
    pos_weight  = torch.tensor(tcfg.pos_weight, device=device)
    logits      = x_dist_smooth.logits                          # [B, T, H, W]
    L_recon = F.binary_cross_entropy_with_logits(
        logits,
        ball_seq,
        pos_weight=pos_weight,
        reduction='none'
    ).sum(dim=(2, 3)).mean()                                    # [B, T]

    # prediction loss
    L_pred = F.mse_loss(a_pred[:, :-1, :], a_seq[:, 1:, :].detach())

    # KL(q(a|x) || N(0,I)) — analytic KL with linear warm-up annealing
    kl = -0.5 * (1 + 2 * a_dist.scale.log()
                     - a_dist.loc.pow(2)
                     - a_dist.scale.pow(2))                     # [B, T, dim_a]
    L_kl = kl.sum(dim=-1).mean()

    # KL(q(a_t|x_t) || q(a_{t-1}|x_{t-1})) — latent smoothness across timesteps
    mu, std = a_dist.loc, a_dist.scale
    kl_trans = -0.5 * (
        1 + (std[:, 1:] / std[:, :-1].clamp(min=1e-6)).log() * 2
        - ((mu[:, 1:] - mu[:, :-1]).pow(2) + std[:, 1:].pow(2))
        / std[:, :-1].clamp(min=1e-6).pow(2)
    )                                                           # [B, T-1, dim_a]
    L_kl_trans = kl_trans.sum(dim=-1).mean()

    loss = (tcfg.lambda_recon           * L_recon +
            tcfg.get_lambda_pred(epoch) * L_pred  +
            tcfg.get_lambda_kl(epoch)   * L_kl    + 
            tcfg.get_lambda_trans       * L_kl_trans
            )

    return {
        'loss':    loss,
        'recon': L_recon.item(),
        'pred':  L_pred.item(),
        'kl':    L_kl.item(),
        'trans': L_kl_trans.item()
    }

def compute_loss(ball_seq, x_dist_smooth,
                a_dist, a_seq, a_smooth, a_pred,
                z_dist, z_smooth, z_pred, 
                R, Q, mask, alpha_seq,
                cfg: VAEConfig, tcfg: TrainConfig, epoch, model_type="kvae", u_seq=None):

    if model_type == "kvae":
        return kvae_compute_loss(ball_seq, x_dist_smooth,
                a_dist, a_seq, a_smooth,
                z_dist, z_smooth, z_pred, 
                R, Q, mask, alpha_seq,
                tcfg, u_seq)
    
    else:
        return vae_compute_loss(ball_seq, x_dist_smooth, a_dist, 
                                a_seq, a_pred, tcfg, epoch, mask)