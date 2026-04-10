import torch
import torch.nn.functional as F
import torch.distributions as D
import math
from config.train_config import TrainConfig
from config.vae_config import VAEConfig


def innovation_loss(a_filt, a_seq, R, mask):
    """
    Stable innovation negative log-likelihood.

    a_filt : [B, T, dim_a]        — predicted observation 
    a_seq  : [B, T, dim_a]        — true observation
    R : [B, dim_a, dim_a] — innovation covariance
    """

    B, T, dim_a = a_filt.shape
    device = a_filt.device

    mu = a_filt        # [B, T, dim_a]
    x  = a_seq         # [B, T, dim_a]
    S = R.unsqueeze(0).expand(B * T, dim_a, dim_a)    # [B*T, dim_a, dim_a]

    mu = mu.reshape(-1, dim_a)
    x  = x.reshape(-1, dim_a)

    mask_flat = mask.reshape(-1)

    #S = S + 1e-4 * torch.eye(dim_a, device=S.device).unsqueeze(0)
    L = torch.linalg.cholesky(S)

    dist = D.MultivariateNormal(mu, scale_tril=L)
    log_prob = dist.log_prob(x)      # [B*T]
    log_prob = log_prob * mask_flat

    return - log_prob.sum() / mask_flat.sum()


def compute_loss(ball_seq, x_dist_smooth, x_dist_pred,
                a_dist, a_seq, a_smooth, z_dist_full,
                z_dist, z_smooth, P_smooth, z_pred, P_pred, 
                R, Q, mask, alpha_seq,
                cfg: VAEConfig, tcfg: TrainConfig, epoch, phase=1, u_seq=None):
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
        S_pred:      [B, T, dim_a, dim_a]
        Q:           [dim_z, dim_z]          — transition noise covariance
    """
    B, T, dim_z = z_smooth.shape if z_smooth is not None else (*a_seq.shape[:2], 0)
    device = ball_seq.device

    if mask is None:
        mask = torch.ones(ball_seq.shape[:2], device=ball_seq.device)
    cur_mask = torch.ones(ball_seq.shape[:2], device=ball_seq.device)

    if phase == 0:
        # log p(x | a) — reconstruction
        logits = x_dist_smooth.logits
        pos_weight = torch.tensor(tcfg.pos_weight, device=device)
        L_recon = F.binary_cross_entropy_with_logits(
            logits,
            ball_seq,
            pos_weight=3,
            reduction='none'
        ).sum(dim=(2, 3)).mean()

        # KL(q(a|x) || N(0,I)) — encoder regularization
        mu  = a_dist.loc
        var = a_dist.scale ** 2
        L_regularization = 0.5 * (mu**2 + var - var.log() - 1).sum(-1).mean()

        loss = (0.3*tcfg.lambda_recon * L_recon
              + tcfg.get_lambda_kl(epoch) * L_regularization)

        terms = {
            "loss":    loss.item(),
        }
        
        return loss, terms
    
    if phase == 1 or phase == 2:
        # log p(x | a) — reconstruction
        logits = x_dist_smooth.logits
        pos_weight = torch.tensor(tcfg.pos_weight, device=device)
        L_recon = F.binary_cross_entropy_with_logits(
            logits,
            ball_seq,
            pos_weight=torch.tensor(3, device=device),
            reduction='none'
        ).sum(dim=(2, 3)).mean()

        #L_recon_pred = F.binary_cross_entropy_with_logits(
        #    x_dist_pred.logits[:, :-1, :, :],
        #    ball_seq[:, 1:, :, :],
        #    pos_weight=pos_weight,
        #    reduction='none'
        #).sum(dim=(2, 3))
        #L_recon_pred = (L_recon_pred * mask[:, 1:]).sum() / mask[:, 1:].sum()

        # log p(a | z) — innovation
        L_innov = innovation_loss(a_smooth, a_seq, R, cur_mask)

        # log p(z | u) — state prior
        L_prior = - D.MultivariateNormal(
            loc=z_pred[:, :-1, :].reshape(-1, dim_z),
            scale_tril=torch.linalg.cholesky(Q)
        ).log_prob(z_smooth[:, 1:, :].reshape(-1, dim_z))
        mask_z = cur_mask[:, 1:].reshape(-1)
        L_prior = (L_prior * mask_z).sum() / mask_z.sum() 
        # log q(a | x) — encoder entropy
        L_entropy = - a_dist.log_prob(a_seq).sum(-1)
        L_entropy = (L_entropy * cur_mask).sum() / cur_mask.sum()

        # log p(z | a, u) — Kalman posterior 
        L_posterior = -z_dist.log_prob(z_smooth)
        L_posterior = (L_posterior * cur_mask).sum() / cur_mask.sum()

        mask_da = mask[:, 1:] * mask[:, :-1]      
        mask_bounce = mask_da[:, :-1] * mask_da[:, 1:] 

        has_gravity = (u_seq.abs().sum(dim=[1, 2]) > 1e-6)  # [B]
        no_gravity_mask = (~has_gravity).float()            # [B]

        da = a_seq[:, 1:, :] - a_seq[:, :-1, :]
        da_prev = da[:, :-1, :]   # [B, T-2, 2]
        da_next = da[:, 1:, :]    # [B, T-2, 2]
        cos_sim = F.cosine_similarity(da_prev, da_next, dim=-1)  # [B, T-2]
        bounce_signal = (1 - cos_sim).clamp(0, 1)
        bounce_signal = bounce_signal
        bounce_signal = bounce_signal / (bounce_signal.sum() + 1e-8)  

        alpha_cos = F.cosine_similarity(
            alpha_seq[:, 2:, :],
            alpha_seq[:, 1:-1, :],
            dim=-1
        )
        loss_per_example = (bounce_signal.detach() * (alpha_cos**2) * mask_bounce).sum(dim=1)
        L_alpha_bounce = (loss_per_example * no_gravity_mask).sum()

        num_no_gravity = no_gravity_mask.sum()
        if num_no_gravity > 0:
            L_alpha_bounce = L_alpha_bounce / num_no_gravity
        else:
            L_alpha_bounce = L_alpha_bounce * 0.0
        
        if phase == 1:
            loss = (0.1 * tcfg.lambda_recon * L_recon +
                #0.05 * tcfg.lambda_recon * L_recon_pred +
                (1 + 0*tcfg.lambda_innov) * L_innov +
                (1 + 0*tcfg.lambda_prior) * L_prior -
                #(1.0 + 0*tcfg.lambda_entropy) * L_entropy  -
                (1.0 + 0*tcfg.lambda_posterior) * L_posterior + 
                0.3 * L_alpha_bounce
            )
        else:
            loss =(
                (2 + 0*tcfg.lambda_innov) * L_innov +
                (1 + 0*tcfg.lambda_prior) * L_prior -
                (0.3 + 0*tcfg.lambda_posterior) * L_posterior 
                + 0.15 * L_alpha_bounce
            )

        terms = {
            "loss":      loss.item(),
            "recon":     L_recon.item(),
            "innov":     L_innov.item(),
            "prior":     L_prior.item(),
            "entropy":   L_entropy.item(),
            "posterior": L_posterior.item(),
        }
        return loss, terms


        



    # CV_VAE / GRU_VAE
    return None, None  # skip prediction loss for now
    # KL(q(a|x) || N(0,I)) — encoder regularization
    mu  = a_dist.loc
    var = a_dist.scale ** 2
    L_regularization = 0.5 * (mu**2 + var - var.log() - 1).sum(-1).mean()

    # log p(a_{t+1} | z_pred_t) — prediction
    a_dist_next = D.Normal(a_dist.loc[:, 1:, :], a_dist.scale[:, 1:, :])
    L_prediction = -a_dist_next.log_prob(a_pred[:, :-1, :]).sum(-1).mean()

    # log q(a | x) — encoder entropy
    L_entropy =  a_dist.log_prob(a_seq).sum(-1).mean()

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