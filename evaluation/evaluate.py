import torch
import numpy as np
import argparse
import os
import sys
import logging
sys.path.append("..")
from torch.utils.data import DataLoader
from config.vae_config import VAEConfig
from config.simulation_config import SimulationConfig
from config.train_config import TrainConfig
from training.train import setup_logger
from models.kvae import KVAE
from models.cv_vae import CVVAE
from models.gru_vae import GRUVAE
from dataset.dataset import BallDataset
from utils.visualize import (
    plot_trajectories,
    plot_reconstruction_grid,
    plot_alpha,
    plot_prediction_mse,
    plot_uncertainty,
    plot_imputation
)
logger = logging.getLogger("kvae.evaluate")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,  default="kvae",
                        choices=["kvae", "gru_vae", "cv_vae"])
    parser.add_argument("--checkpoint",  type=str,  default=None)
    parser.add_argument("--results_dir", type=str,  default=None)
    parser.add_argument("--smoother",    action="store_true",
                        help="Use RTS smoother instead of Kalman filter")
    return parser.parse_args()

def load_model(checkpoint_path, model_name, cfg, sim_cfg, tcfg, device):
    if model_name == "kvae":
        model = KVAE(cfg, sim_cfg, tcfg)
    elif model_name == "gru_vae":
        model = GRUVAE(cfg, sim_cfg, tcfg)
    elif model_name == "cv_vae":
        model = CVVAE(cfg, sim_cfg, tcfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = model.to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Model loaded from: {checkpoint_path}")
    return model


@torch.no_grad()
def run_model(model, ball_seq, obstacle_img, u_seq, mask, smoother, device):
    """
    Unified model forward pass returning a flat dict of numpy arrays.
    """
    out = model(ball_seq, obstacle_img, u_seq=u_seq, mask=mask,
                smoother=smoother)
 
    (x_dist, a_dist, a_seq, a_smooth, a_pred,
     z_dist, z_smooth, z_pred,
     R, Q, alpha_seq) = out
 
    result = {
        "x_hat":     to_numpy(x_dist.mean[0]),
        "a_mu":      to_numpy(a_dist.loc[0]),
        "a_smooth":  to_numpy(a_smooth[0]),
        "a_pred":    to_numpy(a_pred[0]),
        "z_smooth":  to_numpy(z_smooth[0]) if z_smooth is not None else None,
        "alpha_seq": to_numpy(alpha_seq[0]) if alpha_seq is not None else None,
        "ball_seq":  to_numpy(ball_seq[0]),
    }
 
    if z_dist is not None and z_smooth is not None:
        P = to_numpy(z_dist.covariance_matrix[0])   # [T, dim_z, dim_z]
        result["P_diag"] = np.diagonal(P, axis1=-2, axis2=-1)   # [T, dim_z]
 
    return result

@torch.no_grad()
def compute_mse_per_step(model, loader, max_steps, smoother, device):
    """
    Compute MSE for 1..max_steps ahead prediction,
    split into episodes with and without gravity.
    """
    mse_no_grav  = np.zeros(max_steps)
    mse_grav     = np.zeros(max_steps)
    cnt_no_grav  = np.zeros(max_steps)
    cnt_grav     = np.zeros(max_steps)
 
    for batch in loader:
        ball_seq, obstacle_img, *rest = batch
        u_seq = rest[0].to(device) if rest and rest[0] is not None else None
 
        ball_seq     = ball_seq.to(device)
        obstacle_img = obstacle_img.to(device)
        if u_seq is not None:
            u_seq = u_seq.to(device)
 
        B, T, H, W = ball_seq.shape
 
        # determine which episodes have gravity
        has_gravity = (u_seq.abs().sum(dim=[1, 2]) > 0).cpu().numpy() \
                      if u_seq is not None else np.zeros(B, dtype=bool)
 
        for n in range(1, max_steps + 1):
            start = T // 2 - n // 2
            end   = start + n

            mask = torch.ones(B, T, device=device)
            mask[:, start:end] = 0.0
 
            (x_dist, *_) = model(ball_seq, obstacle_img, u_seq=u_seq,
                                 mask=mask, smoother=smoother)
 
            mse = ((x_dist.mean[:, start:end] - ball_seq[:, start:end]) ** 2) \
                    .mean(dim=(2, 3)).mean(dim=1).cpu().numpy()  # [B]
 
            mse_no_grav[n-1]  += mse[~has_gravity].sum()
            cnt_no_grav[n-1]  += (~has_gravity).sum()
            mse_grav[n-1]     += mse[has_gravity].sum()
            cnt_grav[n-1]     += has_gravity.sum()
 
    return (mse_no_grav / cnt_no_grav.clip(min=1),
            mse_grav    / cnt_grav.clip(min=1))


@torch.no_grad()
def compute_metrics(model, loader, smoother, device):
    """
    Compute reconstruction MSE and latent smoothness on the full test set.
    """
    total = {"recon_mse": 0.0, "latent_smoothness": 0.0}
    count = 0
 
    for batch in loader:
        ball_seq, obstacle_img, *rest = batch
        u_seq = rest[0].to(device) if rest and rest[0] is not None else None
 
        ball_seq     = ball_seq.to(device)
        obstacle_img = obstacle_img.to(device)
        if u_seq is not None:
            u_seq = u_seq.to(device)
 
        B = ball_seq.shape[0]
 
        (x_dist, a_dist, a_seq, a_smooth, *_) = \
            model(ball_seq, obstacle_img, u_seq=u_seq, mask=None,
                  smoother=smoother)
        
        total["recon_mse"] += ((x_dist.mean - ball_seq) ** 2).mean().item() * B
        total["latent_smoothness"] += ((a_smooth[:, 1:] - a_smooth[:, :-1]) ** 2) \
                                        .mean().item() * B
        count += B
 
    return {k: v / count for k, v in total.items()}


def evaluate(checkpoint_path, model_name, results_dir,
             cfg, sim_cfg, tcfg, device,
             smoother=False, max_pred_steps=10, n_samples=5):
    """
    Main evaluation pipeline.
    """
    os.makedirs(results_dir, exist_ok=True)
 
    model = load_model(checkpoint_path, model_name, cfg, sim_cfg, tcfg, device)
 
    test_dataset = BallDataset(sim_cfg, cfg, tcfg, split="test")
    test_loader  = DataLoader(test_dataset, batch_size=32,
                              shuffle=False, num_workers=4)
 
    # Metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_metrics(model, test_loader, smoother, device)
    logger.info("Test Set Performance:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.6f}")
    np.save(os.path.join(results_dir, "metrics.npy"), metrics)
 
    # Multi-step MSE 
    logger.info(f"Computing MSE for 1..{max_pred_steps} steps ahead...")
    mse_no_grav, mse_grav = compute_mse_per_step(
        model, test_loader, max_pred_steps, smoother, device)
    np.save(os.path.join(results_dir, "mse_no_grav.npy"), mse_no_grav)
    np.save(os.path.join(results_dir, "mse_grav.npy"),    mse_grav)
 
    plot_prediction_mse(
        mse_kvae      = mse_no_grav,
        mse_kvae_grav = mse_grav,
        save_path     = os.path.join(results_dir, "plots/mse_per_step.png")
    )
 
    # Per-sample visualizations 
    logger.info(f"Generating visualizations for {n_samples} samples...")
 
    for i in range(n_samples):
        ball_seq, obstacle_img, *rest = test_dataset[i]
        u_seq = rest[0] if rest and rest[0] is not None else None
 
        ball_seq     = ball_seq.unsqueeze(0).to(device)
        obstacle_img = obstacle_img.unsqueeze(0).to(device)
        if u_seq is not None:
            u_seq = u_seq.unsqueeze(0).to(device)
 
        sample_dir = os.path.join(results_dir, f"plots/sample_{i}")
 
        # standard forward (no masking)
        out = run_model(model, ball_seq, obstacle_img, u_seq,
                        mask=None, smoother=smoother, device=device)
 
        plot_trajectories(
            a_mu      = out["a_mu"],
            a_smooth  = out["a_smooth"],
            smoother  = smoother,
            save_path = os.path.join(sample_dir, "trajectories.png")
        )
 
        plot_reconstruction_grid(
            ball_seq   = out["ball_seq"],
            x_hat_filt = out["x_hat"],
            save_path  = os.path.join(sample_dir, "reconstruction.png")
        )
 
        if model_name == "kvae":
            plot_alpha(
                alpha_seq = out["alpha_seq"],
                ball_seq  = out["ball_seq"],
                save_path = os.path.join(sample_dir, "alpha.png")
            )
 
        # uncertainty / masking — mask last max_pred_steps frames
        mask = torch.ones(1, sim_cfg.T, device=device)
        mask[:, sim_cfg.T - max_pred_steps:] = 0.0
        mask_np = mask[0].cpu().numpy()
 
        out_masked = run_model(model, ball_seq, obstacle_img, u_seq,
                               mask=mask, smoother=smoother, device=device)
 
        if "P_diag" in out_masked:
            plot_uncertainty(
                P_diag    = out_masked["P_diag"],
                mask      = mask_np,
                smoother  = smoother,
                save_path = os.path.join(sample_dir, "uncertainty.png")
            )

        # imputation — only meaningful for KVAE
        if model_name == "kvae":
            out_filt = run_model(model, ball_seq, obstacle_img, u_seq,
                                 mask=mask, smoother=False, device=device)
            out_smth = run_model(model, ball_seq, obstacle_img, u_seq,
                                 mask=mask, smoother=True,  device=device)
            plot_imputation(
                a_mu      = out_filt["a_mu"],
                a_filt    = out_filt["a_smooth"],
                a_smooth  = out_smth["a_smooth"],
                mask      = mask_np,
                save_path = os.path.join(sample_dir, "imputation.png")
            )
 
        logger.info(f"  Sample {i} saved in {sample_dir}")

    logger.info(f"Evaluation complete. Results in: {results_dir}")
    return metrics, mse_no_grav, mse_grav


if __name__ == "__main__":
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    cfg     = VAEConfig()
    sim_cfg = SimulationConfig()
    tcfg    = TrainConfig()
 
    checkpoint  = args.checkpoint or f"checkpoints/{args.model}/best_{args.model}.pt"
    results_dir = args.results_dir or f"results/{args.model}"
 
    logger = setup_logger(log_dir=f"logs/{args.model}", log_file="evaluate.log")
 
    evaluate(
        checkpoint_path = checkpoint,
        model_name      = args.model,
        results_dir     = results_dir,
        cfg             = cfg,
        sim_cfg         = sim_cfg,
        tcfg            = tcfg,
        device          = device,
        smoother        = args.smoother,
        max_pred_steps  = int(sim_cfg.T * tcfg.max_mask_ratio),
        n_samples       = 5,
    )