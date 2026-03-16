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
    plot_latent_space,
)

logger = logging.getLogger("kvae.evaluate")

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="kvae", choices=["kvae", "gru_vae", "cv_vae"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--results_dir",type=str, default=None)
    return parser.parse_args()


def load_model(checkpoint_path, model_name, cfg, sim_cfg, device):
    if model_name == "kvae":
        model = KVAE(cfg, sim_cfg)
    elif model_name == "gru_vae":
        model = GRUVAE(cfg, sim_cfg)
    elif model_name == "cv_vae":
        model = CVVAE(cfg, sim_cfg)
    else: 
        raise ValueError(f"Unknown model: {model_name}")
    model = model.to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Model loaded from: {checkpoint_path}")
    return model


@torch.no_grad()
def predict_multistep(model, ball_seq, obstacle_img, n_steps, device):
    """
    Multi-step free-running prediction for last n_step steps.

    ball_seq: [1, T, H, W]
    obstacle_img: [1, H, W]
    """
    B, T, H, W = ball_seq.shape

    mask = torch.ones(B, T, device=device)
    mask[:, T - n_steps:] = 0.0

    (x_dist_filt, x_dist_pred,
     a_dist, a_seq, a_filt, a_pred,
     z_filt, P_filt, z_pred, P_pred,
    alpha_seq, R, Q) = model(ball_seq, obstacle_img, mask=mask)

    return {
        "a_mu":        to_numpy(a_dist.loc[0]),
        "a_filt":      to_numpy(a_filt[0]),
        "a_pred_free": to_numpy(a_filt[0, T - n_steps:]),
        "x_hat_filt":  to_numpy(x_dist_filt.mean[0]),
        "x_hat_pred":  to_numpy(x_dist_pred.mean[0]) if x_dist_pred is not None else None,
        "alpha_seq":   to_numpy(alpha_seq[0]) if alpha_seq is not None else None,
        "ball_seq":    to_numpy(ball_seq[0]),
    }


@torch.no_grad()
def compute_mse_per_step(model, loader, max_steps, device):
    """
    Compute MSE for 1..max_steps ahead prediction."
    """
    mse_per_step = np.zeros(max_steps)
    counts = np.zeros(max_steps)

    for batch in loader:
        if len(batch) == 2:
            ball_seq, obstacle_img = batch
            u_seq = None
        else:
            ball_seq, obstacle_img, u_seq = batch

        ball_seq     = ball_seq.to(device)
        obstacle_img = obstacle_img.to(device)
        B, T, H, W   = ball_seq.shape

        for n in range(1, max_steps + 1):
            if T - n < 0:
                break

            mask = torch.ones(B, T, device=device)
            mask[:, T - n:] = 0.0

            (x_dist_filt, x_dist_pred,
            a_dist, a_seq, a_filt, a_pred,
            z_filt, P_filt, z_pred, P_pred,
            alpha_seq, R, Q) = model(ball_seq, obstacle_img, mask=mask)

            mse = ((x_dist_filt.mean[:, T - n:] - ball_seq[:, T - n:]) ** 2).mean().item()
            mse_per_step[n - 1] += mse * B
            counts[n - 1]       += B

    return mse_per_step / counts.clip(min=1)


@torch.no_grad()
def compute_metrics(model, loader, device):
    """
    Evaluate all metrics on the test set.
    """
    total = {"recon_mse": 0.0, "pred_mse": 0.0, "latent_smoothness": 0.0}
    count = 0

    for batch in loader:
        if len(batch) == 2:
            ball_seq, obstacle_img = batch
            u_seq = None
        else:
            ball_seq, obstacle_img, u_seq = batch

        ball_seq     = ball_seq.to(device)
        obstacle_img = obstacle_img.to(device)
        B, T, H, W   = ball_seq.shape

        (x_dist_filt, x_dist_pred,
        a_dist, a_seq, a_filt, a_pred,
        z_filt, P_filt, z_pred, P_pred,
        alpha_seq, R, Q) = model(ball_seq, obstacle_img)

        # Reconstruction MSE
        recon_mse = ((x_dist_filt.mean - ball_seq) ** 2).mean().item()

        # Prediction MSE (1 step)
        pred_mse  = ((x_dist_pred.mean[:, :-1] - ball_seq[:, 1:]) ** 2).mean().item() if x_dist_pred is not None else 0.0

        # Latent smoothness
        smoothness = ((a_filt[:, 1:] - a_filt[:, :-1]) ** 2).mean().item()

        total["recon_mse"] += recon_mse * B
        total["pred_mse"] += pred_mse  * B
        total["latent_smoothness"] += smoothness * B
        count += B

    return {k: v / count for k, v in total.items()}


def evaluate(checkpoint_path, model_name, results_dir, cfg, sim_cfg, tcfg, device, max_pred_steps=10, n_samples=3):
    """
    Main evaluation pipeline:
        Load pre-trained model checkpoint.
        Compute performance metrics on the test set.
        Analyze MSE across a multi-step prediction horizon.
        Generate visual reconstructions for n_samples.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Model
    model = load_model(checkpoint_path, model_name, cfg, sim_cfg, device)

    # Dataset
    test_dataset = BallDataset(sim_cfg, tcfg, split="test")
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_metrics(model, test_loader, device)
    logger.info("Test Set Performance:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.6f}")

    np.save(os.path.join(results_dir, "metrics.npy"), metrics)

    # Multi-step Error Analysis
    logger.info(f"Analyzing MSE for 1..{max_pred_steps} prediction steps ahead...")
    mse_per_step = compute_mse_per_step(model, test_loader, max_pred_steps, device)
    np.save(os.path.join(results_dir, "mse_per_step.npy"), mse_per_step)

    plot_prediction_mse(mse_per_step_kvae = mse_per_step, save_path = os.path.join(results_dir, "plots/mse_per_step.png"))

    logger.info(f"Generating visualizations for {n_samples} samples...")
    for i in range(n_samples):
        ball_seq, obstacle_img = test_dataset[i]
        ball_seq     = ball_seq.unsqueeze(0).to(device)
        obstacle_img = obstacle_img.unsqueeze(0).to(device)

        out = predict_multistep(model, ball_seq, obstacle_img, max_pred_steps, device)

        sample_dir = os.path.join(results_dir, f"plots/sample_{i}")

        plot_trajectories(a_mu=out["a_mu"],a_filt=out["a_filt"], a_pred_free=out["a_pred_free"], save_path=os.path.join(sample_dir, "trajectories.png"))

        plot_reconstruction_grid(ball_seq=out["ball_seq"], x_hat_filt=out["x_hat_filt"], x_hat_pred=out["x_hat_pred"], save_path=os.path.join(sample_dir, "reconstruction.png"))

        if model_name == "kvae": plot_alpha(alpha_seq = out["alpha_seq"], save_path = os.path.join(sample_dir, "alpha.png"))

        plot_latent_space(a_mu=out["a_mu"], a_filt=out["a_filt"], save_path=os.path.join(sample_dir, "latent_space.png"))

        logger.info(f"Sample {i} saved in {sample_dir}")

    logger.info(f"Evaluation complete. Results saved to: {results_dir}")
    return metrics, mse_per_step


if __name__ == "__main__":
    args = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = VAEConfig()
    sim_cfg = SimulationConfig()
    tcfg = TrainConfig()

    checkpoint = args.checkpoint or f"checkpoints/{args.model}/best.pt"
    results = args.results_dir or f"results/{args.model}"
    
    logger = setup_logger(log_dir=f"logs/{args.model}", log_file="evaluate.log")

    evaluate(
        checkpoint_path = checkpoint,
        model_name      = args.model,
        results_dir     = "results/",
        cfg             = cfg,
        sim_cfg         = sim_cfg,
        tcfg            = tcfg,
        device          = device,
        max_pred_steps  = tcfg.free_running_steps,
        n_samples       = 5,
    )