import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
sys.path.append("..")


def plot_trajectories(a_mu, a_filt, a_pred_free=None, save_path=None):
    """
    Plot trajectory in latent space.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(a_mu[:, 0],   a_mu[:, 1],   color="steelblue",  linewidth=1.2, alpha=0.7, label="Encoder $a_\\mu$")
    ax.plot(a_filt[:, 0], a_filt[:, 1], color="seagreen",   linewidth=2.0, label="Kalman filtered $\\hat{a}$")

    ax.scatter(*a_filt[0],  color="seagreen", s=80, zorder=5, marker="o", label="Start")
    ax.scatter(*a_filt[-1], color="seagreen", s=80, zorder=5, marker="X", label="End")

    if a_pred_free is not None:
        ax.plot(a_pred_free[:, 0], a_pred_free[:, 1], color="tomato", linewidth=2.0, linestyle="--", label="Free-running prediction")
        ax.scatter(*a_pred_free[-1], color="tomato", s=80, zorder=5, marker="X")

    ax.set_xlabel("$a_1$", fontsize=12)
    ax.set_ylabel("$a_2$", fontsize=12)
    ax.set_title("Latent Space Trajectories", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_reconstruction_grid(ball_seq, x_hat_filt, x_hat_pred, frame_indices=None, save_path=None):
    """
    Reconstuction grid plot: original | filtered | predicted
    """
    T = ball_seq.shape[0]
    if frame_indices is None: frame_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]

    n_frames = len(frame_indices)
    row_labels = ["Original", "Filtered", "Predicted"]
    rows = [ball_seq, x_hat_filt, x_hat_pred]

    fig, axes = plt.subplots(3, n_frames, figsize=(n_frames * 2.5, 7))

    for row_idx, (label, data) in enumerate(zip(row_labels, rows)):
        for col_idx, t in enumerate(frame_indices):
            ax = axes[row_idx, col_idx]
            ax.imshow(data[t].squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=11, rotation=90, labelpad=40, va="center")
            if row_idx == 0:
                ax.set_title(f"t={t}", fontsize=10)

    plt.suptitle("Reconstruction Grid", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_alpha(alpha_seq, save_path=None):
    """
    Stacked area plot of alpha weights over time.
    """
    T, K = alpha_seq.shape
    t    = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ["steelblue", "seagreen", "tomato", "mediumpurple", "orange"]
    alphas_np = alpha_seq if isinstance(alpha_seq, np.ndarray) else alpha_seq.numpy()

    ax.stackplot(t, alphas_np.T,
                 labels=[f"$\\alpha_{i+1}$" for i in range(K)],
                 colors=colors[:K], alpha=0.8)

    ax.set_xlabel("Time step $k$", fontsize=12)
    ax.set_ylabel("Mixing weight", fontsize=12)
    ax.set_title("Matrix Mixing Weights $\\alpha_k$ over Time", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_prediction_mse(mse_per_step_kvae, mse_per_step_vae=None, save_path=None):
    """
    MSE vs. Prediction Horizon.
    """
    steps = np.arange(1, len(mse_per_step_kvae) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(steps, mse_per_step_kvae, color="seagreen", linewidth=2.0,
            marker="o", markersize=5, label="VAE + Kalman (ours)")

    if mse_per_step_vae is not None:
        ax.plot(steps, mse_per_step_vae, color="steelblue", linewidth=2.0,
                marker="s", markersize=5, linestyle="--", label="VAE baseline")

    ax.set_xlabel("Prediction horizon (steps)", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("Prediction MSE vs Horizon", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_latent_space(a_mu, a_filt, save_path=None):
    """
    Scatter plot of latent space.
    """
    T = a_mu.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title, color in zip( axes, [a_mu, a_filt], ["Encoder output $a_\\mu$ (noisy)", "Kalman filtered $\\hat{a}$ (smooth)"], ["steelblue", "seagreen"]):
        sc = ax.scatter(data[:, 0], data[:, 1], c=t, cmap="viridis", s=20, alpha=0.8)
        ax.plot(data[:, 0], data[:, 1], color=color, linewidth=0.8, alpha=0.5)
        ax.set_xlabel("$a_1$", fontsize=12)
        ax.set_ylabel("$a_2$", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Time step")

    plt.suptitle("Latent Space: Encoder vs Kalman", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()