import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
sys.path.append("..")


def plot_trajectories(a_mu, a_smooth, smoother=False, save_path=None):
    """
    Plot trajectory in latent space.
 
    a_mu:        [T, dim_a] — encoder output (noisy)
    a_smooth:    [T, dim_a] — filtered or smoothed trajectory
    smoother:    bool       — whether a_smooth is RTS-smoothed or Kalman-filtered
    """
    fig, ax = plt.subplots(figsize=(7, 7))
 
    smooth_label = "RTS smoothed $\\hat{a}^s$" if smoother else "Kalman filtered $\\hat{a}$"
 
    ax.plot(a_mu[:, 0], a_mu[:, 1],
            color="gray", linestyle="--", linewidth=1.5, alpha=0.6,
            label="Encoder $a_\\mu$", zorder=1)
 
    ax.plot(a_smooth[:, 0], a_smooth[:, 1],
            color="darkorange", linewidth=2.5,
            label=smooth_label, zorder=2)
 
    step = max(1, len(a_smooth) // 8)
    for i in range(0, len(a_smooth) - 1, step):
        ax.annotate("", xy=(a_smooth[i+1, 0], a_smooth[i+1, 1]),
                    xytext=(a_smooth[i, 0], a_smooth[i, 1]),
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5))
 
    ax.scatter(*a_smooth[0],  color="forestgreen", s=150, zorder=5,
               edgecolors="black", label="Start")
    ax.scatter(*a_smooth[-1], color="crimson", s=150, zorder=5,
               marker="X", edgecolors="black", label="End")
 
 
    ax.set_xlabel("$a_1$", fontsize=12)
    ax.set_ylabel("$a_2$", fontsize=12)
    ax.set_title("Latent Space Trajectories", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.axis("equal")
 
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_reconstruction_grid(ball_seq, x_hat_filt, x_hat_pred=None, frame_indices=None, save_path=None):
    """
    Reconstruction grid: original | filtered | predicted (optional)
 
    ball_seq:    [T, H, W] — ground truth frames
    x_hat_filt:  [T, H, W] — filtered reconstructions
    x_hat_pred:  [T, H, W] — predicted reconstructions (optional)
    """
    T = ball_seq.shape[0]
    if frame_indices is None:
        frame_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
 
    rows = [("Original", ball_seq), ("Filtered", x_hat_filt)]
    if x_hat_pred is not None:
        rows.append(("Predicted", x_hat_pred))
 
    n_frames = len(frame_indices)
    n_rows   = len(rows)
 
    fig, axes = plt.subplots(n_rows, n_frames, figsize=(n_frames * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
 
    for row_idx, (label, data) in enumerate(rows):
        for col_idx, t in enumerate(frame_indices):
            ax = axes[row_idx, col_idx]
            ax.imshow(data[t].squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=11, rotation=90, labelpad=40, va="center")
            if row_idx == 0:
                ax.set_title(f"$t={t}$", fontsize=10)
 
    plt.suptitle("Reconstruction Grid", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save_path)

def plot_alpha(alpha_seq, ball_seq=None, save_path=None):
    """
    Stacked area plot of alpha weights over time, optionally with
    a strip of ball frames along the top for reference.
 
    alpha_seq: [T, K] — mixing weights
    ball_seq:  [T, H, W] — original frames (optional)
    """
    T, K = alpha_seq.shape
    t    = np.arange(T)
 
    has_frames = ball_seq is not None
    fig = plt.figure(figsize=(12, 5 if not has_frames else 6.5))
 
    if has_frames:
        gs     = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)
        ax_img = fig.add_subplot(gs[0])
        ax     = fig.add_subplot(gs[1])
 
        n_show = min(T, 20)
        step   = max(1, T // n_show)
        shown  = list(range(0, T, step))[:n_show]
 
        img_strip = np.concatenate([ball_seq[i].squeeze() for i in shown], axis=1)
        ax_img.imshow(img_strip, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax_img.axis("off")
        ax_img.set_title("Ball frames (reference)", fontsize=10, loc="left")
    else:
        ax = fig.add_subplot(111)
 
    colors   = ["steelblue", "seagreen", "tomato", "mediumpurple", "orange"]
    alphas_np = alpha_seq if isinstance(alpha_seq, np.ndarray) else alpha_seq.numpy()
 
    ax.stackplot(t, alphas_np.T,
                 labels=[f"$\\alpha^{{({i+1})}}$" for i in range(K)],
                 colors=colors[:K], alpha=0.85)
 
    ax.set_xlabel("Time step $t$", fontsize=12)
    ax.set_ylabel("Mixing weight", fontsize=12)
    ax.set_title("Dynamics Mixing Weights $\\alpha_t$ over Time", fontsize=13)
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

def plot_alpha_space(model, grid_positions, obstacle_img, u_seq, device, save_path=None):
    """
    Visualize mixing weights alpha^(k) as a heatmap over the latent space,
    overlaid with encoded observations from test sequences.
 
    model:          KVAE model (eval mode)
    grid_positions: [N, H, W] — synthetic frames covering all ball positions
    obstacle_img:   [1, H, W] — static obstacle image
    u_seq:          [1, T, dim_u] — control input (or None)
    device:         torch.device
    """
    model.eval()
    with torch.no_grad():
        frames = torch.tensor(grid_positions, dtype=torch.float32).unsqueeze(0).to(device)
        obs    = obstacle_img.unsqueeze(0).to(device)
 
        a_dist = model.encoder(frames)
        a_mu   = a_dist.loc[0].cpu().numpy()   # [N, dim_a]
 
    K = model.cfg.num_matrices
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 5))
    if K == 1:
        axes = [axes]
 
    colors = ["Blues", "Greens", "Reds", "Purples", "Oranges"]
 
    for k, ax in enumerate(axes):
        with torch.no_grad():
            alpha_seq = model.get_alpha_grid(frames, obs, u_seq, device)
            alpha_k   = alpha_seq[:, k].cpu().numpy()   # [N]
 
        sc = ax.scatter(a_mu[:, 0], a_mu[:, 1],
                        c=alpha_k, cmap=colors[k % len(colors)],
                        s=15, alpha=0.85, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f"$\\alpha^{{({k+1})}}$", fontsize=13)
        ax.set_xlabel("$a_1$", fontsize=11)
        ax.set_ylabel("$a_2$", fontsize=11)
        ax.grid(True, alpha=0.3)
 
    plt.suptitle("Mixing Weight Intensity in Latent Space", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, save_path)

def plot_latent_space(a_mu, a_smooth, smoother=False, save_path=None):
    """
    Side-by-side scatter of encoder output vs filtered/smoothed trajectory,
    coloured by time step.
 
    a_mu:     [T, dim_a] — encoder output
    a_smooth: [T, dim_a] — filtered or smoothed trajectory
    smoother: bool       — label accordingly
    """
    T      = a_mu.shape[0]
    t      = np.arange(T)
    label2 = "RTS smoothed $\\hat{a}^s$" if smoother else "Kalman filtered $\\hat{a}$"
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    for ax, data, title, color in zip(
        axes,
        [a_mu, a_smooth],
        [f"Encoder output $a_\\mu$ (raw)", label2],
        ["steelblue", "darkorange"]
    ):
        sc = ax.scatter(data[:, 0], data[:, 1],
                        c=t, cmap="viridis", s=20, alpha=0.85)
        ax.plot(data[:, 0], data[:, 1],
                color=color, linewidth=0.8, alpha=0.5)
        ax.set_xlabel("$a_1$", fontsize=12)
        ax.set_ylabel("$a_2$", fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label="Time step")
 
    plt.suptitle("Latent Space: Encoder vs Kalman", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, save_path)

def plot_uncertainty(P_diag, mask=None, smoother=False, save_path=None):
    """
    Plot diagonal of covariance matrix P_t over time.
 
    P_diag:   [T, dim_z] — diagonal of P_t at each timestep
    mask:     [T]        — binary mask (0 = masked timestep)
    smoother: bool       — label accordingly
    """
    T, dim_z = P_diag.shape
    t        = np.arange(T)
    label    = "RTS $\\mathrm{diag}(P^s_t)$" if smoother else "Filter $\\mathrm{diag}(P_t)$"
 
    fig, ax = plt.subplots(figsize=(10, 4))
 
    if mask is not None:
        masked = np.where(mask == 0)[0]
        if len(masked):
            ax.axvspan(masked[0], masked[-1], color="lightgray", alpha=0.5,
                       label="Masked steps")
 
    colors = ["steelblue", "seagreen", "tomato", "mediumpurple"]
    for d in range(dim_z):
        ax.plot(t, P_diag[:, d],
                color=colors[d % len(colors)], linewidth=1.8,
                label=f"$P_{{t,{d+1}{d+1}}}$")
 
    ax.set_xlabel("Time step $t$", fontsize=12)
    ax.set_ylabel("Variance", fontsize=12)
    ax.set_title(f"Posterior Covariance over Time — {label}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    _save_or_show(fig, save_path)
 
 
def plot_prediction_mse(mse_kvae, mse_gru=None, mse_cv=None,
                        mse_kvae_grav=None, mse_gru_grav=None, mse_cv_grav=None,
                        save_path=None):
    """
    MSE vs prediction horizon for all models, optionally split by gravity.
 
    mse_*:      [max_steps] arrays — one per model / condition
    """
    steps = np.arange(1, len(mse_kvae) + 1)
    has_grav = any(x is not None for x in [mse_kvae_grav, mse_gru_grav, mse_cv_grav])
 
    fig, axes = plt.subplots(1, 2 if has_grav else 1,
                             figsize=(14 if has_grav else 8, 5),
                             sharey=True)
    if not has_grav:
        axes = [axes]
 
    titles = ["Without gravity", "With gravity"]
    groups = [
        (mse_kvae,      mse_gru,      mse_cv),
        (mse_kvae_grav, mse_gru_grav, mse_cv_grav),
    ]
 
    for ax, title, (mk, mg, mc) in zip(axes, titles, groups):
        if mk is not None:
            ax.plot(steps, mk, color="seagreen", linewidth=2.0,
                    marker="o", markersize=5, label="KVAE (ours)")
        if mg is not None:
            ax.plot(steps, mg, color="steelblue", linewidth=2.0,
                    marker="s", markersize=5, linestyle="--", label="GRU-VAE")
        if mc is not None:
            ax.plot(steps, mc, color="tomato", linewidth=2.0,
                    marker="^", markersize=5, linestyle=":", label="CV-VAE")
 
        ax.set_xlabel("Prediction horizon $n$ (steps)", fontsize=12)
        ax.set_ylabel("MSE", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
 
    plt.suptitle("Prediction MSE vs Horizon", fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, save_path)
 
 
def plot_imputation(a_mu, a_filt, a_smooth, mask=None, save_path=None):
    """
    Compare filter vs smoother trajectories during masked steps.
 
    a_mu:     [T, dim_a] — encoder output (ground truth reference)
    a_filt:   [T, dim_a] — Kalman filter trajectory
    a_smooth: [T, dim_a] — RTS smoother trajectory
    mask:     [T]        — binary mask (0 = masked)
    """
    fig, ax = plt.subplots(figsize=(7, 7))
 
    ax.plot(a_mu[:, 0], a_mu[:, 1],
            color="black", linewidth=1.5, alpha=0.7,
            label="Encoder $a_\\mu$ (reference)", zorder=1)
    ax.plot(a_filt[:, 0], a_filt[:, 1],
            color="steelblue", linewidth=2.0, linestyle="--",
            label="Kalman filter", zorder=2)
    ax.plot(a_smooth[:, 0], a_smooth[:, 1],
            color="seagreen", linewidth=2.0,
            label="RTS smoother", zorder=3)
 
    if mask is not None:
        masked = np.where(mask == 0)[0]
        if len(masked):
            ax.scatter(a_filt[masked, 0], a_filt[masked, 1],
                       color="steelblue", s=40, alpha=0.5, zorder=4)
            ax.scatter(a_smooth[masked, 0], a_smooth[masked, 1],
                       color="seagreen", s=40, alpha=0.5, zorder=4)
 
    ax.scatter(*a_mu[0],  color="forestgreen", s=150, zorder=5,
               edgecolors="black", label="Start")
    ax.scatter(*a_mu[-1], color="crimson", s=150, zorder=5,
               marker="X", edgecolors="black", label="End")
 
    ax.set_xlabel("$a_1$", fontsize=12)
    ax.set_ylabel("$a_2$", fontsize=12)
    ax.set_title("Imputation: Filter vs RTS Smoother", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.axis("equal")
 
    plt.tight_layout()
    _save_or_show(fig, save_path)
 
 
def _save_or_show(fig, save_path):
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()