import torch
import logging
import argparse
import os
import sys
sys.path.append("..")
from torch.utils.data import DataLoader
from config.vae_config import VAEConfig
from config.simulation_config import SimulationConfig
from config.train_config import TrainConfig
from models.kvae import KVAE
from models.cv_vae import CVVAE
from models.gru_vae import GRUVAE
from training.loss import compute_loss
from dataset.dataset import BallDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="kvae", choices=["kvae", "gru_vae", "cv_vae"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Continue training from checkpoint")
    return parser.parse_args()

def build_model(model_name, cfg, sim_cfg):
    if model_name == "kvae":
        return KVAE(cfg, sim_cfg)
    elif model_name == "gru_vae":
        return GRUVAE(cfg, sim_cfg)
    elif model_name == "cv_vae":
        return CVVAE(cfg, sim_cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def setup_logger(log_dir: str = "logs", log_file: str = "train.log") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("kvae")
    logger.setLevel(logging.DEBUG)

    # Format
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def get_optimizer(model, tcfg: TrainConfig):
    if tcfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=tcfg.learning_rate)
    elif tcfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {tcfg.optimizer}")


def get_scheduler(optimizer, tcfg: TrainConfig):
    if tcfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.epochs)
    elif tcfg.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=tcfg.lr_step_size, gamma=tcfg.lr_gamma)
    elif tcfg.lr_scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {tcfg.lr_scheduler}")


def train_epoch(model, loader, optimizer, cfg, tcfg, epoch, mask, device):
    model.train()
    if isinstance(model, KVAE):
        total_terms = {"loss": 0, "recon": 0, "innov": 0, "prior": 0, "entropy": 0, "posterior": 0}
    else:
        total_terms = {"loss": 0, "recon": 0, "reg": 0, "pred": 0, "entropy": 0}

    for batch in loader:
        if len(batch) == 2:
            ball_seq, obstacle_img = batch
            u_seq = None
        else:
            ball_seq, obstacle_img, u_seq = batch

        ball_seq     = ball_seq.to(device)
        obstacle_img = obstacle_img.to(device)
        u_seq        = u_seq.to(device) if u_seq is not None else None
        if mask is not None:
            current_mask = mask.expand(batch[0].shape[0], -1)
        else: current_mask = None
    
        (x_dist_filt, x_dist_pred, a_dist, a_seq, a_filt, a_pred, z_filt, P_filt, z_pred, P_pred, alpha_seq, R, Q) = model(ball_seq, obstacle_img, u_seq=u_seq, mask=current_mask)

        loss, terms = compute_loss(
            ball_seq   = ball_seq,
            x_dist_filt = x_dist_filt,
            a_dist     = a_dist,
            a_seq      = a_seq,
            a_pred     = a_pred,
            a_filt     = a_filt,
            z_pred     = z_pred,
            z_filt     = z_filt,
            P_pred     = P_pred,
            P_filt     = P_filt,
            R          = R,
            Q          = Q,
            cfg        = cfg,
            tcfg       = tcfg,
            epoch      = epoch
        )
        
        optimizer.zero_grad()
        loss.backward()
        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        for k, v in terms.items():
            if k in total_terms:
                total_terms[k] += v

    n = len(loader)
    return {k: v / n for k, v in total_terms.items()}


@torch.no_grad()
def eval_epoch(model, loader, cfg, tcfg, epoch, mask,device):
    model.eval()
    if isinstance(model, KVAE):
        total_terms = {"loss": 0, "recon": 0, "innov": 0, "prior": 0, "entropy": 0, "posterior": 0}
    else:
        total_terms = {"loss": 0, "recon": 0, "reg": 0, "pred": 0, "entropy": 0}

    for batch in loader:
        if len(batch) == 2:
            ball_seq, obstacle_img = batch
            u_seq = None
        else:
            ball_seq, obstacle_img, u_seq = batch

        ball_seq     = ball_seq.to(device)
        obstacle_img = obstacle_img.to(device)
        u_seq        = u_seq.to(device) if u_seq is not None else None
        if mask is not None:
            current_mask = mask.expand(batch[0].shape[0], -1)
        else: current_mask = None

        (x_dist_filt, x_dist_pred, a_dist, a_seq, a_filt, a_pred, z_filt, P_filt, z_pred, P_pred, alpha_seq, R, Q) = model(ball_seq, obstacle_img, u_seq=u_seq, mask=current_mask)
        _, terms = compute_loss(
            ball_seq   = ball_seq,
            x_dist_filt = x_dist_filt,
            a_dist     = a_dist,
            a_seq      = a_seq,
            a_pred     = a_pred,
            a_filt     = a_filt,
            z_pred     = z_pred,
            z_filt     = z_filt,
            P_pred     = P_pred,
            P_filt     = P_filt,
            R          = R,
            Q          = Q,
            cfg        = cfg,
            tcfg       = tcfg,
            epoch      = epoch
        )

        for k, v in terms.items():
            if k in total_terms:
                total_terms[k] += v

    n = len(loader)
    return {k: v / n for k, v in total_terms.items()}


def save_checkpoint(model, optimizer, epoch, terms, tcfg, logger, path=None):
    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
    path = path or f"{tcfg.checkpoint_dir}/ckpt_epoch{epoch}.pt"
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "terms":     terms,
    }, path)
    logger.info(f"Checkpoint saved: {path}")


def train(model, train_loader, val_loader, cfg, sim_cfg, tcfg, device, logger):
    model = model.to(device)
    optimizer = get_optimizer(model, tcfg)
    scheduler = get_scheduler(optimizer, tcfg)

    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info(f"Device:     {device}")
    logger.info(f"Epochs:     {tcfg.epochs}")
    logger.info(f"Batch size: {tcfg.batch_size}")
    logger.info(f"Optimizer:  {tcfg.optimizer}  lr={tcfg.learning_rate}")
    logger.info(f"Scheduler:  {tcfg.lr_scheduler}")
    logger.info("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(1, tcfg.epochs + 1):

        mask_val = torch.ones(sim_cfg.T, device=device)
        mask_val[sim_cfg.T - tcfg.free_running_steps:] = 0.0
        if epoch >= tcfg.free_running_warmup:
            mask = mask_val.clone()
            rand_mask = torch.rand(sim_cfg.T, device=device) < tcfg.p_mask
            rand_mask[:int(0.2*sim_cfg.T)] = False
            mask[rand_mask] = 0.0
        else:
            mask = torch.ones(sim_cfg.T, device=device)
            rand_mask = torch.rand(sim_cfg.T, device=device) < tcfg.p_mask
            rand_mask[:int(0.2*sim_cfg.T)] = False
            mask[rand_mask] = 0.0

        train_terms = train_epoch(model, train_loader, optimizer, cfg, tcfg, epoch, mask, device)
        val_terms = eval_epoch(model, val_loader, cfg, tcfg, epoch, mask_val, device)

        if scheduler is not None:
            scheduler.step()

        # INFO log 
        if epoch % tcfg.log_every == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            if isinstance(model, KVAE):
                logger.info(
                    f"Epoch {epoch:4d}/{tcfg.epochs} | lr={lr:.2e} | "
                    f"loss={train_terms['loss']:.4f}  "
                    f"recon={train_terms['recon']:.4f}  "
                    f"innov={train_terms['innov']:.4f}  "
                    f"prior={train_terms['prior']:.4f}  "
                    f"entropy={train_terms['entropy']:.4f}  "
                    f"posterior={train_terms['posterior']:.4f} | "
                    f"val_loss={val_terms['loss']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch:4d}/{tcfg.epochs} | lr={lr:.2e} | "
                    f"loss={train_terms['loss']:.4f}  "
                    f"recon={train_terms['recon']:.4f}  "
                    f"reg={train_terms['reg']:.4f}  "
                    f"pred={train_terms['pred']:.4f}  "
                    f"entropy={train_terms['entropy']:.4f} | "
                    f"val_loss={val_terms['loss']:.4f}"
                )

        # Checkpoint
        if epoch % tcfg.save_every == 0:
            save_checkpoint(model, optimizer, epoch, train_terms, tcfg, logger)

        # Best model
        if val_terms["loss"] < best_val_loss:
            best_val_loss = val_terms["loss"]
            save_checkpoint(model, optimizer, epoch, val_terms, tcfg, logger,
                            path=f"{tcfg.checkpoint_dir}/best.pt")
            logger.info(f"  New best val loss: {best_val_loss:.4f}")

    save_checkpoint(model, optimizer, epoch, val_terms, tcfg, logger,
                    path=f"{tcfg.checkpoint_dir}/best.pt")

    logger.info("=" * 60)
    logger.info(f"Training done. Best val loss: {best_val_loss:.4f}")
    logger.info("=" * 60)
    return model


if __name__ == "__main__":
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg     = VAEConfig()
    sim_cfg = SimulationConfig()
    tcfg    = TrainConfig()
    
    tcfg.checkpoint_dir = f"checkpoints/{args.model}"

    logger = setup_logger(log_dir=f"logs/{args.model}", log_file="train.log")
    logger.info(f"Model: {args.model}")
    
    torch.manual_seed(sim_cfg.seed)

    model = build_model(args.model, cfg, sim_cfg)
    
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Checkpoint loaded: {args.checkpoint}")

    train_dataset = BallDataset(sim_cfg=sim_cfg, tcfg=tcfg, split="train")
    val_dataset   = BallDataset(sim_cfg=sim_cfg, tcfg=tcfg, split="val")

    train_loader = DataLoader(train_dataset, batch_size=tcfg.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=tcfg.batch_size, shuffle=False, num_workers=4)

    train(model, train_loader, val_loader, cfg, sim_cfg, tcfg, device, logger)