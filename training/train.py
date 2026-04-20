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

def build_model(model_name, cfg, sim_cfg, tcfg):
    if model_name == "kvae":
        return KVAE(cfg, sim_cfg, tcfg)
    elif model_name == "gru_vae":
        return GRUVAE(cfg, sim_cfg, tcfg)
    elif model_name == "cv_vae":
        return CVVAE(cfg, sim_cfg, tcfg)
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
    
def get_optimizer(model, tcfg: TrainConfig, phase=0):
    if isinstance(model, KVAE):
        lr_decay = 1 if phase != 3 else tcfg.enc_dec_lr_factor
            
        param_groups = [
            {"params": model.ball_encoder.parameters(),     "lr": lr_decay * tcfg.learning_rate},
            {"params": model.obstacle_encoder.feature_proj.parameters(), "lr": tcfg.learning_rate},
            {"params": model.decoder.parameters(),          "lr": lr_decay * tcfg.learning_rate},
            {"params": model.alpha_net.parameters(),        "lr": tcfg.learning_rate},
            {"params": model.kalman.parameters(),           "lr": tcfg.learning_rate},
            {"params": [model.A_matrices],                  "lr": tcfg.learning_rate},
            {"params": [model.C_matrices],                  "lr": tcfg.learning_rate},
        ]
        if model.B_matrices is not None:
            param_groups.append(
                {"params": [model.B_matrices], "lr": tcfg.learning_rate}
            )
    else:
        param_groups = [{"params": model.parameters(), "lr": tcfg.learning_rate}]

    if tcfg.optimizer == "adam":
        return torch.optim.Adam(param_groups)
    elif tcfg.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=tcfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {tcfg.optimizer}")

def get_scheduler(optimizer, tcfg: TrainConfig, model_type="kvae"):
    if tcfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.get_total_epochs(model_type), eta_min=tcfg.lr_eta_min)
    elif tcfg.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=tcfg.lr_step_size, gamma=tcfg.lr_gamma)
    elif tcfg.lr_scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {tcfg.lr_scheduler}")

def run_epoch(model, loader, optimizer, cfg, tcfg, epoch, mask, device,
              model_type="kvae", is_train=True):
    model.train() if is_train else model.eval()
    
    is_kvae = (model_type == "kvae")
    if is_kvae:
        keys = ["loss", "recon", "innov", "prior", "entropy", "posterior", "bounce"]
    else:
        keys = ["loss", "recon", "pred", "kl"]
        if model_type == "gru":
            keys.append("trans")

    total_terms = {k: 0.0 for k in keys}

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            ball_seq, obstacle_img, *rest = batch
            u_seq = rest[0].to(device) if rest and rest[0] is not None else None
            
            ball_seq = ball_seq.to(device)
            obstacle_img = obstacle_img.to(device)
            current_mask = (mask[:ball_seq.shape[0], :] if mask is not None else None)

            outputs = model(ball_seq, obstacle_img, u_seq=u_seq, mask=current_mask, epoch=epoch, smoother=True)
            
            (x_dist_smooth, a_dist, a_seq, a_smooth, a_pred_smooth,
             z_dist, z_smooth, z_pred, R, Q, alpha_seq) = outputs

            loss, terms = compute_loss(
                ball_seq=ball_seq, x_dist_smooth=x_dist_smooth,
                a_dist=a_dist, a_seq=a_seq, a_smooth=a_smooth, a_pred=a_pred_smooth,
                z_dist=z_dist, z_smooth=z_smooth, z_pred=z_pred,
                R=R, Q=Q, alpha_seq=alpha_seq, mask=current_mask, cfg=cfg,
                tcfg=tcfg, epoch=epoch, model_type=model_type, u_seq=u_seq
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if tcfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                optimizer.step()

            for k, v in terms.items():
                if k in total_terms:
                    total_terms[k] += v.item() if hasattr(v, 'item') else v

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

def make_mask(B, T, device, free_running_steps, p_mask, randomize_start=False):
    # initialize mask (1 = keep, 0 = mask)
    mask = torch.ones(B, T, device=device)

    for b in range(B):
        if randomize_start:
            min_start = T // 2
            max_start = max(T - free_running_steps, T // 2 + 1)
            start = torch.randint(min_start, max_start, (1,)).item()
        else:
            start = T - free_running_steps

        # mask contiguous segment
        mask[b, start:start + free_running_steps] = 0.0

    # random masking
    rand_mask = torch.rand(B, T, device=device) < p_mask
    rand_mask[:, :int(0.1 * T)] = False 
    mask[rand_mask] = 0.0

    return mask

def log_epoch(model_type, model, epoch, total_epochs, optimizer, train_terms, val_terms, tcfg, logger):
    if epoch % tcfg.log_every != 0 and epoch != 1:
        return
    
    lr = optimizer.param_groups[0]["lr"]

    if model_type == "kvae":
        logger.info(
            f"Epoch {epoch:4d}/{total_epochs} | lr={lr:.2e} | "
            f"loss={train_terms['loss']:.4f}  "
            f"recon={train_terms['recon']:.4f}  "
            f"innov={train_terms['innov']:.4f}  "
            f"prior={train_terms['prior']:.4f}  "
            f"entropy={train_terms['entropy']:.4f}  "
            f"posterior={train_terms['posterior']:.4f} | "
            f"val={val_terms['loss']:.4f}"
        )
    elif model_type == "gru":
        logger.info(
            f"Epoch {epoch:4d}/{total_epochs} | lr={lr:.2e} | "
            f"loss={train_terms['loss']:.4f}  "
            f"recon={train_terms['recon']:.4f}  "
            f"pred={train_terms['pred']:.4f}  "
            f"kl={train_terms['kl']:.4f}  "
            f"kl_trans={train_terms.get('trans', 0):.4f} | "
            f"val={val_terms['loss']:.4f}"
        )
    else:  # cv
        logger.info(
            f"Epoch {epoch:4d}/{total_epochs} | lr={lr:.2e} | "
            f"loss={train_terms['loss']:.4f}  "
            f"recon={train_terms['recon']:.4f}  "
            f"pred={train_terms['pred']:.4f}  "
            f"kl={train_terms['kl']:.4f} | "
            f"val={val_terms['loss']:.4f}"
        )

    if epoch % tcfg.save_every == 0:
        save_checkpoint(model, optimizer, epoch, train_terms, tcfg, logger)

def load_kvae_weights(model, kvae_path, logger, device):
    """
    Loads compatible weights from a pretrained KVAE checkpoint into model.
    """
    if not kvae_path or not os.path.exists(kvae_path):
        logger.warning(f"  KVAE checkpoint not found at {kvae_path}, skipping weight transfer")
        return

    checkpoint = torch.load(kvae_path, map_location=device)
    old_sd     = checkpoint.get('model', checkpoint)
    new_sd     = model.state_dict()

    compatible_sd = {
        k: v for k, v in old_sd.items()
        if k in new_sd and new_sd[k].shape == v.shape
    }

    new_sd.update(compatible_sd)
    model.load_state_dict(new_sd)

    logger.info(f"  Loaded {len(compatible_sd)}/{len(new_sd)} params from KVAE checkpoint")
    logger.info(f"  Transferred: {list(compatible_sd.keys())}")

def train_KVAE(model, train_loader, val_loader, cfg, sim_cfg, tcfg, device, logger):
    model = model.to(device)
    optimizer = get_optimizer(model, tcfg)
    scheduler = get_scheduler(optimizer, tcfg, "kave")
    best_val_loss = float("inf")
    cur_epoch = 0

    phases = [
        {"name": "Phase 1: Warmup", "epochs": tcfg.alpha_warmup_epochs, "mask_type": "none"},
        {"name": "Phase 2: Full Training", "epochs": tcfg.full_training_epochs, "mask_type": "none"},
        {"name": "Phase 3: Finetuning", "epochs": tcfg.finetune_epochs, "mask_type": "none", "reset_opt": True},
        {"name": "Phase 4: Random Masking", "epochs": tcfg.masking_epochs, "mask_type": "random"},
        {"name": "Phase 5: Progressive Masking", "epochs": tcfg.mask_ramp_epochs, "mask_type": "ramp"},
        {"name": "Phase 6: Decoder Only", "epochs": tcfg.decoder_only_epochs, "mask_type": "none", "freeze_encoder": True}
    ]

    logger.info("=" * 60)
    logger.info(f"Starting KVAE Training | Device: {device}")
    logger.info("=" * 60)

    for phase in phases:
        if phase["epochs"] <= 0: continue
        
        logger.info(f"\n>>> {phase['name']} for {phase['epochs']} epochs")
        
        if phase.get("reset_opt"):
            optimizer = get_optimizer(model, tcfg, phase=3)
            scheduler = get_scheduler(optimizer, tcfg, "kvae")
            
        if phase.get("freeze_encoder"):
            for param in model.parameters(): param.requires_grad = False
            for param in model.decoder.parameters(): param.requires_grad = True

        phase_start_epoch = cur_epoch
        for epoch in range(cur_epoch, cur_epoch + phase["epochs"]):
            B, T = tcfg.batch_size, sim_cfg.T
            
            if phase["mask_type"] == "none":
                mask = make_mask(B, T, device, 0, 0, randomize_start=True)
            elif phase["mask_type"] == "random":
                mask = make_mask(B, T, device, 0, tcfg.p_mask, randomize_start=True)
            elif phase["mask_type"] == "ramp":
                steps = int(T * tcfg.max_mask_ratio * (epoch - phase_start_epoch) / phase["epochs"])
                mask = make_mask(B, T, device, steps, tcfg.p_mask, randomize_start=True)

            mask_val = make_mask(B, T, device, int(T * tcfg.max_mask_ratio), p_mask=0, randomize_start=False)

            train_terms = run_epoch(model, train_loader, optimizer, cfg, tcfg,
                                    epoch, mask, device, model_type="kvae", is_train=True)
            val_terms   = run_epoch(model, val_loader, None, cfg, tcfg,
                                    epoch, mask_val, device, model_type="kvae", is_train=False)

            if scheduler: scheduler.step()

            log_epoch("kvae", model, epoch, tcfg.get_total_epochs("kvae"), optimizer,
                      train_terms, val_terms, tcfg, logger)
            
            if val_terms["loss"] < best_val_loss:
                best_val_loss = val_terms["loss"]
                save_checkpoint(model, optimizer, epoch, val_terms, tcfg, logger, path=f"{tcfg.checkpoint_dir}/best_kvae.pt")
                logger.info(f"  New best val loss: {best_val_loss:.4f}")

        cur_epoch += phase["epochs"]
        
        if phase.get("freeze_encoder"):
            for param in model.parameters(): param.requires_grad = True

    logger.info(f"\nTraining done. Best val loss: {best_val_loss:.4f}")

def train_VAE(model, train_loader, val_loader, cfg, sim_cfg, tcfg, device, logger, model_type='cv'):
    """
    Unified training loop for CVVAE ('cv') and GRUVAE ('gru').
    """
    assert model_type in ('cv', 'gru'), f"Unknown model_type: {model_type}"

    use_kvae_weights = getattr(tcfg, f'use_kvae_weights_{model_type}', False)
    do_train         = getattr(tcfg, f'train_{model_type}', True)
    ckpt_name        = f"best_{model_type}_vae.pt"

    model = model.to(device)

    if use_kvae_weights:
        logger.info(f"  Transferring KVAE weights into {model_type.upper()}VAE...")
        load_kvae_weights(model, tcfg.kvae_checkpoint_path, logger, device)

    if not do_train:
        logger.info(f"  train_{model_type}=False — skipping training, model used as frozen encoder/decoder")
        os.makedirs(tcfg.checkpoint_dir, exist_ok=True)
        torch.save({"model": model.state_dict()}, f"{tcfg.checkpoint_dir}/{ckpt_name}")
        logger.info(f"Checkpoint saved: {tcfg.checkpoint_dir}/{ckpt_name}")
        return model

    phases = [
        {"name": f"Phase 1: Warmup",             "epochs": tcfg.pred_warmup_epochs_vae,   "mask_type": "none"},
        {"name": f"Phase 2: Full Training",      "epochs": tcfg.full_training_epochs_vae, "mask_type": "none"},
        {"name": f"Phase 3: Random Masking",     "epochs": tcfg.masking_epochs_vae,       "mask_type": "random"},
        {"name": f"Phase 4: Progressive Masking","epochs": tcfg.mask_ramp_epochs_vae,     "mask_type": "ramp"},
    ]

    optimizer    = get_optimizer(model, tcfg)
    scheduler    = get_scheduler(optimizer, tcfg, model_type)
    best_val_loss = float("inf")
    cur_epoch     = 0

    logger.info("=" * 60)
    logger.info(f"Starting {model_type.upper()}VAE Training | Device: {device}")
    logger.info("=" * 60)

    for phase in phases:
        if phase["epochs"] <= 0: continue

        logger.info(f"\n>>> {phase['name']} for {phase['epochs']} epochs")
        phase_start_epoch = cur_epoch

        for epoch in range(cur_epoch, cur_epoch + phase["epochs"]):
            B, T = tcfg.batch_size, sim_cfg.T

            if phase["mask_type"] == "none":
                mask = make_mask(B, T, device, 0, 0, randomize_start=True)
            elif phase["mask_type"] == "random":
                mask = make_mask(B, T, device, 0, tcfg.p_mask, randomize_start=True)
            elif phase["mask_type"] == "ramp":
                steps = int(T * tcfg.max_mask_ratio
                            * (epoch - phase_start_epoch) / phase["epochs"])
                mask = make_mask(B, T, device, steps, tcfg.p_mask, randomize_start=True)

            mask_val = make_mask(B, T, device, int(T * tcfg.max_mask_ratio),
                                p_mask=0, randomize_start=False)

            train_terms = run_epoch(model, train_loader, optimizer, cfg, tcfg,
                                    epoch, mask, device, model_type, is_train=True)
            val_terms   = run_epoch(model, val_loader, None, cfg, tcfg,
                                    epoch, mask_val, device, model_type, is_train=False)

            if scheduler:
                scheduler.step()

            log_epoch(model_type, model, epoch, tcfg.get_total_epochs(model_type), optimizer,
                      train_terms, val_terms, tcfg, logger)
            
            if val_terms["loss"] < best_val_loss:
                best_val_loss = val_terms["loss"]
                save_checkpoint(model, optimizer, epoch, val_terms, tcfg, logger,
                                path=f"{tcfg.checkpoint_dir}/{ckpt_name}")
                logger.info(f"  New best val loss: {best_val_loss:.4f}")

        cur_epoch += phase["epochs"]

    logger.info(f"\nTraining done. Best val loss: {best_val_loss:.4f}")

    return model

def train(model, train_loader, val_loader, cfg, sim_cfg, tcfg, device, logger):
    if isinstance(model, KVAE):
        train_KVAE(model, train_loader, val_loader, cfg, sim_cfg, tcfg, device, logger)
    elif isinstance(model, CVVAE):
        train_VAE(model, train_loader, val_loader, cfg, sim_cfg, tcfg, device, logger, model_type='cv')
    elif isinstance(model, GRUVAE):
        train_VAE(model, train_loader, val_loader, cfg, sim_cfg, tcfg, device, logger, model_type='gru')
    return model

if __name__ == "__main__":
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg     = VAEConfig()
    sim_cfg = SimulationConfig()
    tcfg    = TrainConfig()

    {"name": "Phase 1: Warmup", "epochs": tcfg.alpha_warmup_epochs, "mask_type": "none"},
    {"name": "Phase 2: Full Training", "epochs": tcfg.full_training_epochs, "mask_type": "none"},
    {"name": "Phase 3: Finetuning", "epochs": tcfg.finetune_epochs, "mask_type": "none", "reset_opt": True},
    {"name": "Phase 4: Random Masking", "epochs": tcfg.masking_epochs, "mask_type": "random"},
    {"name": "Phase 5: Progressive Masking", "epochs": tcfg.mask_ramp_epochs, "mask_type": "ramp"},
    {"name": "Phase 6: Decoder Only", "epochs": tcfg.decoder_only_epochs, "mask_type": "none", "freeze_encoder": True}

    tcfg.checkpoint_dir = f"checkpoints/{args.model}"

    logger = setup_logger(log_dir=f"logs/{args.model}", log_file="train.log")
    logger.info(f"Model: {args.model}")
    
    torch.manual_seed(sim_cfg.seed)

    model = build_model(args.model, cfg, sim_cfg, tcfg)
    
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        logger.info(f"Checkpoint loaded: {args.checkpoint}")

    train_dataset = BallDataset(sim_cfg=sim_cfg, cfg=cfg, tcfg=tcfg, split="train")
    val_dataset   = BallDataset(sim_cfg=sim_cfg, cfg=cfg, tcfg=tcfg, split="val")

    train_loader = DataLoader(train_dataset, batch_size=tcfg.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=tcfg.batch_size, shuffle=False, num_workers=4)

    train(model, train_loader, val_loader, cfg, sim_cfg, tcfg, device, logger)