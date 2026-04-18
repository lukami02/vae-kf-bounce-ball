from dataclasses import dataclass, field
import torch.optim as optim


@dataclass
class TrainConfig:

    # training kvae
    alpha_warmup_epochs: int = 20      # train without alpha net
    full_training_epochs: int = 20     # full KVAE training
    finetune_epochs: int = 40          # after LR decay + optimizer reset
    masking_epochs: int = 40           # random masking phase
    mask_ramp_epochs: int = 60         # progressive masking phase
    decoder_only_epochs: int = 20      # final decoder-only training

    ## masking 
    p_mask: float = 0.3                # random masking probability
    max_mask_ratio: float = 0.25       # max % of sequence masked

    # trainig cvvae/gruvae
    pred_warmup_epochs:   int = 20     # train without prediction loss
    full_training_epochs: int = 80     # full VAE training
    masking_epochs:       int = 30     # random masking phase
    mask_ramp_epochs:     int = 30     # progressive masking phase
    
    kl_annealing: bool = True
    kl_warmup_epochs: int = 30
    kl_trans_warmup_epochs: int = 50

    ## weight transfer
    use_kvae_weights_cv:  bool = True
    use_kvae_weights_gru: bool = True
    kvae_checkpoint_path: str  = "checkpoints/kvae/best_kvae.pt"

    ## skip training
    train_cv:  bool = True
    train_gru: bool = False

    # optimization 
    learning_rate: float = 5e-3
    enc_dec_lr_factor: float = 0.1     # LR reduction for encoder/decoder
    optimizer: str = "adamw"
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"
    lr_eta_min: float = 1e-5
    lr_step_size: int = 5
    grad_clip: float = 1.0

    # data
    batch_size: int = 128
    val_split: float = 1/16
    test_split: float = 1/16
    seed: int = 42

    # loss / traing tricks 
    pos_weight: float = 3.0        # BCE positive class weight
    burn_in: int = 3               # steps ignored at sequence start
    epoch_burn_in: int = 5         # epochs using burn-in

    ## loss weights
    lambda_recon: float = 0.3      # reconstruction loss
    lambda_innov: float = 1.0      # innovation loss
    lambda_posterior: float = 1.0  # posterior loss
    lambda_prior: float = 1.0      # prior loss
    lambda_entropy: float = 1.0    # Entropy loss
    lambda_alpha: float = 1.0      # Alpha loss
    lambda_pred: float = 1.0       # prediction loss
    lambda_kl_trans: float = 0.0   # transition loss

    # logging
    log_every: int = 5
    save_every: int = 10
    checkpoint_dir: str = "checkpoints/"

    def get_total_epochs(self) -> int:
        """Total epochs for KVAE training"""
        return self.alpha_warmup_epochs + self.full_training_epochs + self.finetune_epochs + \
               self.masking_epochs + self.mask_ramp_epochs + self.decoder_only_epochs

    def get_lambda_kl(self, epoch: int) -> float:
        """Linear KL annealing."""
        if not self.kl_annealing:
            return self.lambda_kl
        if epoch >= self.kl_warmup_epochs:
            return self.lambda_kl
        return self.lambda_kl * (epoch / self.kl_warmup_epochs)

    def get_lambda_pred(self, epoch: int) -> float:
        """Prediction loss warmup."""
        if epoch < self.pred_warmup_epochs:
            return self.lambda_pred * (epoch / self.pred_warmup_epochs)
        return self.lambda_pred
    
    def get_lambda_kl_trans(self, epoch: int) -> float:
        """Transition loss warmup"""
        if epoch < self.kl_trans_warmup_epochs:
            return self.lambda_kl_trans * (epoch / self.kl_trans_warmup_epochs)
        return self.kl_trans_warmup_epochs