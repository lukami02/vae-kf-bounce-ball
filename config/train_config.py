from dataclasses import dataclass, field
import torch.optim as optim


@dataclass
class TrainConfig:

    # Training
    epochs: int = 100
    vae_pretrain_epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    seed: int = 42
    val_split:  float = 0.1    
    test_split: float = 0.01
    pos_weight: float = 5.0      # binary cross entropy positive class weight

    # Optimizer
    optimizer: str   = "adamw"     
    weight_decay: float = 1e-4       
    lr_scheduler: str = "cosine" 
    lr_step_size: int = 50 
    lr_gamma: float = 0.5       

    # Loss weights
    
    lambda_recon: float = 1.0      # reconstruction loss
    lambda_innov: float = 1.2      # innovation loss
    lambda_posterior: float = 0.0  # posterior loss
    lambda_prior: float = 0.5      # prior loss
    lambda_entropy: float = 0.1    # Entropy loss
    lambda_alpha: float = 0.0      # Alpha loss
    lambda_imm: float = 0.5

    lambda_pred: float = 0.3       # prediction loss
    lambda_kl: float = 1           # KL divergence
    lambda_reg: float = 0.3        # Regularization loss

    # KL annealing
    kl_annealing: bool = True
    kl_warmup_epochs: int = 10  

    # Free-running training
    free_running_steps: int = 15     # autoregressive rollout length
    free_running_warmup: int = 50
    p_mask: float = 0.1

    imm_warmup_epochs: int  = 10

    # Logging
    log_every: int = 5
    save_every: int = 10
    checkpoint_dir: str = "checkpoints/"

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
    
    def get_lambda_free(self, epoch: int) -> float:
        """Enable free-running loss after a specified epoch"""
        if epoch < self.free_running_warmup:
            return self.lambda_free * (epoch / self.free_running_warmup)
        return self.lambda_free
    
    def get_lambda_imm(self, epoch: int) -> float:
        if epoch < self.imm_warmup_epochs:
            return 0.0
        ramp = min(1.0, (epoch - self.imm_warmup_epochs) / 20)
        return self.lambda_imm * ramp