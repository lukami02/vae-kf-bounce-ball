from dataclasses import dataclass, field
import torch.optim as optim


@dataclass
class TrainConfig:

    # Training
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    grad_clip: float = 5.0
    seed: int = 42

    # Optimizer
    optimizer: str   = "adam"     
    weight_decay: float = 1e-4       
    lr_scheduler: str = "cosine" 
    lr_step_size: int = 50 
    lr_gamma: float = 0.5       

    # Loss weights
    lambda_recon: float = 1.0      # reconstruction loss
    lambda_pred: float = 1.0       # prediction loss
    lambda_kl: float = 1.0         # KL divergence
    lambda_innov: float = 1.0      # innovation loss

    # KL annealing
    kl_annealing: bool = True
    kl_warmup_epochs: int = 50  

    # Prediction loss warmup
    pred_warmup_epochs: int = 50     

    # Logging
    log_every: int = 10
    save_every: int = 50
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
            return 0.0
        return self.lambda_pred