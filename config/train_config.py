from dataclasses import dataclass, field
import torch.optim as optim


@dataclass
class TrainConfig:

    # Training
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    grad_clip: float = 5.0
    seed: int = 42
    val_split:  float = 0.1    
    test_split: float = 0.01

    # Optimizer
    optimizer: str   = "adamw"     
    weight_decay: float = 1e-4       
    lr_scheduler: str = "cosine" 
    lr_step_size: int = 50 
    lr_gamma: float = 0.5       

    # Loss weights
    lambda_recon: float = 0.8      # reconstruction loss
    lambda_pred: float = 1.0       # prediction loss
    lambda_kl: float = 1.5         # KL divergence
    lambda_innov: float = 0.3      # innovation loss
    lambda_free: float = 1.0       # autoregressive loss

    # KL annealing
    kl_annealing: bool = True
    kl_warmup_epochs: int = 5  

    # Prediction loss warmup
    pred_warmup_epochs: int = 5     

    # Free-running training
    free_running_steps: int = 10     # autoregressive rollout length
    free_running_warmup: int = 10

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
            return self.pred_warmup_epochs * (epoch / self.pred_warmup_epochs)
        return self.lambda_pred
    
    def get_lambda_free(self, epoch: int) -> float:
        """Enable free-running loss after a specified epoch"""
        if epoch < self.free_running_warmup:
            return self.lambda_free * (epoch / self.free_running_warmup)
        return self.lambda_free