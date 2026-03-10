from dataclasses import dataclass, field
import torch.optim as optim


@dataclass
class TrainConfig:

    # Training
    epochs: int = 100
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
    lambda_recon: float = 1.0      # reconstruction loss
    lambda_pred: float = 0.8       # prediction loss
    lambda_kl: float = 0.5         # KL divergence
    lambda_innov: float = 0.05     # innovation loss
    lambda_free: float = 1.5       # autoregressive loss

    # KL annealing
    kl_annealing: bool = True
    kl_warmup_epochs: int = 20  

    # Prediction loss warmup
    pred_warmup_epochs: int = 20     

    # Free-running training
    free_running_steps: int = 10     # autoregressive rollout length
    free_running_warmup: int = 40

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
    
    def get_lambda_free(self, epoch: int) -> float:
        """Enable free-running loss after a specified epoch"""
        if epoch < self.free_running_warmup:
            return 0.0
        return self.lambda_free