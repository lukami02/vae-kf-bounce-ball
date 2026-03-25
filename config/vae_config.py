from dataclasses import dataclass, field
import torch.nn as nn

@dataclass
class VAEConfig:
    # Kalman filter dimensions 
    dim_a: int = 2                         # dimension of action vector
    dim_z: int = 4                         # dimension of latent state
    dim_u: int = 0                         # dimension of control input (set to 0 if not used)
    num_matrices: int = 3                  # number of different matrices to experiment with
    burn_in: int = 3                       # number of first steps not to train

    # GRU dimension
    gru_hidden_dim: int = 16
    gru_layers: int = 2

    # standard deviations for initialization 
    A_std: float = 0.1                    # standard deviation for A matrices
    B_std: float = 0.1                     # standard deviation for B matrices
    C_std: float = 0.1                     # standard deviation for C matrices
    Q_std: float = 0.1                     # standard deviation for process noise covariance Q
    R_std: float = 0.1                     # standard deviation for observation noise covariance R
    QR_reg: float = 1e-3                   # diagonal regularization added to Q and R for numerical stability

    # encoder architecture
    encoder_ball_channels: list = field(default_factory=lambda: [32, 64] )    # filters for moving ball CNN
    encoder_obstacle_channels: list = field(default_factory=lambda: [32, 64]) # filters for static obstacle CNN
    dim_obstacle: int = 16                          # latent size for obstacle features
    enc_activation: type = nn.ELU                   # activation class for encoder layers

    # decoder architecture
    decoder_channels: list = field(default_factory=lambda: [64, 32])         # filters for moving ball CNN
    dec_activation: type = nn.ELU                   # activation class for decoder layers

    # alpha network
    alpha_units: int = 32                  # hidden units in alpha network

    def get_temperature(self, epoch: int) -> float:
        temp = max(0.1, 1.0 * (0.95 ** epoch))
        return temp
    
