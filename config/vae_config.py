from dataclasses import dataclass, field
import torch.nn as nn

@dataclass
class VAEConfig:
    # Kalman filter dimensions 
    dim_a: int = 2                         # dimension of action vector
    dim_z: int = 4                         # dimension of latent state
    dim_u: int = 1                         # dimension of control input (set to 0 if not used)
    num_matrices: int = 3                  # number of different matrices to experiment with

    # GRU dimension 
    gru_hidden_dim: int = 32               # hidden units in alpha network
    gru_layers: int = 2                    # nuber of gru layers in gru_vae

    # standard deviations for initialization 
    A_std: float = 0.05                    # standard deviation for A matrices
    B_std: float = 0.05                    # standard deviation for B matrices
    C_std: float = 0.05                    # standard deviation for C matrices
    Q_std: float = 0.2                     # standard deviation for process noise covariance Q
    R_std: float = 0.3                     # standard deviation for observation noise covariance R

    # encoder architecture
    encoder_ball_channels: list = field(default_factory=lambda: [32, 32, 32] )    # filters for moving ball CNN
    dim_obstacle: int = 32                          # latent size for obstacle features
    enc_activation: type = nn.ReLU                  # activation class for encoder layers

    # decoder architecture
    decoder_channels: list = field(default_factory=lambda: [32, 32, 32])    # filters for moving ball CNN
    dec_activation: type = nn.ELU                   # activation class for decoder layers
    
