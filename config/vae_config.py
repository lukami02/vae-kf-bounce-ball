from dataclasses import dataclass
import torch.nn as nn

@dataclass
class VAEConfig:
    # Kalman filter dimensions 
    dim_a: int = 2                         # dimension of action vector
    dim_z: int = 4                         # dimension of latent state
    dim_u: int = 0                         # dimension of control input (set to 0 if not used)
    num_matrices: int = 3                  # number of different matrices to experiment with

    # standard deviations for initialization 
    B_std: float = 0.01                    # standard deviation for B matrices
    C_std: float = 0.01                    # standard deviation for C matrices
    Q_std: float = 0.025                   # standard deviation for process noise covariance Q
    R_std: float = 0.01                    # standard deviation for observation noise covariance R

    # encoder architecture
    encoder_ball_channels: list = [8, 16, 32]       # filters for moving ball CNN
    encoder_obstacle_channels: list = [16, 32, 32]  # filters for static obstacle CNN
    dim_obstacle: int = 16                          # latent size for obstacle features
    enc_activation: type = nn.ELU                   # activation class for encoder layers

    
