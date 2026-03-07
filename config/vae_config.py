from dataclasses import dataclass

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



    
