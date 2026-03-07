from dataclasses import dataclass

@dataclass
class SimulationConfig:
    # simulation 
    size: tuple[int, int] = (32, 32)       # (H, W)
    gravity: bool = False                  # enable gravity
    seed: int | None = None                # RNG seed
    T: int = 60                            # number of frames per episode
    substeps: int = 4                      # number of physics substeps per frame
    episodes: int = 10000                  # number of episodes to generate

    # ball
    ball_scale: float = 0.03               # ball radius as proportion of size
    ball_sigma: float = 0.03               # radius of Gaussian blob in pixels
    speed_range: tuple[float, float] = (0.03, 0.08)
    

    # obstacles 
    obstacle_min_scale: float = 0.12       # min obstacle size proportion
    obstacle_max_scale: float = 0.30       # max obstacle size proportion
    num_obstacles: int = 2                 # default number of obstacles

    # physics 
    gravity_strength_scale: float = 0.002  # gravity strength proportion

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



    
