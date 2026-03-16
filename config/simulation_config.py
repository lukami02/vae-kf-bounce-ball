from dataclasses import dataclass

@dataclass
class SimulationConfig:
    # simulation 
    size: tuple[int, int] = (32, 32)       # (H, W)
    gravity: bool = False                  # enable gravity
    seed: int = 42                         # RNG seed
    T: int = 60                            # number of frames per episode
    substeps: int = 4                      # number of physics substeps per frame
    episodes: int = 8196                   # number of episodes to generate
    data_dir:  str = "dataset/"            # directory to save generated .npy files

    # ball
    ball_scale: float = 0.03               # ball radius as proportion of size
    ball_sigma: float = 0.03               # radius of Gaussian blob in pixels
    ball_radius: float = 0.06
    speed_range: tuple[float, float] = (0.03, 0.08)
    ball_gaussian: bool = False
    

    # obstacles 
    obstacle_min_scale: float = 0.12       # min obstacle size proportion
    obstacle_max_scale: float = 0.30       # max obstacle size proportion
    num_obstacles: int = 2                 # default number of obstacles

    # physics 
    gravity_strength_scale: float = 0.002  # gravity strength proportion