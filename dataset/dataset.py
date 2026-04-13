import torch
import numpy as np
import os
from torch.utils.data import Dataset
import sys
sys.path.append("..")
from config.simulation_config import SimulationConfig
from config.train_config import TrainConfig
from config.vae_config import VAEConfig
from simulator.bounce_ball import BouncingBallSim


class BallDataset(Dataset):
    """
    Dataset for ball sequences with obstacles.
    
    If the dataset file exists — load it.
    If it doesn't exist — generate and save it.
    """
    def __init__(self, sim_cfg: SimulationConfig, cfg: VAEConfig, tcfg: TrainConfig, split: str = "train"):
        super().__init__()
        assert split in ("train", "val", "test")

        self.sim_cfg = sim_cfg
        self.cfg     = cfg
        self.split   = split

        ball_path     = os.path.join(sim_cfg.data_dir, f"ball_{split}.npy")
        obstacle_path = os.path.join(sim_cfg.data_dir, f"obstacle_{split}.npy")
        control_path  = os.path.join(sim_cfg.data_dir, f"control_{split}.npy")

        if os.path.exists(ball_path) and os.path.exists(obstacle_path):
            self.ball_data     = np.load(ball_path)      # [N, T, H, W]
            self.obstacle_data = np.load(obstacle_path)  # [N, H, W]
            self.control_data  = np.load(control_path)   # [N, T, C]
        else:
            print(f"[Dataset] File not found, generating {split}...")
            self.ball_data, self.obstacle_data, self.control_data = self._generate_and_save(
                ball_path, obstacle_path, control_path, split, tcfg
            )

    def _generate_and_save(self, ball_path, obstacle_path, control_path, split, tcfg):
        n_episodes = {
            "train": self.sim_cfg.episodes,
            "val":   int(self.sim_cfg.episodes * tcfg.val_split),
            "test":  int(self.sim_cfg.episodes * tcfg.test_split),
        }[split]

        sim = BouncingBallSim(self.sim_cfg)
        sim.cfg.episodes = n_episodes
        ball_data, obstacle_data, raw_control = sim.generate_dataset(seed=self.sim_cfg.seed + {"train": 0, "val": 1, "test": 2}[split])

        N, T, _ = raw_control.shape
        dim_u = self.cfg.dim_u
        if dim_u > 0:
            control_data = np.zeros((N, T, dim_u), dtype=np.float32)
            curr_c = min(raw_control.shape[2], dim_u)
            control_data[:, :, :curr_c] = raw_control[:, :, :curr_c]
        else:
            control_data = np.array([])

        os.makedirs(self.sim_cfg.data_dir, exist_ok=True)
        np.save(ball_path,     ball_data)
        np.save(obstacle_path, obstacle_data)
        np.save(control_path,  control_data)
        print(f"[Dataset] Saved: {ball_path}, {obstacle_path}")

        return ball_data, obstacle_data, control_data

    def __len__(self):
        return len(self.ball_data)

    def __getitem__(self, idx):
        ball     = torch.from_numpy(self.ball_data[idx])      # [T, H, W]
        obstacle = torch.from_numpy(self.obstacle_data[idx])  # [H, W]]

        if self.cfg.dim_u > 0:
            control = torch.from_numpy(self.control_data[idx]) # [T, dim_u]
        else:
            control = None
            
        return ball, obstacle, control