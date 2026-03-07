import torch
import torch.nn as nn
import sys
sys.path.append("..")
from config.vae_config import VAEConfig
from config.simulation_config import SimulationConfig


class BallDecoder(nn.Module):
    """
    Decodes latent vector back to ball image using Pixel Shuffle upsampling.
    """
    def __init__(self, vae_cfg: VAEConfig, sim_cfg: SimulationConfig):
        super().__init__()
        self.cfg = vae_cfg

        self.n_upsample = len(vae_cfg.decoder_channels)
        H, W = sim_cfg.size
        assert H == W, "Only square images supported"

        self.init_size = H // (2 ** self.n_upsample)
        self.init_channels = vae_cfg.decoder_channels[0]

        self.fc = nn.Linear(vae_cfg.dim_a, self.init_channels * self.init_size * self.init_size)

        # Upsample blocks
        layers = []
        channels = vae_cfg.decoder_channels 
        for i in range(len(channels)):
            in_ch  = channels[i]
            out_ch = channels[i + 1] if i < len(channels) - 1 else 1

            layers += [
                nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1),  # *4 for PixelShuffle
                nn.PixelShuffle(2),                                        # out_ch x 2H x 2W
                vae_cfg.dec_activation(),
            ]

        layers += [nn.Conv2d(1, 1, kernel_size=3, padding=1), nn.Sigmoid()]
        self.upsample = nn.Sequential(*layers)

    def forward(self, a_seq):
        B, T, _ = a_seq.shape

        a_flat = a_seq.view(B * T, self.cfg.dim_a)                              # [B*T, dim_a]
        x = self.fc(a_flat)                                                     # [B*T, C*h*w]
        x = x.view(B * T, self.init_channels, self.init_size, self.init_size)   # [B*T, C, h, w]
        x = self.upsample(x)                                                    # [B*T, 1, H, W]

        return x.view(B, T, self.sim_cfg.size[0], self.sim_cfg.size[1])         # [B, T, H, W]