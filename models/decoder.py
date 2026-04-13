import torch
import torch.nn as nn
import torch.distributions as D
import sys
sys.path.append("..")
from config.vae_config import VAEConfig
from config.simulation_config import SimulationConfig


class BallDecoder(nn.Module):
    """
    Decodes latent vector back to ball image.
    """
    def __init__(self, vae_cfg: VAEConfig, sim_cfg: SimulationConfig):
        super().__init__()
        self.cfg = vae_cfg
        self.sim_cfg = sim_cfg

        self.n_upsample = len(vae_cfg.decoder_channels)
        H, W = sim_cfg.size
        assert H == W, "Only square images supported"

        # Calculate bottleneck spatial dimensions
        self.init_size = H // (2 ** self.n_upsample)
        self.init_channels = vae_cfg.decoder_channels[0]

        # Project latent vector
        self.fc = nn.Linear(vae_cfg.dim_a, self.init_channels * self.init_size * self.init_size)

        # Upsample blocks
        layers = []
        channels = vae_cfg.decoder_channels 
        for i in range(len(channels)):
            in_ch  = channels[i]
            out_ch = channels[i + 1] if i < len(channels) - 1 else 1

            # Double spatial resolution in each step
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))

            if i < len(channels) - 1:
                layers.append(vae_cfg.dec_activation())

        self.upsample = nn.Sequential(*layers)

    def forward(self, a_seq):
        """
        a_seq: [B, T, dim_a] — Latent state sequence
    
        dist:  [B, T, H, W]   — Bernoulli distribution of pixels
        """
        B, T, _ = a_seq.shape

        a_flat = a_seq.view(B * T, self.cfg.dim_a)                              # [B*T, dim_a]

        # Project and reshape to 4D tensor
        x = self.fc(a_flat)                                                     # [B*T, C*h*w]
        x = x.view(B * T, self.init_channels, self.init_size, self.init_size)   # [B*T, C, h, w]

        # Upsample to target resolution
        x = self.upsample(x)                                                    # [B*T, 1, H, W]

        # Restore sequence dimensions
        x = x.view(B, T, self.sim_cfg.size[0], self.sim_cfg.size[1])            # [B, T, H, W]
        
        return D.Bernoulli(logits=x)