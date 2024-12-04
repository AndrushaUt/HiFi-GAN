from src.model.hifi_gan.residual_block import ResBlock

import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.initial_conv = nn.Conv1d(80, 512, 7, padding=3)

        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, 16, 8, padding=4),
            nn.ConvTranspose1d(256, 128, 16, 8, padding=4),
            nn.ConvTranspose1d(128, 64, 4, 2, padding=1),
            nn.ConvTranspose1d(64, 32, 4, 2, padding=1)
        ])

        self.resblocks = nn.ModuleList([
            ResBlock(256, 3, [1, 3, 5]),
            ResBlock(128, 3, [1, 3, 5]),
            ResBlock(64, 3, [1, 3, 5]),
            ResBlock(32, 3, [1, 3, 5])
        ])

        self.final_conv = nn.Conv1d(32, 1, 7, padding=3)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.initial_conv(x)
        for upsample, resblock in zip(self.upsample_layers, self.resblocks):
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)
            x = resblock(x)
        x = self.final_conv(F.leaky_relu(x, 0.1))
        x = self.activation(x)
        return x
