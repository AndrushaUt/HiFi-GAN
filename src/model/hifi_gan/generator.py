from src.model.hifi_gan.residual_block import MRF

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,
                 dims,
                 kernel_sizes,
                 mrf_kernel_sizes,
                 dilations,
                 paddings):
        super().__init__()
        self.initial_conv =nn.utils.weight_norm(
                nn.Conv1d(
                    in_channels=80,
                    out_channels=dims[0],
                    kernel_size=7,
                    dilation=1,
                    padding="same"
                )
            )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.ConvTranspose1d(
                    in_channels=dims[i],
                    out_channels=dims[i] // 2,
                    kernel_size=kernel_sizes[i],
                    stride=kernel_sizes[i] // 2,
                    padding=paddings[i],
                )),
                MRF(
                    num_channels=dims[i] // 2,
                    kernel_sizes=mrf_kernel_sizes,
                    dilations=dilations
                )
            )
            for i in range(len(kernel_sizes))
        ])
        self.final_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.utils.weight_norm(
                nn.Conv1d(
                    in_channels=32,
                    out_channels=1,
                    kernel_size=7,
                    padding="same"
                )
            ),
            nn.Tanh()
        )

        self.blocks = nn.ModuleList()
        self.blocks.append(self.initial_conv)
        self.blocks.extend(self.layers)
        self.blocks.append(self.final_layer)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x