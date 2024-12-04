import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorP(nn.Module):
    def __init__(self, period):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(512, 1024, (5, 1), 1, padding=(2, 0)),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
                nn.LeakyReLU(0.1)
            )
        ])
        self.final_conv = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            pad_length = self.period - (t % self.period)
            x = F.pad(x, (0, pad_length), "reflect")
            t = t + pad_length

        x = x.view(b, c, t // self.period, self.period)
        for layer in self.conv_layers:
            x = layer(x)
            fmap.append(x)
        x = self.final_conv(x)
        fmap.append(x)
        x = x.flatten(1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([
            DiscriminatorP(p) for p in self.periods
        ])

    def forward(self, x):
        outputs = []
        feature_maps = []
        for disc in self.discriminators:
            out, fmap = disc(x)
            outputs.append(out)
            feature_maps.append(fmap)
        return outputs, feature_maps


# class DiscriminatorS(nn.Module):
#     def __init__(self):
#         super(DiscriminatorS, self).__init__()
#         self.conv_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv1d(1, 128, 15, 1, padding=7),
#                 nn.LeakyReLU(0.1)
#             ),
#             nn.Sequential(
#                 nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
#                 nn.LeakyReLU(0.1)
#             ),
#             nn.Sequential(
#                 nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
#                 nn.LeakyReLU(0.1)
#             ),
#             nn.Sequential(
#                 nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
#                 nn.LeakyReLU(0.1)
#             ),
#             nn.Sequential(
#                 nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
#                 nn.LeakyReLU(0.1)
#             ),
#             nn.Sequential(
#                 nn.Conv1d(1024, 1024, 5, 1, padding=2),
#                 nn.LeakyReLU(0.1)
#             )
#         ])
#         self.final_conv = nn.Conv1d(1024, 1, 3, 1, padding=1)

#     def forward(self, x):
#         fmap = []
#         for layer in self.conv_layers:
#             x = layer(x)
#             fmap.append(x)
#         x = self.final_conv(x)
#         fmap.append(x)
#         x = x.flatten(1, -1)
#         return x, fmap


# class MultiScaleDiscriminator(nn.Module):
#     def __init__(self):
#         super(MultiScaleDiscriminator, self).__init__()
#         self.discriminators = nn.ModuleList([
#             DiscriminatorS(),
#             DiscriminatorS(),
#             DiscriminatorS()
#         ])
#         self.pooling = nn.AvgPool1d(4, 2, padding=2)

#     def forward(self, x):
#         outputs = []
#         feature_maps = []
#         for i, disc in enumerate(self.discriminators):
#             if i != 0:
#                 x_in = self.pooling(x)
#             else:
#                 x_in = x
#             out, fmap = disc(x_in)
#             outputs.append(out)
#             feature_maps.append(fmap)
#         return outputs, feature_maps

import math
from typing import List

class MSDSub(nn.Module):
    def __init__(self,
                 factor: int,
                 kernel_sizes: List[int],
                 strides: List[int],
                 groups: List[int],
                 channels: List[int]):
        super().__init__()
        self.factor = factor

        if factor == 1:
            self.pooling = nn.Identity()
            norm_module = nn.utils.spectral_norm
        else:
            self.pooling = nn.Sequential(
                *[nn.AvgPool1d(kernel_size=4, stride=2, padding=2) for _ in range(int(math.log2(factor)))]
            )
            norm_module = nn.utils.weight_norm

        # Adding first input channel
        channels = [1] + channels

        layers = []

        for i in range(len(kernel_sizes)):
            layers.append(
                nn.Sequential(
                    norm_module(
                        nn.Conv1d(
                            in_channels=channels[i],
                            out_channels=channels[i + 1],
                            kernel_size=kernel_sizes[i],
                            stride=strides[i],
                            groups=groups[i],
                            padding=(kernel_sizes[i] - 1) // 2
                        )
                    ),
                    nn.LeakyReLU()
                )
            )

        layers.append(
            norm_module(
                nn.Conv1d(
                    in_channels=channels[-1],
                    out_channels=1,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        features_from_layers = []
        x = self.pooling(x)
        for layer in self.layers:
            x = layer(x)
            features_from_layers.append(x)
        return x, features_from_layers[:-1]


class MultiScaleDiscriminator(nn.Module):
    def __init__(self,
                 factors = [1, 2, 4],
                 kernel_sizes = [15, 41, 41, 41, 41, 41, 5],
                 strides=[1, 2, 2, 4, 4, 1, 1],
                 groups =[1, 4, 16, 16, 16, 16, 1],
                 channels = [128, 128, 256, 512, 1024, 1024, 1024]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            MSDSub(
                factor=factor,
                kernel_sizes=kernel_sizes,
                strides=strides,
                groups=groups,
                channels=channels
            )
            for factor in factors
        ])

    def forward(self, x):
        disc_outputs = []
        disc_features = []
        for disc in self.discriminators:
            output, features_list = disc(x)
            disc_outputs.append(output)
            disc_features.append(features_list)
        return disc_outputs, disc_features