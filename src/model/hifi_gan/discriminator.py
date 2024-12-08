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


class DiscriminatorS(nn.Module):
    def __init__(self):
        super(DiscriminatorS, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 128, 15, 1, padding=7),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
                nn.LeakyReLU(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(1024, 1024, 5, 1, padding=2),
                nn.LeakyReLU(0.1)
            )
        ])
        self.final_conv = nn.Conv1d(1024, 1, 3, 1, padding=1)

    def forward(self, x):
        fmap = []
        for layer in self.conv_layers:
            x = layer(x)
            fmap.append(x)
        x = self.final_conv(x)
        fmap.append(x)
        x = x.flatten(1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS()
        ])
        self.pooling = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x):
        outputs = []
        feature_maps = []
        for i, disc in enumerate(self.discriminators):
            if i != 0:
                x_in = self.pooling(x)
            else:
                x_in = x
            out, fmap = disc(x_in)
            outputs.append(out)
            feature_maps.append(fmap)
        return outputs, feature_maps