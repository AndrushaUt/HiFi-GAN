import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorP(nn.Module):
    def __init__(self,
                 period):
        super().__init__()
        self.period = period

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv2d(
                            in_channels=1,
                            out_channels=32,
                            kernel_size=(5, 1),
                            stride=(3, 1),
                            padding=(2, 0)
                        )
                    ),
                    nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv2d(
                            in_channels=32,
                            out_channels=128,
                            kernel_size=(5, 1),
                            stride=(3, 1),
                            padding=(2, 0)
                        )
                    ),
                    nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv2d(
                            in_channels=128,
                            out_channels=512,
                            kernel_size=(5, 1),
                            stride=(3, 1),
                            padding=(2, 0)
                        )
                    ),
                    nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv2d(
                            in_channels=512,
                            out_channels=1024,
                            kernel_size=(5, 1),
                            stride=(3, 1),
                            padding=(2, 0)
                        )
                    ),
                    nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv2d(
                            in_channels=1024,
                            out_channels=1024,
                            kernel_size=(5, 1),
                            padding=(2, 0)
                        )
                    ),
                    nn.LeakyReLU()
                ),
                nn.utils.weight_norm(
                nn.Conv2d(
                    in_channels=1024,
                    out_channels=1,
                    kernel_size=(3, 1),
                    padding="same"
                )
            )
            ]
        )


    def forward(self, x):
        fmp = []
        if x.shape[-1] % self.period > 0:
            x = F.pad(x, (0, self.period - x.shape[-1] % self.period), mode="reflect")
        x = x.reshape(x.shape[0], 1, x.shape[-1] // self.period, self.period)
        for layer in self.layers:
            x = layer(x)
            fmp.append(x)
        return x.flatten(-2, -1), fmp[:-1]


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(
                period=period
            )
            for period in periods
        ])
    def forward(self, x):
        outputs, fmp = [], []
        for disc in self.discriminators:
            output, features_list = disc(x)
            outputs.append(output)
            fmp.append(features_list)
        return outputs, fmp

class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            in_channels=1,
                            out_channels=128,
                            kernel_size=15,
                            stride=2,
                            groups=1,
                            padding=7
                        )
                    ),
                    nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            in_channels=128,
                            out_channels=128,
                            kernel_size=41,
                            stride=2,
                            groups=4,
                            padding=20
                        )
                    ),
                    nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            in_channels=128,
                            out_channels=256,
                            kernel_size=41,
                            stride=2,
                            groups=16,
                            padding=20
                        )
                    ),
                    nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            in_channels=256,
                            out_channels=512,
                            kernel_size=41,
                            stride=4,
                            groups=16,
                            padding=20
                        )
                    ),
                    nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            in_channels=512,
                            out_channels=1024,
                            kernel_size=41,
                            stride=4,
                            groups=16,
                            padding=20
                        )
                    ),
                    nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            in_channels=1024,
                            out_channels=1024,
                            kernel_size=41,
                            stride=1,
                            groups=16,
                            padding=20
                        )
                    ),
                    nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            in_channels=1024,
                            out_channels=1024,
                            kernel_size=5,
                            stride=1,
                            groups=1,
                            padding=2
                        )
                    ),
                    nn.LeakyReLU()
                ),
                nn.Sequential(
                    nn.utils.weight_norm(
                        nn.Conv1d(
                            in_channels=1024,
                            out_channels=1,
                            kernel_size=3,
                            stride=1,
                            groups=1,
                            padding=1
                        )
                    )
                )
            ]
        )


    def forward(self, x):
        fmp = []
        for layer in self.layers:
            x = layer(x)
            fmp.append(x)
        return x, fmp[:-1]


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
        self.discriminators = nn.ModuleList([
            DiscriminatorS(),
            DiscriminatorS(),
            DiscriminatorS()
        ])

    def forward(self, x):
        outputs, fmp = [], []
        for i in range(len(self.discriminators)):
            temp_x = x
            if i == 1:
                temp_x = self.pool(temp_x)
            elif i == 2:
                temp_x = self.pool(self.pool(temp_x))
            output, features = self.discriminators[i](temp_x)
            outputs.append(output)
            fmp.append(features)
        return outputs, fmp