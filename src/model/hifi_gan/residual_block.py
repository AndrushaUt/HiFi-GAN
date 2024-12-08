import torch.nn as nn


class ResBlock(nn.Module):    
    def __init__(self, num_channels, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dilations)):
            inner_layers =nn.Sequential(
                        nn.LeakyReLU(),
                        nn.utils.weight_norm(
                            nn.Conv1d(
                                in_channels=num_channels,
                                out_channels=num_channels,
                                kernel_size=kernel_size,
                                dilation=dilations[i],
                                padding="same"
                            )
                        ),
                        nn.LeakyReLU(),
                        nn.utils.weight_norm(
                            nn.Conv1d(
                                in_channels=num_channels,
                                out_channels=num_channels,
                                kernel_size=kernel_size,
                                dilation=1,
                                padding="same"
                            )
                        )
                    )
            self.layers.append(inner_layers)

    def forward(self, x):
        for layer in self.layers:
            temp = x
            x = layer(x)
            x = x + temp
        return x
        

class MRF(nn.Module):
    def __init__(self,num_channels, kernel_sizes, dilations):
        super().__init__()
        self.layers = nn.ModuleList([
            ResBlock(num_channels, kernel_sizes[i], dilations
            )
            for i in range(len(kernel_sizes))
        ])

    def forward(self, x):
        result = self.layers[0](x)
        for block in self.layers[1:]:
            result += block(x)
        return result