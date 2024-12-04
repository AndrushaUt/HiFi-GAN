import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResBlock, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, padding=dil*(kernel_size-1)//2, dilation=dil),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size-1)//2)
            ) for dil in dilation
        ])

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer(x) + residual
        return x
