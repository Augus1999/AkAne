# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Akane readout modules
"""
import torch.nn as nn
from torch import Tensor


class Readout(nn.Module):
    def __init__(self, channel: int = 512, n_layer: int = 2, n_task: int = 1):
        """
        Readout block.

        :param channel: hidden layer features
        :param n_layer: number of layers
        :param n_task: number of output tasks
        """
        super().__init__()
        self.mlp = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(channel, channel), nn.SELU())
                for _ in range(n_layer)
            ]
        )
        self.out = nn.Linear(channel, n_task)

    def forward(self, x: Tensor) -> Tensor:
        x = x.sum(dim=-2)  # sum pooling
        for layer in self.mlp:
            x = layer(x)
        return self.out(x)


if __name__ == "__main__":
    ...
