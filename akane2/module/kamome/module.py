# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Kamome modules
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor


class EFConv(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        """
        Edge-filtering convolution block.

        :param hidden_dim: hidden layer features
        """
        super().__init__()
        self.w = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim * 2,
                out_features=hidden_dim,
                bias=True,
            ),
            nn.SELU(),
        )
        self.filter_net = nn.Sequential(
            nn.Linear(
                in_features=30,
                out_features=hidden_dim,
                bias=True,
            ),
            nn.SELU(),
        )

    def forward(
        self, h: Tensor, e: Tensor, node_mask: Tensor, edge_mask: Tensor
    ) -> Tensor:
        """
        :param h: node hidden state;     shape: (n_b, n_a, n_f)
        :param e: edge hidden state;     shape: (n_b, n_a, n_a, n_f)
        :param node_mask: node masks;    shape: (n_b, n_a, 1)
        :param edge_mask: edge masks;    shape: (n_b, n_a, n_a, 1)
        :return: new node hidden state;  shape: (n_b, n_a, n_f)
        """
        e = self.filter_net(e) * edge_mask
        h1 = (h.unsqueeze(dim=-3) * e).sum(dim=-2)
        h1 = self.w(torch.cat([h, h1], dim=-1)) * node_mask
        return h1 + h  # skip connect


class MultiHeadAttention(nn.Module):
    def __init__(
        self, hidden_dim: int = 256, num_head: int = 8, temperature_coeff: float = 2.0
    ) -> None:
        """
        Multi-head self-attention block.

        :param hidden_dim: hidden layer features
        :param num_head: number of heads
        :param temperature_coeff: attention temperature coefficient
        """
        super().__init__()
        assert (
            hidden_dim % num_head == 0
        ), f"cannot split {num_head} heads when the feature is {hidden_dim}."
        self.f = hidden_dim  # output dimension
        self.d = hidden_dim // num_head  # head dimension
        self.nh = num_head  # number of heads
        self.tp = (temperature_coeff * self.d) ** 0.5  # attention temperature
        self.q = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.k = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.v = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.activate = nn.Softmax(dim=-1)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        adjacency_matrix: Optional[Tensor],
        return_attn_matrix: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param x: input tensor;                     shape: (n_b, n_a, n_f)
        :param mask: attention mask;                shape: (1, n_b, n_a, n_a)
        :param adjacency_matrix: adjacency matrix;  shape: (1, n_b, n_a, n_a)
        :param return_attn_matrix: whether to return the averaged attention matrix over heads
        :return: attentioned output;                shape: (n_b, n_a, n_f)
                 attention matrix;                  shape: (1, n_b, n_a, n_a)
        """
        #-- NOTE: committed code is the second way to embed ratio and temperature --
        n_b, n_a, _ = x.shape
        q = self.q(x).view(n_b, n_a, self.nh, self.d).permute(2, 0, 1, 3).contiguous()
        k_t = self.k(x).view(n_b, n_a, self.nh, self.d).permute(2, 0, 3, 1).contiguous()
        v = self.v(x).view(n_b, n_a, self.nh, self.d).permute(2, 0, 1, 3).contiguous()
        a = q @ k_t / self.tp
        # if freq_matrix is not None:
        #     a += freq_matrix
        if adjacency_matrix is not None:
            a += adjacency_matrix
        alpha = self.activate(a.masked_fill(mask, -torch.inf))
        atten_out = (alpha @ v).permute(1, 2, 0, 3).contiguous().view(n_b, n_a, self.f)
        if return_attn_matrix:
            return atten_out + x, alpha.mean(dim=0, keepdim=True)
        return atten_out + x, None  # skip connect


class Interaction(nn.Module):
    def __init__(
        self, hidden_dim: int = 256, num_head: int = 8, temperature_coeff: float = 2.0
    ) -> None:
        """
        Interaction block.

        :param hidden_dim: hidden layer features
        :param num_head: number of heads
        :param temperature_coeff: attention temperature coefficient
        """
        super().__init__()
        self.conv = EFConv(hidden_dim=hidden_dim)
        self.attention = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_head=num_head,
            temperature_coeff=temperature_coeff,
        )

    def forward(
        self,
        h: Tensor,
        e: Tensor,
        node_mask: Tensor,
        edge_mask: Tensor,
        attention_mask,
        adjacency_matrix: Optional[Tensor],
        return_attn_matrix: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param h: node hidden state;                shape: (n_b, n_a, n_f)
        :param e: edge hidden state;                shape: (n_b, n_a, n_a, n_f)
        :param node_mask: node masks;               shape: (n_b, n_a, 1)
        :param edge_mask: edge masks;               shape: (n_b, n_a, n_a, 1)
        :param attention_mask: attention mask;      shape: (1, n_b, n_a, n_a)
        :param adjacency_matrix: adjacency matrix;  shape: (1, n_b, n_a, n_a)
        :param return_adj_matrix: whether to return the attention matrices
        :return: new node hidden state;             shape: (n_b, n_a, n_f)
                 attention matrix;                  shape: (1, n_b, n_a, n_a)
        """
        h = self.conv(h, e, node_mask, edge_mask)
        h, attn = self.attention(
            h, attention_mask, adjacency_matrix, return_attn_matrix
        )
        return h, attn


class MLP(nn.Module):
    def __init__(self, hidden_dim: int = 256, n_layer: int = 2):
        """
        MLP block.

        :param hidden_dim: hidden layer features
        :param n_layer: number of layers
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        in_features=hidden_dim, out_features=hidden_dim, bias=True
                    ),
                    nn.SELU(),
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    ...
