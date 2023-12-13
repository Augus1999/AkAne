# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Akane encoder modules
"""
from typing import Tuple, Dict
import torch
import torch.nn as nn
from torch import Tensor


class GraphEncoder(nn.Module):
    def __init__(self, channel: int = 512) -> None:
        """
        Edge-aware graph convolution as a graph structure encoding.

        :param channel: hidden layer features
        """
        super().__init__()
        self.w = nn.Sequential(nn.Linear(channel * 2, channel), nn.SELU())
        self.filter_net = nn.Sequential(nn.Linear(30, channel), nn.SELU())

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
        h1 = (h[:, None, :, :] * e).sum(dim=-2)
        h1 = self.w(torch.cat([h, h1], dim=-1)) * node_mask
        return h1 + h  # skip connect


class EncoderAttention(nn.Module):
    def __init__(
        self, channel: int = 512, num_head: int = 8, temperature_coeff: float = 2.0
    ) -> None:
        """
        Multi-head self-attention block.

        :param channel: hidden layer features
        :param num_head: number of heads
        :param temperature_coeff: attention temperature coefficient
        """
        super().__init__()
        assert (
            channel % num_head == 0
        ), f"cannot split {num_head} heads when the feature is {channel}."
        self.d = channel // num_head  # head dimension
        self.nh = num_head  # number of heads
        self.tp = (temperature_coeff * self.d) ** 0.5  # attention temperature
        self.q = nn.Linear(in_features=channel, out_features=channel, bias=False)
        self.k = nn.Linear(in_features=channel, out_features=channel, bias=False)
        self.v = nn.Linear(in_features=channel, out_features=channel, bias=False)
        self.activate = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Tensor, adjacency_matrix: Tensor) -> Tensor:
        """
        :param x: input tensor;                     shape: (n_b, n_a, n_f)
        :param mask: attention mask;                shape: (1, n_b, n_a, n_a)
        :param adjacency_matrix: adjacency matrix;  shape: (1, n_b, n_a, n_a)
        :return: attentioned output;                shape: (n_b, n_a, n_f)
        """
        n_b, n_a, _ = x.shape
        q = self.q(x).view(n_b, n_a, self.nh, self.d).permute(2, 0, 1, 3).contiguous()
        k_t = self.k(x).view(n_b, n_a, self.nh, self.d).permute(2, 0, 3, 1).contiguous()
        v = self.v(x).view(n_b, n_a, self.nh, self.d).permute(2, 0, 1, 3).contiguous()
        a = q @ k_t / self.tp + adjacency_matrix
        alpha = self.activate(a.masked_fill(mask, -torch.inf))
        atten_out = (alpha @ v).permute(1, 2, 0, 3).contiguous().view(n_b, n_a, _)
        return atten_out + x  # skip connect


class EncoderLayer(nn.Module):
    def __init__(
        self, channel: int = 512, num_head: int = 8, temperature_coeff: float = 2.0
    ) -> None:
        """
        Encoder layer.

        :param channel: hidden layer features
        :param num_head: number of heads
        :param temperature_coeff: attention temperature coefficient
        """
        super().__init__()
        self.graph_encoder = GraphEncoder(channel=channel)
        self.attention = EncoderAttention(channel, num_head, temperature_coeff)

    def forward(
        self,
        h: Tensor,
        e: Tensor,
        node_mask: Tensor,
        edge_mask: Tensor,
        attention_mask: Tensor,
        adjacency_matrix: Tensor,
    ) -> Tensor:
        """
        :param h: node hidden state;                shape: (n_b, n_a, n_f)
        :param e: edge hidden state;                shape: (n_b, n_a, n_a, n_f)
        :param node_mask: node masks;               shape: (n_b, n_a, 1)
        :param edge_mask: edge masks;               shape: (n_b, n_a, n_a, 1)
        :param attention_mask: attention mask;      shape: (1, n_b, n_a, n_a)
        :param adjacency_matrix: adjacency matrix;  shape: (1, n_b, n_a, n_a)
        :return: new node hidden state;             shape: (n_b, n_a, n_f)
        """
        h = self.graph_encoder(h, e, node_mask, edge_mask)
        h = self.attention(h, attention_mask, adjacency_matrix)
        return h


class Encoder(nn.Module):
    def __init__(
        self,
        channel: int = 512,
        n_layer: int = 6,
        num_head: int = 8,
        temperature_coeff: float = 2.0,
    ) -> None:
        """
        Encoder block.

        :param channel: hidden layer features
        :param n_layer: number of encoder layers
        :param num_head: number of heads
        :param temperature_coeff: attention temperature coefficient
        """
        super().__init__()
        self.embedding = nn.Linear(in_features=62, out_features=channel, bias=False)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(channel, num_head, temperature_coeff) for _ in range(n_layer)]
        )
        self.mu = nn.Linear(channel, channel, False)
        self.sigma = nn.Linear(channel, channel, False)
        self.kl = torch.empty((1, 1))

    def forward(
        self, mol: Dict[str, Tensor], skip_kl: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        :param mol: molecule = {
            "node: node matrix;   shape: (n_b, n_a, 62)
            "edge: edge matrix;   shape: (n_b, n_a, n_a, 30)
        }
        :param skip_kl: whether to skip latent space regularisation
        :return: encoder latent vectors;  shape: (n_b, n_a, n_f)
                 node mask;               shape: (n_b, n_a, 1)
        """
        h, e = mol["node"], mol["edge"]
        node_mask = (h.sum(dim=-1) != 0).float()[:, :, None]
        edge_mask = (e.sum(dim=-1) != 0).float()[:, :, :, None]
        attention_mask = (node_mask + node_mask.transpose(-2, -1) == 1)[None, :, :, :]
        adjacency = edge_mask.squeeze(dim=-1)[None, :, :, :]
        h = self.embedding(h)
        for layer in self.encoder_layers:
            h = layer(h, e, node_mask, edge_mask, attention_mask, adjacency)
        if skip_kl:
            return h, node_mask
        mu, sigma = self.mu(h), self.sigma(h).exp()
        z = (mu + sigma * torch.randn_like(h)) * node_mask
        if self.training and self.embedding.weight.requires_grad:
            kl = sigma.pow(2) * node_mask + mu.pow(2) - sigma.log() - 0.5
            self.kl = kl.mean()
        return z, node_mask


if __name__ == "__main__":
    ...
