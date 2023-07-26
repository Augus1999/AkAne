# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Akane decoder modules
"""
from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoder(nn.Module):
    def __init__(self, channel: int = 512) -> None:
        super().__init__()
        self.d = channel

    def forward(self, size: int) -> Tensor:
        pos = torch.arange(size)[:, None]
        pe = pos / torch.pow(10000, 2.0 * torch.arange(self.d)[None, :] / self.d)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe[None, :, :]  # shape: (1, n_t, n_f)


class DecoderAttention(nn.Module):
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
        self.norm = nn.LayerNorm(channel)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        :param x: input tensor;       shape: (n_b, n_t, n_f)
        :param mask: attention mask;  shape: (1, n_b, n_t, n_t)
        :return: attentioned output;  shape: (n_b, n_t, n_f)
        """
        n_b, n_a, _ = x.shape
        split = (n_b, n_a, self.nh, self.d)
        shape = x.shape
        q = self.q(x).view(split).permute(2, 0, 1, 3).contiguous()
        k_t = self.k(x).view(split).permute(2, 0, 3, 1).contiguous()
        v = self.v(x).view(split).permute(2, 0, 1, 3).contiguous()
        a = q @ k_t / self.tp
        alpha = self.activate(a.masked_fill(mask, -torch.inf))
        atten_out = (alpha @ v).permute(1, 2, 0, 3).contiguous().view(shape)
        return self.norm(atten_out + x)  # skip connect


class CrossAttention(nn.Module):
    def __init__(
        self, channel: int = 512, num_head: int = 8, temperature_coeff: float = 2.0
    ) -> None:
        """
        Multi-head cross-attention block.

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
        self.norm = nn.LayerNorm(channel)

    def forward(self, x: Tensor, y: Tensor, mask: Tensor, x_mask: Tensor) -> Tensor:
        """
        :param x: input tensor;       shape: (n_b, n_t, n_f)
        :param y: input tensor;       shape: (n_b, n_a, n_f)
        :param mask: attention mask;  shape: (1, n_b, n_t, n_a, 1)
        :param x_mask: x mask;        shape: (n_b, n_t, 1)
        :return: attentioned output;  shape: (n_b, n_t, n_f)
        """
        (n_b, n_x, _), n_y = x.shape, y.shape[-2]
        xsplit = (n_b, n_x, self.nh, self.d)
        ysplit = (n_b, n_y, self.nh, self.d)
        xshape = x.shape
        q = self.q(x).view(xsplit).permute(2, 0, 1, 3).contiguous()
        k_t = self.k(y).view(ysplit).permute(2, 0, 3, 1).contiguous()
        v = self.v(y).view(ysplit).permute(2, 0, 1, 3).contiguous()
        a = q @ k_t / self.tp
        alpha = self.activate(a.masked_fill(mask, -torch.inf))
        atten_out = (alpha @ v).permute(1, 2, 0, 3).contiguous().view(xshape)
        return self.norm(atten_out + x) * x_mask  # skip connect


class DecoderLayer(nn.Module):
    def __init__(
        self, channel: int = 512, num_head: int = 8, temperature_coeff: float = 2.0
    ) -> None:
        """
        Decoder layer.

        :param channel: hidden layer features
        :param num_head: number of heads
        :param temperature_coeff: attention temperature coefficient
        """
        super().__init__()
        self.attention = DecoderAttention(channel, num_head, temperature_coeff)
        self.cross = CrossAttention(channel, num_head, temperature_coeff)
        self.ffn = nn.Sequential(
            nn.Linear(channel, channel * 4), nn.SELU(), nn.Linear(channel * 4, channel)
        )

    def forward(
        self, x: Tensor, y: Tensor, ca_mask: Tensor, sa_mask: Tensor, mask: Tensor
    ) -> Tensor:
        """
        :param x: input tensor;                 shape: (n_b, n_t, n_f)
        :param y: input tensor;                 shape: (n_b, n_a, n_f)
        :param ca_mask: cross-attention masks;  shape: (1, n_b, n_t, n_a, 1)
        :param sa_mask: self-attention mask;    shape: (1, n_b, n_a, n_a)
        :param mask: x mask;                    shape: (n_b, n_t, 1)
        :return: new node hidden state;         shape: (n_b, n_t, n_f)
        """
        x = self.attention(x, sa_mask)
        x = self.cross(x, y, ca_mask, mask)
        x = self.ffn(x) * mask + x  # skip connect
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        channel: int = 512,
        n_layer: int = 6,
        num_head: int = 8,
        temperature_coeff: float = 2.0,
    ) -> None:
        """
        Encoder layer.

        :param n_vocab: number of vocabulary
        :param channel: hidden layer features
        :param n_layer: number of encoder layers
        :param num_head: number of heads
        :param temperature_coeff: attention temperature coefficient
        """
        super().__init__()
        self.embedding = nn.Embedding(n_vocab, channel)
        self.position = PositionalEncoder(channel)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(channel, num_head, temperature_coeff) for _ in range(n_layer)]
        )
        self.out = nn.Linear(channel, n_vocab)

    def forward(self, x: Tensor, y: Tensor, y_mask: Tensor) -> Tuple[Tensor]:
        """
        :param x: input tokens;                              shape: (n_b, n_t)
        :param y: node representation from encoder;          shape: (n_b, n_a, n_f)
        :param y_mask: node mask;                            shape: (n_b, n_a, 1)
        :return: probability distribution (before softmax);  shape: (n_b, n_t, n_vocab)
                 token mask;                                 shape: (n_b, n_t, 1)
        """
        n_t = x.shape[1]
        x_mask = (x != 0).float()[:, :, None]
        sa_mask = (x_mask + x_mask.transpose(-2, -1) == 1).float()
        ca_mask = (y_mask.transpose(-2, -1) == 0)[None, :, :, :]
        ca_mask = ca_mask.repeat(1, 1, n_t, 1)
        mask = x_mask + x_mask.transpose(-2, -1) == 0
        look_ahead = torch.triu(torch.ones_like(sa_mask), diagonal=1)
        look_ahead = look_ahead.masked_fill(mask, 0)
        look_ahead = ((look_ahead + sa_mask) != 0)[None, :, :, :]
        x_pe = self.position(n_t).to(x.device)
        y_pe = self.position(y.shape[1]).to(x.device)
        x = (self.embedding(x) + x_pe) * x_mask
        y = (y + y_pe) * y_mask
        for layer in self.decoder_layers:
            x = layer(x, y, ca_mask, look_ahead, x_mask)
        z = self.out(x) * x_mask
        return z, x_mask


if __name__ == "__main__":
    ...
