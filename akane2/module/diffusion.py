# -*- coding: utf-8 -*-
# Author: Nianze A. Tao (Omozawa Sueno)
"""
Akane diffusion modules
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TimeEmbedding(nn.Module):
    def __init__(self, channel: int = 512) -> None:
        super().__init__()
        self.d = channel

    def forward(self, time: Tensor) -> Tensor:
        """
        :param time: time;      shape: (n_b, 1)
        :return: embedded vec;  shape: (n_b, 1, n_f)
        """
        base = torch.pow(10000, 2.0 * torch.arange(self.d)[None, :] / self.d)
        te = time / base.to(time.device)
        te[:, 0::2] = torch.sin(te[:, 0::2])
        te[:, 1::2] = torch.cos(te[:, 1::2])
        return te[:, None, :]


class ValueLabelEmbed(nn.Module):
    def __init__(self, channel: int = 512, max_len: int = 2) -> None:
        super().__init__()
        self.embed = nn.Linear(max_len, channel)

    def forward(self, label: Tensor) -> Tensor:
        """
        :param label: label values;  shape: (n_b, n_l)
        :return: embedded labels;    shape: (n_b, 1, n_f)
        """
        return self.embed(label)[:, None, :]


class ClassLabelEmbed(nn.Module):
    def __init__(self, channel: int = 512, max_len: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(max_len, channel)

    def forward(self, label: Tensor) -> Tensor:
        """
        :param label: label tokens;  shape: (n_b, 1)
        :return: embedded labels;    shape: (n_b, 1, n_f)
        """
        return self.embed(label.to(torch.long))


class TextLabelEmbed(nn.Module):
    def __init__(self, channel: int = 512, max_len: int = 23) -> None:
        super().__init__()
        self.d = channel
        self.embed = nn.Embedding(max_len, channel)

    def forward(self, label: Tensor) -> Tensor:
        """
        :param label: label tokens;  shape: (n_b, n_t)
        :return: embedded labels;    shape: (n_b, 1, n_f)
        """
        mask = (label != 0).float()[:, :, None]
        le = self.embed(label)
        le += self.position(label.shape[1]).to(label.device)
        return (mask * le).sum(dim=-2, keepdim=True)

    def position(self, size: int) -> Tensor:
        pos = torch.arange(size)[:, None]
        pe = pos / torch.pow(10000, 2.0 * torch.arange(self.d)[None, :] / self.d)
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe[None, :, :]  # shape: (1, n_t, n_f)


#################################### DiT ###########################################


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class Attention(nn.Module):
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

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        :param x: input tensor;       shape: (n_b, n_a, n_f)
        :param mask: attention mask;  shape: (1, n_b, n_a, n_l, 1)
        :return: attentioned output;  shape: (n_b, n_a, n_f)
        """
        xshape = x.shape
        xsplit = (xshape[0], xshape[1], self.nh, self.d)
        q = self.q(x).view(xsplit).permute(2, 0, 1, 3).contiguous()
        k_t = self.k(x).view(xsplit).permute(2, 0, 3, 1).contiguous()
        v = self.v(x).view(xsplit).permute(2, 0, 1, 3).contiguous()
        a = q @ k_t / self.tp
        alpha = self.activate(a.masked_fill(mask, -torch.inf))
        atten_out = (alpha @ v).permute(1, 2, 0, 3).contiguous().view(xshape)
        return atten_out


class DiTBlock(nn.Module):
    def __init__(
        self, channel: int = 512, num_head: int = 8, temperature_coeff: float = 2.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channel, elementwise_affine=False, eps=1e-6)
        self.attention = Attention(channel, num_head, temperature_coeff)
        self.norm2 = nn.LayerNorm(channel, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(channel, channel * 4), nn.SELU(), nn.Linear(channel * 4, channel)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SELU(), nn.Linear(channel, 6 * channel)
        )
        # zero-out adaLN layer
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor, mask: Tensor, x_mask: Tensor) -> Tensor:
        c = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = c.chunk(6, -1)
        x = x + gate_msa * self.attention(
            modulate(self.norm1(x), shift_msa, scale_msa) * x_mask, mask
        )
        x = x + gate_ffn * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))
        return x * x_mask


class FinalLayer(nn.Module):
    def __init__(self, channel: int = 512) -> None:
        """
        The final layer of DiT.

        :param channel: hidden layer features
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(channel, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(channel, channel)
        self.adaLN_modulation = nn.Sequential(
            nn.SELU(), nn.Linear(channel, 2 * channel)
        )
        # zero-out this layer
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        channel: int = 512,
        n_layer: int = 6,
        num_head: int = 8,
        temperature_coeff: float = 2.0,
        label_mode: str = "value:2",
    ) -> None:
        """
        Scalable Diffusion Transformer.

        :param channel: hidden layer features
        :param n_layer: number of DiT layers
        :param num_head: number of attention head(s)
        :param temperature_coeff: attention temperature coefficient
        :param label_mode: label mode chosen from 'value:x', 'class:x' and 'text:x'
        """
        super().__init__()
        mode, max_len = label_mode.split(":")
        if mode == "value":
            self.label = ValueLabelEmbed(channel, int(max_len))
        elif mode == "text":
            self.label = TextLabelEmbed(channel, int(max_len))
        elif mode == "class":
            self.label = ClassLabelEmbed(channel, int(max_len))
        else:
            raise NotImplementedError
        self.time = TimeEmbedding(channel)
        self.dit_layers = nn.ModuleList(
            [DiTBlock(channel, num_head, temperature_coeff) for _ in range(n_layer)]
        )
        self.ffn = FinalLayer(channel)

    def forward(self, x: Tensor, time: Tensor, label: Tensor, x_mask: Tensor) -> Tensor:
        lemb = self.label(label)
        temb = self.time(time)
        c = lemb + temb
        sa_mask = (x_mask + x_mask.transpose(-2, -1) == 1).unsqueeze(0)
        for layer in self.dit_layers:
            x = layer(x, c, sa_mask, x_mask)
        return self.ffn(x, c) * x_mask


################################### diffusion method ########################################


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps: int = 1000) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.betas = self.linear_beta_schedule()

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )

        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
        self.reversed_t_range = torch.flip(
            torch.linspace(0, timesteps - 1, timesteps, dtype=torch.int64), (0,)
        )

    def linear_beta_schedule(self) -> Tensor:
        """
        compute betas
        """
        scale = 1000 / self.timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)

    # get the param of given timestep t
    def _extract(self, a: Tensor, t: Tensor) -> Tensor:
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, 1, 1)
        return out

    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(
        self, x_start: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mean = self._extract(self.sqrt_alphas_cumprod, t) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t)
        return mean, variance, log_variance

    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(
        self, x_start: Tensor, x_t: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t) * x_start
            + self._extract(self.posterior_mean_coef2, t) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t) * noise
        )

    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(
        self, model: DiT, x_t: Tensor, t: Tensor, y: Tensor, x_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # predict noise using model
        pred_noise = model.forward(x_t, t, y, x_mask)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t.reshape(-1), pred_noise)
        x_recon = torch.clamp(x_recon, min=-1.0, max=1.0)
        return self.q_posterior_mean_variance(x_recon, x_t, t.reshape(-1))

    # denoise_step: sample x_{t-1} from x_t and pred_noise
    def p_sample(
        self, model: DiT, x_t: Tensor, t: Tensor, y: Tensor, x_mask: Tensor
    ) -> Tensor:
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(
            model, x_t, t, y, x_mask
        )
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, 1, 1)
        # compute x_{t-1}
        pred_x = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_x

    # denoise: reverse diffusion
    def sample(self, model: DiT, x: Tensor, x_mask: Tensor, y: Tensor) -> Tensor:
        # start from pure noise
        for i in self.reversed_t_range:
            x = self.p_sample(model, x, i.view(1, 1).to(x.device), y, x_mask)
        return x

    # compute train losses
    def train_losses(
        self, model: DiT, x_start: Tensor, t: Tensor, y: Tensor, x_mask: Tensor
    ) -> Tensor:
        # generate random noise
        noise = torch.randn_like(x_start) * x_mask
        # get x_t
        x_noisy = self.q_sample(x_start, t.reshape(-1), noise)
        predicted_noise = model.forward(x_noisy, t, y, x_mask)
        loss = (noise - predicted_noise).pow(2).sum()
        return loss / x_mask.repeat(1, 1, x_start.shape[-1]).sum()


if __name__ == "__main__":
    ...
