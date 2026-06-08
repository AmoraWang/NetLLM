"""
Merina 风格带宽 β-VAE 编码器（仅 encoder + reparameterize，无 decoder）。

从 ABRLLM Pensieve 状态 ``(…, 6, 6)`` 的第 2/3 行（吞吐、下载时延历史）提取
``(hist_len, 2)`` 轨迹，压成 ``latent_dim`` 维随机网络条件向量。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def extract_bandwidth_trajectory(state: torch.Tensor) -> torch.Tensor:
    """
    Args:
        state: (B, T, 6, 6) 或 (B, 6, 6)
    Returns:
        (B*T, hist_len, 2)  通道 0=吞吐(row2)，1=下载时延(row3)
    """
    if state.dim() == 3:
        state = state.unsqueeze(1)
    batch_size, seq_len, _, hist_len = state.shape[0], state.shape[1], state.shape[2], state.shape[3]
    throughput = state[..., 2, :]   # (B, T, H)
    delay = state[..., 3, :]      # (B, T, H)
    traj = torch.stack([throughput, delay], dim=-1)  # (B, T, H, 2)
    return traj.reshape(batch_size * seq_len, hist_len, 2)


class BandwidthVAE(nn.Module):
    """
    双通道 Conv1d 编码 + Gaussian latent（Merina ``beta_vae_v6_light`` 结构）。

    ``hist_len=6`` 对应 ABRLLM 状态历史长度；Merina 原实现为 8。
    """

    def __init__(
        self,
        *,
        hist_len: int = 6,
        latent_dim: int = 16,
        fe_channels: int = 128,
        conv_kernel: int = 4,
        beta: float = 0.4,
    ):
        super().__init__()
        self.hist_len = hist_len
        self.latent_dim = latent_dim
        self.beta = beta
        conv_out_len = hist_len - conv_kernel + 1
        if conv_out_len <= 0:
            raise ValueError(f"hist_len={hist_len} 须大于 conv_kernel={conv_kernel}")
        flat_dim = 2 * fe_channels * conv_out_len

        self.encoder_tp = nn.Sequential(
            nn.Conv1d(1, fe_channels, conv_kernel),
            nn.LeakyReLU(),
        )
        self.encoder_dl = nn.Sequential(
            nn.Conv1d(1, fe_channels, conv_kernel),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def encode(self, trajectory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            trajectory: (N, hist_len, 2)
        """
        tp = trajectory[:, :, 0:1].transpose(1, 2)   # (N, 1, H)
        dl = trajectory[:, :, 1:2].transpose(1, 2)
        h1 = self.encoder_tp(tp).flatten(1)
        h2 = self.encoder_dl(dl).flatten(1)
        hidden = torch.cat([h1, h2], dim=1)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, *, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """标准高斯先验下的 mean β-KLD（与 Merina loss_function 一致）。"""
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return self.beta * kld.mean()

    def forward(
        self,
        state: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, T, 6, 6)
        Returns:
            z: (B, T, latent_dim)
            kl: scalar
        """
        traj = extract_bandwidth_trajectory(state)
        mu, logvar = self.encode(traj)
        z = self.reparameterize(mu, logvar, sample=sample)
        batch_size, seq_len = state.shape[0], state.shape[1]
        z = z.view(batch_size, seq_len, self.latent_dim)
        kl = self.kl_loss(mu, logvar)
        return z, kl
