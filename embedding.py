import math
from tkinter import S
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from torchtune.modules import RotaryPositionalEmbeddings

import einops
from jaxtyping import Float

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d: int, base: int = 10000):
        super().__init__()

        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, t: int) -> None:
        if self.cos_cached is not None and t <= self.cos_cached.shape[1]:
            return 

        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d))
        seq_idx = torch.arange(t).float()
        idx_theta = einops.einsum(seq_idx, theta, "s, d_half -> s d_half")
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cached = idx_theta2.cos()[None, :, None, :]
        self.sin_cached = idx_theta2.sin()[None, :, None, :]

    def _neg_half(self, x: Float[Tensor, "b t nh dh"]) -> Float[Tensor, "b t nh dh"]:
        d_2 = self.d // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim = -1)

    def forward(self, x: Float[Tensor, "b t nh dh"]) -> Float[Tensor, "b t nh dh"]:
        t = x.shape[1]
        self._build_cache(x.shape[1])
        neg_half_x = self._neg_half(x)
        x_rope = (x * self.cos_cached[:, :t, :, :]) + (neg_half_x * self.sin_cached[:, :t, :, :])
        return x_rope

if __name__ == "__main__":
    # Test RotaryPositionalEmbedding
    d_model = 64
    batch = 4
    seq_len = 128
    n_heads = 8
    d_head = 8


    # rope = RotaryPositionalEmbedding(d_model)

    x = torch.randn(batch, seq_len, n_heads, d_head, dtype=float)

    test_RoPE = RotaryPositionalEmbeddings(d_head)
    RoPE = RotaryPositionalEmbedding(d_head)

    out = RoPE(x)
    test_out = test_RoPE(x)
    print(out.shape)
    print(test_out.shape)
    torch.testing.assert_close(test_out, out)
    print("Passed Assertion test against Pytorch RoPE")

    