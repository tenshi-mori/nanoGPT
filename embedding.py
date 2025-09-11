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
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()

        self.base = base
        self.dim = dim
        self.cache = None

    def _build_cache(self, t: int) -> None:
        if self.cache is not None and t <= self.cache.shape[1]:
            return 

        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        seq_idx = torch.arange(t).float()
        idx_theta = einops.einsum(seq_idx, theta, "s, d_half -> s d_half")

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.cache = cache[None, :, None, :, :]

    def forward(self, x: Float[Tensor, "b t nh dh"]) -> Float[Tensor, "b t nh dh"]:
        t = x.shape[1]
        self._build_cache(x.shape[1])
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = self.cache[:, :t, :, :, :]

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out = x_out.flatten(3)
        return x_out.type_as(x)

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

    