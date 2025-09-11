import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from torchtune.modules import RotaryPositionalEmbeddings

import einops
from jaxtyping import Float
'''
Implementation based off of mine and Pytorch's. Still haven't figured out why my implementation is slightly different from this. 
'''
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
        idx_theta: Float[Tensor, "s d_half"] = einops.einsum(seq_idx, theta, "s, d_half -> s d_half")

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.cache: Float[Tensor, "1 t 1 dh//2 2"] = cache[None, :, None, :, :]

    def forward(self, x: Float[Tensor, "b t nh dh"]) -> Float[Tensor, "b t nh dh"]:
        t = x.shape[1]
        self._build_cache(t)
        xshaped: Float[Tensor, "b t nh dh//2 2"] = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache: Float[Tensor, "1 t 1 dh//2 2"] = self.cache[:, :t, :, :, :]

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out: Float[Tensor, "b t nh dh"] = einops.rearrange(x_out, "b t nh ... -> b t nh (...)")
        return x_out.type_as(x)

if __name__ == "__main__":
    # Test RotaryPositionalEmbedding
    d_model = 64
    batch = 4
    seq_len = 128
    n_heads = 8
    d_head = 8

    x = torch.randn(batch, seq_len, n_heads, d_head, dtype=float)

    test_RoPE = RotaryPositionalEmbeddings(d_head)
    RoPE = RotaryPositionalEmbedding(d_head)

    out = RoPE(x)
    test_out = test_RoPE(x)
    torch.testing.assert_close(test_out, out)
    print("Passed Assertion test against Pytorch RoPE")

    