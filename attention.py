import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

import einops
from jaxtyping import Float

NEG_INF = torch.tensor(float("-inf"))
EPSILON = 1e-10 # Revert to original epsilon value

class causal_attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias = config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: Float[Tensor, "b t n_embd"]) -> Float[Tensor, "b t n_embd"]:
        b, t, n_embd = x.shape
        q, k, v = einops.rearrange(self.c_attn(x), "b t (a nh hs) -> a b nh t hs", a = 3, nh = self.n_head).unbind(dim = 0)
        head_size = q.shape[-1]
        attn = einops.einsum(q, k, "b nh t hs, b nh s hs -> b nh t s") * (1.0 / math.sqrt(head_size))
        attn = attn.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn_output = attn @ v
        y = einops.rearrange(attn_output, "b nh t hs -> b t (nh hs)")
        y = self.c_proj(y)
        return y

class flash_attn(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias = config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.b_c = config.b_c
        self.b_r = config.b_r

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: Float[Tensor, "b t n_embd"]) -> Float[Tensor, "b t n_embd"]:
        q, k, v = einops.rearrange(self.c_attn(x), "b t (a nh hs) -> a b nh t hs", a = 3, nh = self.n_head).unbind(dim = 0)
        o = torch.zeros_like(q, requires_grad=True)
        l = torch.zeros(q.shape[:-1])[..., None]
        m = torch.ones(q.shape[:-1])[..., None] * NEG_INF
        
        head_size = q.shape[-1]
        q_block_size = min(head_size, self.b_r)
        kv_block_size = self.b_c 

        q_blocks = torch.split(q, q_block_size, dim = 2)
        k_blocks = torch.split(k, kv_block_size, dim = 2)
        v_blocks = torch.split(v, kv_block_size, dim = 2)

        o_blocks = list(torch.split(o, q_block_size, dim = 2))
        l_blocks = list(torch.split(l, q_block_size, dim = 2))
        m_blocks = list(torch.split(m, q_block_size, dim = 2))
        
        Tr = len(q_blocks)
        Tc = len(k_blocks)

        # Dynamic causal masking setup
        q_len = q.shape[2]
        k_len = k.shape[2]

        q_range = torch.arange(q_len)[:,None]
        k_range = torch.arange(k_len)[None,:]

        q_range_blocks = torch.split(q_range, q_block_size, dim=0)
        k_range_blocks = torch.split(k_range, kv_block_size, dim=1)

        for j in range(Tc):
            kj = k_blocks[j]
            vj = v_blocks[j]
            k_range_blocksj = k_range_blocks[j]

            for i in range(Tr):
                qi = q_blocks[i]
                oi = o_blocks[i]
                li = l_blocks[i]
                mi = m_blocks[i]
                q_range_blocksi = q_range_blocks[i]

                qi_scaled = qi * (1.0 / math.sqrt(head_size))

                s_ij = einops.einsum(qi_scaled, kj, "b nh r hs, b nh c hs -> b nh r c")
                
                # Dynamic causal mask application
                causal_mask = q_range_blocksi >= k_range_blocksj
                s_ij = torch.where(causal_mask, s_ij, NEG_INF)

                m_block_ij, _ = torch.max(s_ij, dim = -1, keepdim=True)
                p_ij = torch.exp(s_ij - m_block_ij)

                p_ij = torch.where(causal_mask, p_ij, 0.0)
                l_block_ij = torch.sum(p_ij, dim = -1, keepdim=True)

                p_ij_vj = einops.einsum(p_ij, vj, "b nh r c, b nh c hs -> b nh r hs")

                mi_new = torch.maximum(m_block_ij, mi)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

                o_blocks[i] = (li / li_new) * torch.exp(mi - mi_new) * oi + (torch.exp(m_block_ij - mi_new) / li_new) * p_ij_vj
                l_blocks[i] = li_new
                m_blocks[i] = mi_new
        
        o = torch.cat(o_blocks, dim = 2)
        l = torch.cat(l_blocks, dim = 2)
        m = torch.cat(m_blocks, dim = 2) 
        y = einops.rearrange(o, "b nh t hs -> b t (nh hs)")
        y = self.c_proj(y)
        return y

class grouped_query_attn(nn.Module):
    def __init__(self, config):
        super().__init__()
        # This line is a little confusing becausing do we count key and value heads separately or together.
        # Not really sure but OpenAI people said key and value should be grouped together so I'll follow their lead
        qkv_dim = config.head_dim * (config.num_attn_heads + 2 * config.num_kv_heads) 
        self.c_attn = nn.Linear(config.n_embd, qkv_dim, bias = config.bias)
        self.c_proj = nn.Linear(config.num_attn_heads * config.head_dim, config.n_embd, bias = config.bias)
    def forward(self, x: Float[Tensor, "b t n_embd"]) -> Float[Tensor, "b t n_embd"]:
        qkv = self.c_attn(x)
        qkv_parts = (
            config.num_attn_heads * config.head_dim,
            config.num_kv_heads * config.head_dim,
            config.num_kv_heads * config.head_dim,
        )
        q, k, v = torch.split(qkv, qkv_parts, dim = -1)

        q = einops.rearrange(q, "b t (nh dh) -> b nh t dh", dh = config.head_dim)
        k = einops.rearrange(k, "b s (nh dh) -> b nh s dh", dh = config.head_dim)
        v = einops.rearrange(v, "b s (nh dh) -> b nh s dh", dh = config.head_dim)

        q = einops.rearrange(q, "b (na nkv) t dh -> b na nkv t dh", nkv = config.num_kv_heads)
        attn_score = einops.einsum(q, k, "b na nkv t dh, b nh s dh -> b nh t s") * (1.0 / math.sqrt(config.head_dim))
        attn = F.softmax(attn_score, dim=-1)
        y = attn @ v
        y = einops.rearrange(y, "b nh t dh -> b t (nh dh)")
        return y
if __name__ =="__main__":
    class Config:
        n_embd = 64
        bias = True
        head_dim = 4
        num_attn_heads = 8
        num_kv_heads = 2
    config = Config()
    x = torch.randn(1, 256, 64)
    model = grouped_query_attn(config)
    model(x)
