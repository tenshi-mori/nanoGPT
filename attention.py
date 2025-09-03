import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

import einops
from jaxtyping import Float

class causal_attention(nn.Module):
    def __init__(self):
        pass
    def forward(self, x: float[Tensor, "batch posn n_embd"]) -> float[Tensor, "batch posn n_embd"]:
        pass

class causal_flash_attn(nn.Module):
    def __init__(self):
        pass
    def forward(self, x: float[Tensor, "batch posn n_embd"]) -> float[Tensor, "batch posn n_embd"]:
        pass

    