from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,# [n, 16, 128] 假设为q
    cos: torch.Tensor, # [n,64]
    sin: torch.Tensor, # [n,64]
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1) # x1:[n, 16, 64],  x2:[n, 16, 64]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size#单头维度 形状为128 
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)) # θ_i = base^( - 2(i-1) / d ) 形状为[64] 
        t = torch.arange(max_position_embeddings, dtype=torch.float) # [40960]
        freqs = torch.einsum("i,j -> ij", t, inv_freq) # [40960,64] 爱因斯坦求和约定
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1) # [40960,128] 
        self.register_buffer("cos_sin_cache", cache, persistent=False) #存起来

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor, # [n]
        query: torch.Tensor, # [n,1024]
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions] # [n, 128]
        cos, sin = cos_sin.chunk(2, dim=-1) # [n,64] [n,64]
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1) #算过相同值会缓存，避免重复计算
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
