import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1) #分割，按照-1维，也就是(2 * intermediate_size)
        return F.silu(x) * y #x部分是gate y部分是up
                                                             