import torch
from torch import nn


class Sampler(nn.Module): #采样层

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) #logits除以温度 unsqueeze是为了对齐维度
        probs = torch.softmax(logits, dim=-1) #Vocabulary_Size维度进行归一化
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1) # torch.empty_like(probs).exponential_(1)生成符合指数分布的随机噪声，clamp_min_(1e-10)是防止除零
        return sample_tokens #clamp_min_(1e-10)就是截断1e-10以下的
