from dataclasses import dataclass


@dataclass
class SamplingParams: #对logits的采样策略
    temperature: float = 1.0 #希望多样性 temp大一点，可复现，temp小一点
    max_tokens: int = 64 #最大输出tokens
    ignore_eos: bool = False #是否忽略结束符

    def __post_init__(self): #init后验证
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
