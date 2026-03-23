import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384 #最大 total token nums
    max_num_seqs: int = 512 # 最大seq 请求数
    max_model_len: int = 4096 # 模型可处理最大length
    gpu_memory_utilization: float = 0.9 #最大gpu利用率
    tensor_parallel_size: int = 1  # 张量并行的GPU数量
    enforce_eager: bool = False  #cuda graph是否开关的 False 是cuda graph开
    hf_config: AutoConfig | None = None
    eos: int = -1 # 结束符
    kvcache_block_size: int = 256  #kv cache 块大小，每个块可以存多少token 的kvcache
    num_kvcache_blocks: int = -1 #block 总数 -1将会根据GPU动态计算

    def __post_init__(self): #init 后的执行
        assert os.path.isdir(self.model) #检查路径
        assert self.kvcache_block_size % 256 == 0 #确定整除
        assert 1 <= self.tensor_parallel_size <= 8 #张量并行参数确认
        self.hf_config = AutoConfig.from_pretrained(self.model) #
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len #单批次token长度需要大于模型最大token length

"""

import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="read config")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="model path"
    )
    args = parser.parse_args()
    config = Config(model = args.model)
    print(config)
    
"""