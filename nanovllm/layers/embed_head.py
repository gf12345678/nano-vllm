import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module): #用于编码计算

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size #根据tp进行分割
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor): #embedding 的权重加载
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size #加载对应tp的权重
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1: #多个GPU的情况
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx) #对其x序列中的非当前GPU的token id 进行掩码
            x = mask * (x - self.vocab_start_idx) # 减去起始id 从0开始
        y = F.embedding(x, self.weight) #一个或者多个GPU情况，都要编码
        if self.tp_size > 1: #多个GPU的情况
            y = mask.unsqueeze(1) * y #消除x = mask * (x - self.vocab_start_idx)这一步 mask将非本gpu的token id 归为0的情况(id 为0 也会编码，所以要消除编码影响)
            dist.all_reduce(y) #同步通信，将所有GPU上的y进行合并
        return y


class ParallelLMHead(VocabParallelEmbedding): #用于lm head 和 logits的计算

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor): # 假设x shape (10+20+30)*1024，三段prefill的合并
        context = get_context()
        if context.is_prefill: # cu_seqlens_q的形状就是[0, 10, 30, 60]
            last_indices = context.cu_seqlens_q[1:] - 1 # last_indices 的形状就是[9, 29, 59]
            x = x[last_indices].contiguous() # 这里将prefill的最后一个token的输出取出来合并，x的新形状 3*1024
        logits = F.linear(x, self.weight) # 计算logits
        if self.tp_size > 1: #多GPU情况
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None #rank=0的卡建空的全部的logits
            dist.gather(logits, all_logits, 0) #合并全部logits到0卡的all logits上，gather操作后形状 GPUnum * batchnum * 15W(词表长度)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None #合并GPUnum层
        return logits
