from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False #是否为预填充
    cu_seqlens_q: torch.Tensor | None = None  #记录在合并长序列后的每个seq的偏移量
    cu_seqlens_k: torch.Tensor | None = None  #记录在合并长序列后的每个seq的偏移量
    max_seqlen_q: int = 0  # 当前 Batch 中 Query 和 Key 的最大长度。
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None # 槽位映射表。记录了当前计算的每个 Token 对应的 KV 向量在物理显存池中的绝对偏移地址。
    context_lens: torch.Tensor | None = None # 上下文有效长度。记录每个序列到目前为止总共拥有多少个 Token。
    block_tables: torch.Tensor | None = None # 块寻址表。一个二维矩阵，记录了每个序列按顺序占用的所有物理块 ID。

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
