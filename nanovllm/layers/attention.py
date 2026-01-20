import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr, # 获得KV hidden_dim
):
    idx = tl.program_id(0) #当前实例的id  同时启动了N个实例对应N个Token。idx=0 处理第0个Token，idx=1 处理第 1 个
    slot = tl.load(slot_mapping_ptr + idx) #获取显存地址(相对的)，读出当前Token将要被分配到的显存地址
    if slot == -1: return #代表这个 Token 不需要缓存，比如是 Padding
    key_offsets = idx * key_stride + tl.arange(0, D) # 计算缓冲区中 hidden每个数字对应的显存地址(相对从0开始)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets) # load 操作 从当前的计算缓冲区把K, V 数据“吸”进 GPU 的寄存器（高速缓存）
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D) #计算hidden_dim相对的显存地址
    tl.store(k_cache_ptr + cache_offsets, key) #存储到真实的显存地址中
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape # 获得KV形状
    D = num_heads * head_dim # 获得KV hidden_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads # Query Heads 数
        self.head_dim = head_dim # 每个 Head 的维度
        self.scale = scale # 缩放因子  用于防止 Q KT 点积结果过大导致 Softmax 梯度消失。通常取值是 根号下head_dim分之一
        self.num_kv_heads = num_kv_heads # KV的head num
        self.k_cache = self.v_cache = torch.tensor([])  # runner初始化分配的大显存

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context() #获得全局信息 
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping) #在这里对KV cache进行block缓存
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
