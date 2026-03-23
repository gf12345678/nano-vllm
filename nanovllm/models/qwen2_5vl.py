import torch
from torch import nn
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, ColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope, RotaryEmbedding
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead



class Qwen2_5vlMLP(nn.Module):
    def __init__(self,config:AutoConfig):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = ColumnParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU
    
    def forward(self,input_hidden:torch.tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(input_hidden)) * self.up_proj(input_hidden))



class Qwen2_5Attention(nn.Module):
    def __init__(self,config:AutoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size//self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_proj = ColumnParallelLinear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = ColumnParallelLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = ColumnParallelLinear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = get_rope(self.head_dim,rotary_dim=self.head_dim,max_position=self.max_position_embeddings,base=self.rope_theta,rope_scaling=None)
        self.attn = Attention(self.num_heads,self.head_dim,config.sliding_window,self.num_kv_heads)
    
    def forward(self, input_hiddens:torch.Tensor, position:torch.Tensor) -> torch.Tensor:
        Q = self.q_proj(input_hiddens)
        K = self.k_proj(input_hiddens)
        V = self.v_proj(input_hiddens)
        Q,K = self.rotary_emb(Q,K,position)
        output_hidden = self.attn(Q,K,V)
        output_hidden = self.o_proj(output_hidden)
        return output_hidden


class Qwen2_5DecoderLayer(nn.Module):
    def __init__(self, config:AutoConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn= Qwen2_5Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen2_5vlMLP(config)
        
    def forward(self,input_hiddens:torch.Tensor, position:torch.Tensor) -> torch.Tensor:
        residual = input_hiddens
        input_hiddens = self.input_layernorm(input_hiddens)
        input_hiddens = self.self_attn(input_hiddens,position)
        input_hiddens = residual + input_hiddens
        residual = input_hiddens
        input_hiddens = self.post_attention_layernorm(input_hiddens)
        input_hiddens = self.mlp(input_hiddens)
        input_hiddens = residual + input_hiddens
        return input_hiddens
        
        


class Qwen2_5vlModel(nn.Module):
    def __init__(self,config:AutoConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,config.hidden_size)
        self.layers = nn.ModuleList([Qwen2_5DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
    def forward(self,input_ids:torch.Tensor, position:torch.Tensor) -> torch.Tensor:
        hidden_status = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_status = layer(hidden_status,position)
        hidden_status = self.norm(hidden_status)
        return hidden_status
        


class Qwen2_5vlForCausalLM(nn.Module):
    def __init__(self,config:AutoConfig):
        super().__init__()
        self.model = Qwen2_5vlModel(config)
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        
    def forward(self,input:torch.tensor,position:torch.Tensor) -> torch.Tensor :
        return self.model(input,position)
    
    def compute_logits(self, input:torch.Tensor) -> torch.Tensor :
        return self.lm_head(input)