import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim #记录 当前的维度切的维度 0是输出维度切分，将W行分割，1是输入维度切分，将W列分割
        self.tp_rank = dist.get_rank() #当前rank
        self.tp_size = dist.get_world_size() #全部tp的数量
        self.weight = nn.Parameter(torch.empty(output_size, input_size)) #weight 形状是[Oouputsize,Inputsize]，在liner计算时会转置
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase): #不切的全矩阵

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim) # 获得size(0)，获得output_dim/tp_size（这里是除不是或），也就是当前GPU上的真实output_dim
        start_idx = self.tp_rank * shard_size #计算偏移量
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size) # narrow 就是在不移动数据的前提下，给大矩阵画了一个特定维度的“取景框”。
        param_data.copy_(loaded_weight) #拷贝

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) #这里得到的结果并未像rowparallellinear 进行合并，因为这里是列并行，每一列结果是不影响的


class MergedColumnParallelLinear(ColumnParallelLinear): #mlp中gate和up合并，列并行 output分割

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int): #loaded_shard_id ，0是gate， 1是up
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size #起始偏移量
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size #
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size) #选择拷入区域
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank] #按照tpsize切块，按照0维切分，然后取rank位置的数据
        param_data.copy_(loaded_weight) #考入数据


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size) # num_heads是Q的头数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str): #qkv 被合并在一个矩阵中
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size #Q矩阵长度
            shard_offset = 0 #偏移量 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size #偏移量是 Q的维度
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size #偏移量是Q+K的维度
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase): #按照 input 分割 行分割

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim) #获得input_dim/tp_size（这里是除不是或），也就是当前GPU上的真实input_dim
        start_idx = self.tp_rank * shard_size  #偏移量
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size) #划区域
        param_data.copy_(loaded_weight) #拷贝

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y) #规约合并累加，行并行需要累加，才能获得真正输出
        return y
