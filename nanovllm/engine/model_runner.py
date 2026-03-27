import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size # KV Cache 块大小 16或者32 可以容纳 16或32个token的kv cache
        self.enforce_eager = config.enforce_eager # 强制即时执行模式
        self.world_size = config.tensor_parallel_size # 张量并行度
        self.rank = rank # 当前进程排名
        self.event = event # 事件同步对象，主进程和子进程都可以改变其状态，用来通信

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)  # GPU的通信
        torch.cuda.set_device(rank) # 进行序列与GPU序列进行绑定
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        model_type = getattr(hf_config, "model_type", "qwen3")
        if model_type == "qwen3":
            self.model = Qwen3ForCausalLM(hf_config) #每个进程都起model(model会被切割，按照tp数)
        else:
            raise ValueError(f"Unsupported model type : { model_type }")
                
        
        # self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph() #使用捕捉录制 cuda graph！
        torch.set_default_device("cpu") #模型加载结束，避免部分中间变量 临时变量在GPU中污染显存，所以需要切换回CPU
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20) #主进程的share memory 用于分发任务和tp合并，2的10次是1K，2的20次是1M，2的30次是1GB
                dist.barrier() # 同步
            else:
                dist.barrier() # 阻塞同步等待主进程的barrier
                self.shm = SharedMemory(name="nanovllm")  #子进程 链接 主进程的share memory
                self.loop() #子进程死循环

    def exit(self): #退出进程需要释放资源
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self): #loop 子进程死循环
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args) #子进程调用的具体方法
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait() # 自我阻塞
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4]) #反序列号
        self.event.clear() # 清除标志位
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0 #断言确定是主进程
        data = pickle.dumps([method_name, *args]) #序列号二进制
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little") #前四个字节记录本次传输的长度n
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set() #主进程 通知 子进程解除阻塞(read_shm)

    def call(self, method_name, *args): 
        if self.world_size > 1 and self.rank == 0: #主进程写入共享内存
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None) #子进程读取共享内存
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache() #清空内存
        torch.cuda.reset_peak_memory_stats() #
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)] #构建最大seqs去模拟峰值
        self.run(seqs, True) #prefill的
        torch.cuda.empty_cache() #清空内存

    def allocate_kv_cache(self): #分配KV cache
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size  # 张量并行下的KV头切分逻辑  模型原始配置中定义的 Key-Value Heads 总数//GPU的总数量 
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads) # 计算每个头的dim   hidden_size = num_attention_heads * head_dim
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize  # 这里block_bytes是计算每个块的占用内存大小，2是K和V， num_kv_heads * head_dim是一个token的KV cache大小
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes # 计算可用的block数量
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim) # 申请一大块的空余显存，用于将来存放block，形状为 2*layer_num*block_num*block_size*head_num*head_dim
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]# 将kv_cache的显存进行地址分配 0为K 1为V
                module.v_cache = self.kv_cache[1, layer_id]#
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs] #填充拼接 -1
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]): # 目标是将一个 Batch（批次）中多个不同长度的 Sequence 转换成 GPU 算子可以高效处理的连续平铺张量
        input_ids = [] # 本次需要进入模型计算的 Token ID 列表
        positions = [] # 每个 Token 在其所在序列中的绝对位置索引
        cu_seqlens_q = [0] # 针对 Query 的偏移量 (由于多个请求被拼成了一个长向量，CUDA 算子需要知道哪里是一个请求的结束，哪里是下一个的开始)
        cu_seqlens_k = [0] # 针对 KV 的偏移量 (由于多个请求被拼成了一个长向量，CUDA 算子需要知道哪里是一个请求的结束，哪里是下一个的开始)
        max_seqlen_q = 0 # 当前 Batch 中最长的q
        max_seqlen_k = 0 # 当前 Batch 中最长的kv
        slot_mapping = [] # 这是 PagedAttention 机制中的“地址索引表” 内容一串整数，对应物理显存中每个 Token 的具体槽位索引，相当于token级别的table
        block_tables = None
        for seq in seqs:
            seqlen = len(seq) #sequence中token id的数量
            input_ids.extend(seq[seq.num_cached_tokens:]) #将前面已经被缓存过的切掉(别的sequence存过)，剩余没缓存的token id 并入input_ids
            positions.extend(list(range(seq.num_cached_tokens, seqlen))) #缓存剩余没缓存的token id的绝对位置编码,然后range 序列
            seqlen_q = seqlen - seq.num_cached_tokens # q为需要推理的长度(没缓存过的长度)
            seqlen_k = seqlen # sequence 的全部长度
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q) # 针对 Query 的偏移量 
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k) # 针对 KV 的偏移量 
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    #block为空
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks): #sequence中未缓存的block 
                start = seq.block_table[i] * self.block_size #获得sequence的未还缓存的block id  e.g. [3, 32, 45, 7, 1]
                if i != seq.num_blocks - 1: #是否为block序列中的最后一个块
                    end = start + self.block_size #不是最后一个块，位移直接是 start + size
                else:
                    end = start + seq.last_block_num_tokens #没满 start + 最后的block的token 数量
                slot_mapping.extend(list(range(start, end))) #slot获得对应的显存地址空间，attention计算的时候 用
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # 判断是否存在前缀缓存prefix cache，如果有前缀缓存，则需要建立表格进行查表
            block_tables = self.prepare_block_tables(seqs) # 获得二维的block table
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1) #只取最后一个token的物理映射
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables) #每次decode 都会set context，结束后会reset
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512: #prefill 启动eager模式
            return self.model.compute_logits(self.model(input_ids, positions))
        else: #decode 使用cuda graph模式 
            bs = input_ids.size(0) #
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)] #bs向上取整，获得回放graph对象
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()  # 调用cuda graph 回放操作一次，结果输出到字典的output中
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]: #主进程和子进程同时执行的部分
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None #主进程获得采样器
        logits = self.run_model(input_ids, positions, is_prefill) #调用decode的cuda graph或者prefill的forward
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None #采样只在gpu0上
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self): #cuda graph 设置cuda graph
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512) #按照最大进行预跑
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size #按照最大进行预跑
        input_ids = torch.zeros(max_bs, dtype=torch.int64) #按照最大进行预跑
        positions = torch.zeros(max_bs, dtype=torch.int64)#按照最大进行预跑
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)#按照最大进行预跑
        context_lens = torch.zeros(max_bs, dtype=torch.int32)#按照最大进行预跑
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)#按照最大进行预跑
        outputs = torch.zeros(max_bs, hf_config.hidden_size)#按照最大进行预跑
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs): # 先录制大的，后录制小的，有助于大的找到连续显存空间，解决内存碎片化：CUDA Graph 需要“私有”的内存池（Memory Pool）。
            graph = torch.cuda.CUDAGraph() #创建录像带
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # Warmup先跑一次
            with torch.cuda.graph(graph, self.graph_pool):   # 开始录制 图捕获
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None: #self.graph_pool 共享显存池，第一次的显存最大，把这块显存保留，后面都在这块显存上跑
                self.graph_pool = graph.pool() #显存保留，后面的小bs都用这个显存
            self.graphs[bs] = graph #将录像带存入记录数组中
            torch.cuda.synchronize() #
            reset_context() #清理上下文

        self.graph_vars = dict(    # self加一层引用，保证内存地址不会被释放(input_ids,positions这些) 创建字典，这个字典的地址就是后面所有回放跑的地址
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
