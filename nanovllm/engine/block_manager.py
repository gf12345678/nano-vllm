from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id #当前block的 id 号
        self.ref_count = 0 # 当前block 被多少sequence引用
        self.hash = -1 #当前block的唯一hash值，根据前缀token和当前block的token决定 hash是block满了才计算
        self.token_ids = [] #当前block的kv cache 存储的token id序列

    def update(self, hash: int, token_ids: list[int]): #更新
        self.hash = hash
        self.token_ids = token_ids

    def reset(self): #重置
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:  #全局唯一的大block类

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size # 每个block可以存多少个token的kv cache
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] #申请多少个block，之前根据剩余显存计算过，id是从0到num-1
        self.hash_to_block_id: dict[int, int] = dict() #哈希值对应的block id
        self.free_block_ids: deque[int] = deque(range(num_blocks)) #空余block id，这里是双向队列 
        self.used_block_ids: set[int] = set() #已用block id ，这里是set集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1): #计算哈希，hash id 只有block满了才有，因为满了才能确认前缀有没有重复
        h = xxhash.xxh64() #prefix是上一个block的 hash id
        if prefix != -1: # 第一个block 没有前block  prefix 为 -1
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block: # 内存分配
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block: # 内存释放
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool: # 对当前seq的block需要数量，剩余block是否能满足 prefill阶段
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):  # 给刚进入系统、还没分配内存的请求分配空间 prefill阶段
        assert not seq.block_table  # 判断seq的 block table 为空
        h = -1  #初始化为-1 每轮for循环迭代变化
        cache_miss = False
        for i in range(seq.num_blocks): # 当前token_num//block_size向上取整后的返回值，代表需要多少block
            token_ids = seq.block(i) #对应block的token id的切片
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1 # 只有“满块”才存入hash， 没满存的hash值为-1 ，只有“满块”的内容是固定不变的“历史”，适合作为公共前缀。不满的只分配显存，但是不存入hash table中
            block_id = self.hash_to_block_id.get(h, -1) # 没满的一定查不到，返回-1。满了的可能之前别的sequence已经存储过相同前缀token，所以可以查到对应的block id，没查到是没存过
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids: # 如果block没满，没查到，或者block内token id存储不同(哈希冲突，伪命中)，说明需要新的缓存去存储
                cache_miss = True
            if cache_miss: #需要新的缓存去存储
                block_id = self.free_block_ids[0] # 拿最free 的 block id
                block = self._allocate_block(block_id)
            else:  # 相同前缀和相同token id已经存过了
                seq.num_cached_tokens += self.block_size #sequence中被缓存的token id数量增加
                if block_id in self.used_block_ids: # 这块token id 被某个其他的sequence用
                    block = self.blocks[block_id]
                    block.ref_count += 1 #这块的引用加1 
                else:
                    block = self._allocate_block(block_id) #这块 token id 存在 但是都没被用，属于已经释放，但是id未解除状态，也需要重新分配
            if h != -1: # 当前token id 可以填满整个块的情况下（不论是否存过）
                block.update(h, token_ids) # 更新块内的哈希id和token ids
                self.hash_to_block_id[h] = block_id # 更新table
            seq.block_table.append(block_id) #sequence 的block table更新（不填满的情况下也更新）

    def deallocate(self, seq: Sequence): #释放seq的全部block(不是真释放，是引用-1，引用为零才真释放)
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id) #引用为零真释放
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool: #decode阶段
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1) #只有在多一个token的情况下，并且有free的情况下，才return true

    def may_append(self, seq: Sequence): #decode阶段
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1: #需要新增block，上一个block已经填满
            assert last_block.hash != -1 #-1是没存满，所以必须不是-1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:#当前block刚好填满，需要计算hash并且放入hash table
            assert last_block.hash == -1 #-1是没存满，所以必须是-1，刚好存满，然后改状态
            token_ids = seq.block(seq.num_blocks-1) #返回最后一个块的token id
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1 #获得前一个block的hash值
            h = self.compute_hash(token_ids, prefix) #计算唯一hash
            last_block.update(h, token_ids) #更新block内信息
            self.hash_to_block_id[h] = last_block.block_id #将hash id 存入 hash table中
        else:
            assert last_block.hash == -1 #没存满
