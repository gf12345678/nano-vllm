from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = [] #当前调度器的任务序列
        num_seqs = 0 # 已经被成功调度并且准备进入GPU的sequence数量
        num_batched_tokens = 0 # 当前step中需要处理的总token数量
        while self.waiting and num_seqs < self.max_num_seqs: # 本次调度器可处理sequence的条件 1.有等待的sequence 2.没超过最大限度
            seq = self.waiting[0] # 取等待sequence序列的第一个，但是不将此seq移除wait序列，下面内存条件符合后再移除
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq): #超过batched_token_nums的时候
                break 
            num_seqs += 1
            self.block_manager.allocate(seq) #给sequence分配内存
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft() # 将此seq移除wait序列
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft() #从running序列中获取任务seq
            while not self.block_manager.can_append(seq): #探测当前序列在生成下一个词时，是否会触发“新物理块的申请”且剩余的block有剩余容量
                if self.running: #仍有任务在调度器中 且剩余block不足
                    self.preempt(self.running.pop()) #退水释放最新进入任务序列的seq回waiting序列
                else:
                    self.preempt(seq) #退水释放当前seq回waiting序列
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs)) #将seq加入 running序列
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id) #将刚刚新生成的token id存入sequence
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens: # 若可以结束，则执行结束代码
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
