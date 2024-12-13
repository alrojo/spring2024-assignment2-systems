import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class DDPWrapper_individual(nn.Module):
    def __init__(self, module: torch.nn.Module):
        ''' Given an instantiated PyTorch nn.Module to be parallelized,
            construct a DDP container that will handle gradient
            synchronization across ranks.
        '''
        super().__init__()
        self.module = module
        self.handles = []

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        def grad_sync_hook(param):
            handle = dist.all_reduce(param.grad, async_op=True)
            self.handles.append(handle)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(grad_sync_hook)

    def forward(self, *inputs, **kwargs): 
        ''' Calls the wrapped module’s forward() method with the
            provided positional and keyword arguments.
        '''
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        ''' When called, wait for asynchronous communication calls to
            complete.
        '''
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

byte_count = lambda p: int(p.numel()//p.element_size())

class DDPWrapper_bucketed(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        ''' Given an instantiated PyTorch nn.Module to be parallelized,
            construct a DDP container that will handle gradient
            synchronization across ranks.
        '''
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.bucket_byte_size = bucket_size_mb * 2**20

        params = []
        self.bucket_handles = []

        self._broadcast_parameters()
        params = list(module.parameters())
        self.buckets = self._build_buckets(params)
        self.param_to_bucket_idx = {}
        for b_idx, bucket in enumerate(self.buckets):
            for p_idx, p in enumerate(bucket["params"]):
                self.param_to_bucket[p] = (b_idx, p_idx)
                handle = p.register_post_accumulate_grad_hook(self._bucket_hook(p))
        self.bucket_handles = [None for _ in self.buckets]

    def _bucket_allreduce(b_idx):
        bucket = self.buckets[b_idx]
        grads = [p.grad for p in bucket["params"]]
        flat = _flatten_dense_tensors(grads)
        handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
        self.bucket_handles[b_idx] = (handle, grads)


    def _bucket_hook(self, param):
        def hook(*_):
            b_idx, p_idx = self.param_to_bucket_idx[param]
            bucket = self.buckets[b_idx]
            bucket["ready_count"] += 1
            if bucket["ready_count"] == len(bucket["params"]):
                self._bucket_allreduce(b_idx)
        return hook

    def _broadcast_parameters(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _build_buckets(params)
        buckets = []
        current_bucket = []
        cur_size = 0

        for p in reversed(params):
            if p.requires_grad:
                p_size = byte_count(p) 
                cur_size += p_size
                if cur_size >= self.bucket_byte_size:
                    assert current_bucket
                    buckets.append({"params": current_bucket, "ready_count": 0})
                    cur_size = p_size
                    current_bucket = [p]
                else:
                    current_bucket.append(p)
        if current_bucket:
            buckets.append({"params": current_bucket, "ready_count": 0})
        return buckets

    def finish_gradient_synchronization(self):
        ''' When called, wait for asynchronous communication calls to
            complete.
        '''
        for b_idx, entry in enumerate(self.bucket_handles):
            if entry is not None:
                handle, grads = entry
                handle.wait()
                flat = handle.tensor
                flat.div_(self.world_size)
                unflat = _unflatten_dense_tensors(flat, grads)
                for g, uf in zip(grads, unflat):
                    g.copy_(uf)
                self.bucket_handles[b_idx] = None
                self.buckets[b_idx]["ready_count"] = 0

    def forward(self, *inputs, **kwargs): 
        ''' Calls the wrapped module’s forward() method with the
            provided positional and keyword arguments.
        '''
        return self.module(*inputs, **kwargs)

