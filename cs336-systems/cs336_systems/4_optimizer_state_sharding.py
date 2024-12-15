import torch
from torch.optim import Optimizer
from typing import Any, Type, List
import torch.distributed as dist

class OSWrapper(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        if not dist.is_initialized():
            raise RuntimeError("Distributed package is not initialized.")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        all_group_params = list(params)
        self.total_bytes = 0
        for group in all_group_params:
            for p in group["params"]:
                if p.data is not None:
                    self.total_bytes += p.numel() 
        self.shard_size = torch.ceil(total_size / self.world_size)
        self.param_groups, size = self._shard_param_groups(all_params)
        self.local_optimizer = optimizer_cls(self.param_groups,  **kwargs)
        defaults = self.local_optimizer.defaults
        self.counter=0
        # I want to do something that keeps track of sizes, but not sure how
        
        super().__init__(self.param_groups, defaults)

    def _shard_param_groups(group_params): 
        sharded_param_group = []
        cum_size = 0
        start_size = self.rank*shard_size
        end_size = (self.rank+1) * shard_size
        size = 0
        for group in group_params:
            params = group["params"]
            if params is not None:
                for p in params:
                    cum_size += param.numel() 
            if cum_size > start_size and cum_size < end_size:
                sharded_param_group.append(group)
                size += sum(p.numel() for p in group["params"])
        return sharded_param_group, torch.tensor([size], dtype=torch.int)

    def step(self, closure, **kwargs):
        loss = self.local_optimizer.step(closure, **kwargs)
        """ Why do we need to synchronize? every rank has their own stuff
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    data = param.data
                    owner = self._get_param_owner(data)
                    dist.broadcast(data, src=owner)
        """
        return loss

    def _get_param_owner(data):
        """ Not sure why this is necessary
        """
        raise NotImplemented

    def add_param_group(self, param_group: dict[str, Any]):
        # Optimally, put into rank with least parameters
        # what we do: randomly pick one ...
        torch.manual_seed(self.counter)
        counter += 1
        rank = torch.randint(0, self.world_size-1)
        if rank == self.rank:
            self.param_groups.append(param_group)
