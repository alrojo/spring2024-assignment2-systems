import os
from datetime import timedelta
import torch
import torch.distributed as dist

def setup(device="cuda"):
    # These variables are set via srun
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    # MASTER_ADDR and MASTER_PORT should have been set in our sbatch script,
    # so we make sure that's the case.
    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]
    # Default timeout is 30 minutes. Reducing the timeout here, so the job fails quicker if there's
    # a communication problem between nodes.
    timeout = timedelta(seconds=60)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)
    torch.cuda.set_device(local_rank) if device.startswith("cuda") else None
    return rank, world_size, local_rank, local_world_size

def multinode_distributed_demo(device="cuda"):
    rank, world_size, local_rank, local_world_size = setup()
    device = f"cuda:{local_rank}" if device == "cuda" else device
    print(
            f"World size: {world_size}, global rank: {rank}, "
            f"local rank: {local_rank}, local world size: {local_world_size}"
            f"device: {device}"
            )
    data = torch.ones(5).to(device)
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    print(f"rank {rank} data (after all-reduce): {data}")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    multinode_distributed_demo()
