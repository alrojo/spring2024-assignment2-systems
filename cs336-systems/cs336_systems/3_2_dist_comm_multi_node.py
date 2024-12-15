import os
import argparse
from time import time
from datetime import timedelta
import torch
import torch.distributed as dist
parser = argparse.ArgumentParser(description="A simple example of argparse usage.")
parser.add_argument("--gpu", type=int, default=0, help="gpu usage, default none")
args = parser.parse_args()

def print_bytes(num_bytes):
    if num_bytes < 2**20:
        print("%d KB " % (num_bytes // 2**10))
    elif num_bytes < 2**30:
        print("%d MB " % (num_bytes // 2**20))
    else:
        print("%d GB " % (num_bytes // 2**30))

def setup(backend, device):
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
    timeout = timedelta(seconds=600)
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout)
    torch.cuda.set_device(local_rank) if device.startswith("cuda") else None
    return rank, world_size, local_rank, local_world_size

def multinode_distributed_all_reduce(backend, device, byte_size):
    rank, world_size, local_rank, local_world_size = setup(backend, device)
    device = f"cuda:{local_rank}" if device == "cuda" else device
    num_elems = byte_size // 4
    data = torch.randint(0, 10, (num_elems,)).to(device)

    # warm-up iterations
    for _ in range(5):
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize() if device.startswith("cuda") else None
    # run benchmarking
    start_time = time()
    for _ in range(10):
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize() if device.startswith("cuda") else None
    avg_time = (time()-start_time)/10

    results = avg_time
    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)
    if rank == 0:
        print_bytes(byte_size)
        avg = sum(all_results) / world_size
        print("Time %.2f MS" % (avg*1000))

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    byte_sizes = [2**19, 2**20, 2**23, 2**25, 2**26, 2**29, 2**30]
    my_list = [("cpu", "gloo")]
    if args.gpu:
        my_list = [("cuda", "gloo"), ("cuda", "nccl")]
    for (device, backend) in my_list:
        for byte_size in byte_sizes:
            multinode_distributed_all_reduce(backend, device, byte_size)
