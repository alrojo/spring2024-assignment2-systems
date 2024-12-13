import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from time import time

def setup(rank, world_size, backend, device):
    assert backend in ["gloo", "nccl"]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) if device.startswith("cuda") else None

def benchmark_all_reduce(rank, world_size, backend, device, byte_size):
    device = f"cuda:{rank}" if device == "cuda" else device
    setup(rank, world_size, backend, device)
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
        print(f"Backend: {backend}, Device: {device}")
        avg = sum(all_results) / world_size
        print("Time %.2f MS" % (avg*1000))

    dist.barrier()
    dist.destroy_process_group()

def print_bytes(num_bytes):
    if num_bytes < 2**20:
        print("%d KB " % (num_bytes // 2**10))
    elif num_bytes < 2**30:
        print("%d MB " % (num_bytes // 2**20))
    else:
        print("%d GB " % (num_bytes // 2**30))

if __name__ == "__main__":
    world_size = 4
    byte_sizes = [2**19, 2**20, 2**23, 2**25, 2**26, 2**29, 2**30]
    for (device, backend) in [("cpu", "gloo"), ("cuda", "gloo"), ("cuda", "nccl")]:
        for byte_size in byte_sizes:
            print_bytes(byte_size)
            mp.spawn(fn=benchmark_all_reduce, args=(world_size, backend, device, byte_size), nprocs=world_size, join=True)
