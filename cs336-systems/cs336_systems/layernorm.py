import torch, time
import torch.nn as nn

from cs336_basics.model import RMSNorm
from cs336_systems.triton import RMSNormTriton

rows = 5000 #50000
last_dims = [64, 128, 256, 512]#[1024, 2048, 4096, 8192]
passes = 1000
device="cpu"
to_ms=1000

def sync():
    if device == "gpu":
        torch.cuda.synchronize()

def time_norm(model, x, w):
    time_fwd, time_bck = [],[]
    for _ in range(passes):
        sync()
        t1 = time.time()
        out = model(x)
        sync()
        t2 = time.time()
        out.backward(w)
        sync()
        time_bck.append(time.time()-t2)
        time_fwd.append(t2-t1)
    return sum(time_fwd)*to_ms/passes, sum(time_bck)*to_ms/passes

def warm_ups(model, x, warm_ups=5):
    for _ in range(warm_ups):
        model(x)
print("device: ", device)
print("rows: ", rows)
for last_dim in last_dims:
    x = torch.randn(rows, last_dim, device=device, requires_grad=True)
    w = torch.randn(rows, last_dim, device=device)
    my_norm = RMSNorm(last_dim).to(device)
    torch_norm = nn.LayerNorm(last_dim).to(device)
    triton_norm = RMSNormTriton(last_dim).to(device)
    # warm-ups
    warm_ups(my_norm, x)
    warm_ups(torch_norm, x)
    warm_ups(triton_norm, x)
    # time norms
    my_fwd_time, my_bck_time = time_norm(my_norm, x, w)
    torch_fwd_time, torch_bck_time = time_norm(my_norm, x, w)
    triton_fwd_time, triton_bck_time = time_norm(triton_norm, x, w)
    print("Dim: ", last_dim)
    print("my forward time = %.2f ms, my backward time = %.2f ms: " % (my_fwd_time, my_bck_time))
    print("torch forward time = %.2f ms, torch backward time = %.2f ms: " % (torch_fwd_time, torch_bck_time))
    print("triton forward time = %.2f ms, triton backward time = %.2f ms: " % (triton_fwd_time, triton_bck_time))
