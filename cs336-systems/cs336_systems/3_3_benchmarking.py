import torch
from torch import nn, optim
import torch.distributed as dist
from cs336

## DL models
loss_fn = nn.MSELoss()
class ToyModel(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.fc4 = nn.Linear(10, 50, bias=False)
        self.fc5 = nn.Linear(50, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

## DDP training
def setup(backend):
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, local_world_size

def cleanup():
    dist.destroy_process_group()

def shard_data(batch, rank, world_size):
    total_samples = len(batch)
    samples_per_rank = total_samples // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank
    return batch[start_idx:end_idx]

def broadcast_model(m, src=0):
    for param in m.parameters():
        dist.broadcast(param.data, src=0)

def sync_grads(m, world_size):
    for param in m.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size

def ddp_training(batch, m, o, iters=1, backend="nccl"):
    rank, world_size, local_rank, local_world_size = setup(backend)
    device = f"cuda:{local_rank}"
    x_local = shard_data(batch["x"], rank, world_size)
    y_local = shard_data(batch["y"], rank, world_size)
    m.to(device)# is this the right way to setup the model?
    broadcast_model(m)
    for i in range(iters):
        x_local, y_local = x_local.to(device), y_local.to(device)
        o.zero_grad()
        out_local = m(x_batch)
        loss_local = loss_fn(out_local, y_local)
        loss_local.backward()
        sync_grads(m, world_size)
        o.step()
    dist.barrier()
    cleanup()
    if rank==0:
        return m, o
    else:
        return None, None

## Single training
def single_training(batch, model, o, iters=1):
    for i in range(iters):
        o.zero_grad()
        out = model(batch["x"])
        loss = loss_fn(out, batch["y"])
        loss.backward()
        o.step()
    return model, o 

## Testing
def compare_model(m1, m2):
    m1_sd, m2_sd = m1.state_dict(), m2.state_dict()
    print("testing model ...")
    for key in m1_sd.keys():
        assert torch.allclose(m1_sd[key], m2_sd[key], atol=1e-6), f"Mismatch in {key}"
    print("model is correct!")

def compare_optimizer_instance(o1, o2):
    o1_sd, o2_sd = o1.state_dict(), o2.state_dict()
    print("testing optimizer ...")
    for key in o1_sd.keys():
        assert torch.allclose(o1_sd[key], o2_sd[key], atol=1e-6), f"Mismatch in {key}"
    print("optimizer is correct!")

def testing(training_step_fn, iters=1, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(128, 10)
    y = torch.randn(128, 1)
    batch = {"x": x, "y": y)
    m = ToyModel()
    o = optim.Adam(model.instance.parameters())
    training_step_fn(batch, model_instance, o, iters)
    return model_instance, o 

## Main script
if __name__ == "__main__":
    print("starting single training ...")
    m1, o1 = testing(single_training)
    print("starting ddp training ...")
    m2, o2 = testing(ddp_training)
    if m2 is not None:
        compare_model(m1, m2)
        compare_optim(o1, o2)
