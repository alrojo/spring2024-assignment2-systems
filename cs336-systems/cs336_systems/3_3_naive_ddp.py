import torch
from torch import nn, optim
import torch.distributed as dist
from cs336_basics.nn_utils import cross_entropy, clip_gradient

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
    dist.barrier()
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

def single_training(batch, m, o, iters=1):
    for i in range(iters):
        o.zero_grad()
        out = model(batch["x"])
        loss = loss_fn(out, batch["y"])
        loss.backward()
        o.step()
    return m, o 

## Testing
def compare_model(m1, m2, rank):
    m1_sd, m2_sd = m1.state_dict(), m2.state_dict()
    for key in m1_sd.keys():
        if rank==0:
            assert torch.allclose(m1_sd[key], m2_sd[key], atol=1e-6), f"Mismatch in {key}"
        else:
            assert not torch.allclose(m1_sd[key], m2_sd[key], atol=1e-6), f"Mismatch in {key}"

def compare_optimizer_instance(o1, o2, equal=True):
    o1_sd, o2_sd = o1.state_dict(), o2.state_dict()
    for key in o1_sd.keys():
        if equal: 
            assert torch.allclose(o1_sd[key], o2_sd[key], atol=1e-6), f"Mismatch in {key}"
        else: 
            assert not torch.allclose(o1_sd[key], o2_sd[key], atol=1e-6), f"Mismatch in {key}"

def testing(training_step_fn, iters=1, seed=0):
    m = ToyModel()
    o = optim.Adam(model.instance.parameters())
    training_step_fn(batch, model_instance, o, iters)
    return m_instance, o 

# training script
def ddp_training(batch, iters=5, backend="nccl"):
    rank, world_size, local_rank, local_world_size = setup(backend)
    torch.manual_seed(rank)
    dist.barrier()
    device = f"cuda:{local_rank}"
    x_local = shard_data(batch["x"], rank, world_size)
    y_local = shard_data(batch["y"], rank, world_size)
    single_model = ToyModel().to(device)
    ddp_model = copy.deepcopy(single_model)
    broadcast_model(ddp_model, src=0)
    # if rank 0 should be same, else different
    compare_model(single_model, ddp_model, rank)

    single_optimizer = Adam.Optimizer(single_model.parameters, lr=0.1)
    ddp_optimizer = Adam.Optimizer(ddp_model.parameters, lr=0.1)

    for i in range(iters):
        # single_model_step
        single_optimizer.zero_grad()
        single_out = single_model(batch["x"])
        single_loss = loss_fn(single_out, batch["y"])
        single_loss.backward()
        single_optimizer.step()

        # ddp step
        ddp_optimizer.zero_grad()
        ddp_out = single_model(batch["x"])
        ddp_loss = loss_fn(single_out, batch["y"])
        ddp_loss.backward()
        sync_grads(ddp_model, world_size)
        ddp_optimizer.step()
        compare_model(single_model, ddp_model, rank)
    print(f"Rank: {rank} checks out fine")
    cleanup()

## Main script
if __name__ == "__main__":
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    batch = {"x": x, "y": y)
    ddp_training(batch)
