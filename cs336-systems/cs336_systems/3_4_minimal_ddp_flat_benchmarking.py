from time import time
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, clip_gradient

# Parse the arguments
VOCAB_SIZE=10000
BATCH_SIZE=16
CONTEXT_LENGTH=128
DROP=0.0
LR = 1e-3
WARM_UP_STEPS = 1
NUM_EVALS = 5
configs = {
	"small":{"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
	"medium":{"d_model": 1024,"d_ff": 4096,"num_layers": 24,"num_heads": 16},
	"large":{"d_model": 1280,"d_ff": 5120,"num_layers": 36,"num_heads": 20},
	"xl":{"d_model": 1600,"d_ff": 6400,"num_layers": 48,"num_heads": 25},
	"2.7B":{"d_model": 2560,"d_ff": 10240,"num_layers": 32,"num_heads": 32}
}
config = configs["medium"]
loss_fn = cross_entropy

# basic functionality for ddp
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

def sync_grads(m, world_size):
    for param in m.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size

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

# the fancy function of interest!
def sync_grads_flat(m, world_size):
    grads, shapes = [], []
    for param in m.parameters():
        grads.append(param.grad.data.view(-1))
        shapes.append(param.grad.shape)
    if grads:
        flat_grads = torch.cat(grads)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= world_size
        offset = 0
        for param, shape in zip(model.parameters(), shape):
            numel = param.grad.data.numel()
            param.grad.data.copy_(flat_grads[offset:offset+numel].view(shape))
            offset += numel
 
# training script
def ddp_training(batch, flat_grads, iters=100, backend="nccl"):
    rank, world_size, local_rank, local_world_size = setup(backend)
    torch.manual_seed(rank)
    dist.barrier()
    device = f"cuda:{local_rank}"
    x_local = shard_data(batch["x"], rank, world_size)
    y_local = shard_data(batch["y"], rank, world_size)
    ddp_model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        attn_pdrop=0.0,
        residual_pdrop=0.0,
        norm_layer_type="triton"
    ).to(device)
    flatten_tensor = flatten_state_dict(_model)
    broadcast_model(ddp_model, src=0)
    ddp_optimizer = Adam.Optimizer(ddp_model.parameters, lr=0.1)

    start_time = time()
    for i in range(iters):
        ddp_optimizer.zero_grad()
        ddp_out = single_model(batch["x"])
        ddp_loss = loss_fn(single_out, batch["y"])
        ddp_loss.backward()
        if flat_grads == True:
            sync_grads_flat(ddp_model, world_size)
        else:
            sync_grads(ddp_model, world_size)
        ddp_optimizer.step()
    dist.barrier()
    if rank == 0:
        avg_time = (time()-start_time)/iters
        print("average_time %.2f" % avg_time)
    cleanup()

if __name__ == "__main__":
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), dtype=torch.int64)
    y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), dtype=torch.int64)
    batch = {"x": x, "y": y)
    ddp_training(batch, flat_grads=False)
    ddp_training(batch, flat_grads=True)
