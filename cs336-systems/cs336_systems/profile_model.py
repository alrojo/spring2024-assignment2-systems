import argparse, os, time, torch
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
parser = argparse.ArgumentParser(description="A simple example of argparse usage.")
parser.add_argument("--model", type=str, required=True, help="small, medium, large, xl, 2.7B")
parser.add_argument("--device", type=int, required=True, help="device no. in integer")
parser.add_argument("--backward_pass", type=int, required=True, help="to run backwards pass")
args = parser.parse_args()

# Parse the arguments
VOCAB_SIZE=10000
BATCH_SIZE=16
CONTEXT_LENGTH=16
DROP=0.05
LR = 1e-3
WARM_UP_STEPS = 2
NUM_EVALS = 5
configs = {
	"small":{"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
	"medium":{"d_model": 1024,"d_ff": 4096,"num_layers": 24,"num_heads": 16},
	"large":{"d_model": 1280,"d_ff": 5120,"num_layers": 36,"num_heads": 20},
	"xl":{"d_model": 1600,"d_ff": 6400,"num_layers": 48,"num_heads": 25},
	"2.7B":{"d_model": 2560,"d_ff": 10240,"num_layers": 32,"num_heads": 32}
}
if args.model not in configs.keys():
    print("invalid model choice")
    assert False
config = configs[args.model]
model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        attn_pdrop=DROP,
        residual_pdrop=DROP,
)
optimizer = AdamW(model.parameters())
# generate data
X = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), dtype=torch.int64)
y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), dtype=torch.int64)
if args.device != -1:
    X, y = X.to(args.device), y.to(args.device)
    model.to(args.device)

def forward():
    logits = model(X)
    loss = cross_entropy(logits, y)
    return loss

def backward(loss):
    optimizer.zero_grad()
    loss.backward()

def optimize():
    clip_gradient(model.parameters(), 1.0)
    optimizer.step()

def sync():
    if args.device != -1:
        torch.cuda.synchronize()

def data_pass(backward_pass = True):
    sync()
    loss = forward()
    sync()
    if backward_pass:
        backward(loss)
        optimize()
    sync()

for i in range(WARM_UP_STEPS):
    print("warm-up %d" % i)
    data_pass(args.backward_pass)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    record_shapes=True,
    profile_memory=False,
    with_stack=True
) as prof:
    for i in range(NUM_EVALS):
        with record_function("forward_pass"):
            loss = forward()
            sync()
            prof.step()
        if args.backward_pass:
            with record_function("backward_pass"):
                backward(loss)
                sync()
                prof.step()
            with record_function("optimizer"):
                optimize()
                sync()
                prof.step()

prof.export_stacks("out/%s_lm_profiler_stacks.txt" % args.model, "self_cuda_time_total")
print(prof.key_averages().table(sort_by="cpu_time_total"))
print(prof.events().tree())
