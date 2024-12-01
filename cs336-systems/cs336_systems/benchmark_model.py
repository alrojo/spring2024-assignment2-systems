import argparse, os, time, torch
import numpy as np
from contextlib import nullcontext
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
parser = argparse.ArgumentParser(description="A simple example of argparse usage.")
parser.add_argument("--model", type=str, required=True, help="small, medium, large, xl, 2.7B")
parser.add_argument("--device", type=int, required=True, help="device no. in integer")
parser.add_argument("--backward_pass", type=int, required=True, help="to run backwards pass")
parser.add_argument("--mixed_precision", type=int, default=0, help="mixed vs full precision")
parser.add_argument("--use_rms_norm", type=int, default=1, help="RMSNorm vs LayerNorm")
args = parser.parse_args()

# Parse the arguments
VOCAB_SIZE=10000
BATCH_SIZE=4
CONTEXT_LENGTH=128
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
        use_rms_norm=bool(args.use_rms_norm)
)
optimizer = AdamW(model.parameters())
# generate data
X = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), dtype=torch.int64)
y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), dtype=torch.int64)

train_context = None
if args.mixed_precision:
    if args.device != -1:
        train_context = torch.amp.autocast(device_type=args.device, dtype=torch.bfloat16)
    else:
        train_context = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
else:
    train_context = nullcontext()

if args.device != -1:
    X, y = X.to(args.device), y.to(args.device)
    model.to(args.device)

def forward():
    if args.device != -1:
        torch.cuda.synchronize()
    logits = model(X)
    loss = cross_entropy(logits, y)
    if args.device != -1:
        torch.cuda.synchronize()
    return loss

def backward(loss):
    if args.device != -1:
        torch.cuda.synchronize()
    optimizer.zero_grad()
    loss.backward()
    clip_gradient(model.parameters(), 1.0)
    optimizer.step()
    if args.device != -1:
        torch.cuda.synchronize()

def data_pass(backward_pass = True):
    loss = forward()
    if backward_pass:
        backward(loss)

for i in range(WARM_UP_STEPS):
    print("warm-up %d" % i)
    data_pass(args.backward_pass)

# setup timer on data_pass
times = []
for i in range(NUM_EVALS):
    print("eval %d" % i)
    start_time = time.time()
    with train_context:
        data_pass(args.backward_pass)
    end_time = time.time()
    times.append(end_time-start_time)
avg_time = sum(times) / NUM_EVALS
std_time = np.std(times)
bw = "with" if args.backward_pass else "without"
statement = ("Running %s model %s Backward pass for an average of %.2f"
             "seconds per iteration with a standard deviation of %.2f" % (
                args.model, bw, avg_time, std_time)
            )
print(statement)
