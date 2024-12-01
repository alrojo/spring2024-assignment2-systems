import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("\tfc1", self.fc1.weight.dtype)
        print("\tln", self.ln.weight.dtype)
        print("\tfc2", self.fc2.weight.dtype)
        x = self.fc1(x)
        print("\tfc1(x)", x.dtype)
        x = self.relu(x)
        print("\trelu(fc1(x))", x.dtype)
        x = self.ln(x)
        print("\tln(relu(fc1(x)))", x.dtype)
        x = self.fc2(x)
        print("\tfc2(ln(relu(fc1(x))))", x.dtype)
        return x


device = "cpu"
dtype = torch.bfloat16
batch_size, num_features, num_out = 16, 100, 2
train_context = torch.amp.autocast(device_type=device, dtype=dtype)
with train_context:
    print("printing dtypes")
    model = ToyModel(num_features, num_out)
    X = torch.randn(batch_size, num_features, device=device)
    print("\tX", X.dtype)
    logits = model(X)
    loss = torch.sum(logits**2)
    print("\tloss", loss.dtype)
    loss.backward()
    print("fc1 grad", model.fc1.weight.grad.dtype)
    print("ln grad", model.ln.weight.grad.dtype)
    print("fc2 grad", model.fc2.weight.grad.dtype)
