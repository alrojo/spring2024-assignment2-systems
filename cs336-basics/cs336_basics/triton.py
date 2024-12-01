import torch
class RMSNormTriton(torch.autograd.Function):
    @statismethod
    def forward(ctx, x, w):
        eps = 1e-8
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        x = x * rms
        return self.weight * x

    def beckward(ctx, grad_out):
        raise NotImplementedError
