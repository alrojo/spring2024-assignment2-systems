import torch, triton
import triton.language as tl

def _sum_all_but_last(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 1:
        return x
    else:
        return x.sum(dim=tuple(range(len(x.shape)-1)), keepdim=True)

@triton.jit
def _rms_fwd(
        x_ptr : tl.pointer_type,
        weight_ptr : tl.pointer_type,
        output_ptr : tl.pointer_type,
        H : tl.uint32,
        eps : tl.float32,
        BLOCK_SIZE: tl.constexpr):
    # setup pointers
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * H
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets

    # load in data
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)

    # compute
    norm = tl.sqrt(tl.sum(row*row)/H + eps)
    output = row/norm*weight

    # write back
    output_start_ptr = output_ptr + row_idx * H
    output_ptrs = output_start_ptr + offsets
    tl.store(output_ptrs, output, mask=mask)

@triton.jit
def _rms_bwd(
        x_ptr : tl.pointer_type,
        g_ptr : tl.pointer_type,
        grad_out_ptr : tl.pointer_type,
        grad_x_ptr : tl.pointer_type,
        grad_g_ptr : tl.pointer_type,
        H : tl.uint32,
        eps : tl.float32,
        BLOCK_SIZE: tl.constexpr):

    row_idx = tl.program_id(0)
    # setup input pointers
    x_start_ptr = x_ptr + row_idx * H
    grad_out_start_ptr = grad_out_ptr + row_idx * H

    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_start_ptr + offsets
    g_ptrs = g_ptr + offsets
    grad_out_ptrs = grad_out_start_ptr + offsets

    # load in data
    mask = offsets < H
    x = tl.load(x_ptrs, mask=mask, other=0)
    g = tl.load(g_ptrs, mask=mask, other=0)
    grad_out = tl.load(grad_out_ptrs, mask=mask, other=0)

    # pre compute
    A = tl.sqrt(tl.sum(x*x)/H + eps)

    # compute grad_g
    g_grad = grad_out*(x/A)

    # compute grad_x
    part1 = grad_out * g / A  
    part2 = x/(H*A*A*A) * tl.sum(grad_out * x * g)
    x_grad = part1-part2

    # write back g
    grad_g_start_ptr = grad_g_ptr + row_idx * H
    tl.store(grad_g_start_ptr + offsets, g_grad, mask=mask)

    # write back x
    grad_x_start_ptr = grad_x_ptr + row_idx * H
    tl.store(grad_x_start_ptr + offsets, x_grad, mask=mask)

class RMSNormAutogradFuncTriton(torch.autograd.Function):
    eps = 1e-5
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        H, output_dims = x.shape[-1], x.shape
        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        assert len(w.shape) == 1 and w.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and w.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous() and w.is_contiguous, "Our pointer arithmetic will assume contiguous x and w"

        y = torch.empty(output_dims, device=x.device)
        n_rows = y.numel() // H
        _rms_fwd[(n_rows, )](
            x, w, y, H, eps=RMSNormAutogradFuncTriton.eps,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return y 

    def backward(ctx, grad_out):
        x, g = ctx.saved_tensors
        H = x.shape[-1]
        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        assert len(g.shape) == 1 and g.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and g.is_cuda and grad_out.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous() and g.is_contiguous and grad_out.is_contiguous, "Our pointer arithmetic will assume contiguous x, g, grad_out"

        dx = torch.empty_like(x)
        dg = torch.empty_like(x)
        n_rows = int(grad_out.numel() / H)
        _rms_bwd[(n_rows, )](
            x, g, grad_out, dx, dg, H, eps=RMSNormAutogradFuncTriton.eps,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return dx, _sum_all_but_last(dg)

class RMSNormTriton(torch.nn.Module):
    def __init__(self, H: int):
        super(RMSNormTriton, self).__init__()
        self.g = torch.nn.Parameter(torch.randn(H))

    def forward(self, x):
        return RMSNormAutogradFuncTriton.apply(x, self.g)

class RMSNormAutogradFuncTorch(torch.autograd.Function):
    eps = 1e-5
    def _jvp_g(grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + RMSNormAutogradFuncTorch.eps)
        nabla_y_g = x/norm
        nabla_g_L = grad_output*nabla_y_g
        g_grad = nabla_g_L.view(-1, nabla_g_L.size(-1)).sum(dim=0, keepdim=False)
        return g_grad

    def _jvp_x(grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor):
        H = grad_output.shape[-1]
        f_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + RMSNormAutogradFuncTorch.eps)
        part1 = grad_output * g/f_x
        part2 = x/(H*f_x**3) * torch.sum(grad_output * x * g, dim=-1, keepdim=True)
        grad_x = part1-part2
        return grad_x

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        eps = 1e-8
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        x = x * rms
        return w * x
    @staticmethod
    def backward(ctx, grad_out):
        x, g = ctx.saved_tensors
        grad_x = RMSNormAutogradFuncTorch._jvp_x(grad_out, x, g)
        grad_g = RMSNormAutogradFuncTorch._jvp_g(grad_out, x, g)
        return grad_x, grad_g

if __name__ == '__main__':
    device = torch.device('cuda')

    # Test RMSNorm
    torch.manual_seed(0)
    x = torch.randn((2, 4, 3), requires_grad=True, device=device)
    g = torch.ones(3, requires_grad=True, device=device)
    y_triton = RMSNormAutogradFuncTriton.apply(x, g)
    y_torch = RMSNormAutogradFuncTorch.apply(x, g)
    #print('x matrices:\n', x)
    #print('g vector:\n', g)
    #print('y triton:\n', y_triton)
    #print('y torch:\n', y_torch)
    # Test backward pass with a custom grad_out
    grad_out = torch.randn_like(y_triton)
    dx_triton, dg_triton = torch.autograd.grad(y_triton, (x, g), grad_out, retain_graph=True)
    dx_torch, dg_torch = torch.autograd.grad(y_torch, (x, g), grad_out, retain_graph=True)
    print('\ngrad_out:\n', grad_out)
    print('dx_triton:\n', dx_triton)
    print('dg_triton:\n', dg_triton)
    print('dx_torch:\n', dx_torch)
    print('dg_torch:\n', dg_torch)
