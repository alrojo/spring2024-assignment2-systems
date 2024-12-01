import torch, triton
import triton.language as tl

def rms_fwd(
        x_ptr : tl.pointer_type,
        weight_ptr : tl.pointer_type,
        x_row_stride : tl.uint32,
        output_ptr : tl.pointer_type,
        H : tl.uint32,
        eps : tl.float32,
        BLOCK_SIZE: tl.constexpr):
    # setup pointers
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
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
    output_start_ptr = output_ptr + row_idx * x_row_stride
    output_ptrs = output_start_ptr + offsets
    tl.store(output_ptrs, output, mask=mask)
    

class RMSNormAutogradFunc(torch.autograd.Function):
    eps = 1e-5
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        H, output_dims = x.shape[-1], x.shape
        assert len(weight.shape) == 1 and weight.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous() and g.is_contiguous, "Our pointer arithmetic will assume contiguous x and g"

        y = torch.empty(output_dims, device=x.device)
        n_rows = int(y.shape[0])
        rms_fwd[(n_rows, )](
            x, w, x.stride(0), y, H, eps=RMSNormAutogradFunc.eps,
            num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE)
        return y 

class RMSNormTriton(torch.nn.Module):
    def __init__(self, H: int):
        super(RMSNormTriton, self).__init__()
        self.g = torch.nn.Parameter(torch.randn(H))

    def forward(self, x):
        return RMSNormAutogradFunc.apply(x, self.g)

def _sum_all_but_last(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=tuple(range(len(x.shape)-1)), keepdim=True)
class RMSNormAutogradFuncTorch(torch.autograd.Function):
    eps = 1e-5
    def _jvp_g(grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + RMSNormAutogradFuncTorch.eps)
        nabla_y_g = x/norm
        nabla_g_L = grad_output*nabla_y_g
        g_grad = nabla_g_L.view(-1, nabla_g_L.size(-1)).sum(dim=0, keepdim=False)
        return g_grad
