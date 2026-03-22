
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ============================================================
# Reference implementation from the prompt
# ============================================================

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=1)

batch_size = 4096
dim = 393216

def get_inputs(device="cuda", dtype=torch.float32):
    x = torch.rand(batch_size, dim, device=device, dtype=dtype)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed


# ============================================================
# Benchmark config
# ============================================================

DTYPE = torch.float32
DEVICE = "cuda"
WARMUP = 10
ITERS = 30

assert torch.cuda.is_available(), "CUDA is required to run this benchmark."


# ============================================================
# CUDA extension: forward-only online softmax
#
# Online statistics per row:
#   m = running max
#   l = running sum of exp(x - m)
#
# Combine operator:
#   combine((m1, l1), (m2, l2)) =
#       (m, l1 * exp(m1 - m) + l2 * exp(m2 - m))
#   where m = max(m1, m2)
#
# Final output:
#   y_j = exp(x_j - m) / l
#
# This uses:
#   - pass 1: read x once to compute row (m, l)
#   - pass 2: read x again and write y
# So large-tensor traffic is 2 reads + 1 write.
# ============================================================
# Parallelization strategy:
# 1. block-level parallelism: one block per row
# 2. thread-level parallelism: block-strided partition over columns
# 3. intra-block communication: shared-memory reduction of online-softmax stats
# 4. work per thread: about cols / 256, not 256
# ============================================================

CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor online_softmax_forward(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &online_softmax_forward, "Online softmax forward (CUDA)");
}
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x)

struct OnlineStats {
    float m;
    float l;
};

__device__ __forceinline__ OnlineStats combine_stats(OnlineStats a, OnlineStats b) {
    OnlineStats out;
    out.m = a.m > b.m ? a.m : b.m;
    out.l = a.l * expf(a.m - out.m) + b.l * expf(b.m - out.m);
    return out;
}

template <int BLOCK_SIZE>
__global__ void online_softmax_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int64_t rows,
    int64_t cols
) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_x = x + static_cast<int64_t>(row) * cols;
    float* row_y = y + static_cast<int64_t>(row) * cols;

    OnlineStats local;
    local.m = -CUDART_INF_F;
    local.l = 0.0f;

    // Pass 1: compute online stats for this thread's strided columns.
    for (int64_t col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        const float v = row_x[col];
        const float new_m = local.m > v ? local.m : v;
        local.l = local.l * expf(local.m - new_m) + expf(v - new_m);
        local.m = new_m;
    }

    __shared__ float smem_m[BLOCK_SIZE];
    __shared__ float smem_l[BLOCK_SIZE];
    smem_m[threadIdx.x] = local.m;
    smem_l[threadIdx.x] = local.l;
    __syncthreads();

    // Block-wide reduction using the associative online-softmax combine.
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            OnlineStats a{smem_m[threadIdx.x], smem_l[threadIdx.x]};
            OnlineStats b{smem_m[threadIdx.x + stride], smem_l[threadIdx.x + stride]};
            OnlineStats c = combine_stats(a, b);
            smem_m[threadIdx.x] = c.m;
            smem_l[threadIdx.x] = c.l;
        }
        __syncthreads();
    }

    const float row_m = smem_m[0];
    const float inv_row_l = 1.0f / smem_l[0];

    // Pass 2: write normalized output.
    for (int64_t col = threadIdx.x; col < cols; col += BLOCK_SIZE) {
        row_y[col] = expf(row_x[col] - row_m) * inv_row_l;
    }
}

torch::Tensor online_softmax_forward(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch, dim]");

    const auto rows = x.size(0);
    const auto cols = x.size(1);
    auto y = torch::empty_like(x);

    constexpr int BLOCK_SIZE = 256;
    const dim3 grid(rows);
    const dim3 block(BLOCK_SIZE);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cudaStream_t stream = at::cuda::getDefaultCUDAStream(x.get_device());

    online_softmax_forward_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        rows,
        cols
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
"""

online_softmax_ext = load_inline(
    name="online_softmax_ext_refcmp_v1",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    functions=None,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    with_cuda=True,
    verbose=True,
)


# ============================================================
# Custom CUDA module
# ============================================================

class OnlineSoftmaxCUDA(nn.Module):
    """
    Custom CUDA online softmax.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return online_softmax_ext.forward(x.contiguous())


# ============================================================
# Benchmark helpers
# ============================================================

@torch.no_grad()
def benchmark_ms(fn, x, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        _ = fn(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iters):
        start.record()
        _ = fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    avg = sum(times) / len(times)
    return avg, min(times), max(times)

@torch.no_grad()
def correctness_check():
    print("=" * 80)
    print("Correctness check")
    print("=" * 80)

    x = torch.randn(32, 8192, device=DEVICE, dtype=DTYPE)

    ref_model = Model().to(DEVICE)
    cuda_model = OnlineSoftmaxCUDA().to(DEVICE)

    y_ref = ref_model(x)
    y_cuda = cuda_model(x)

    max_abs = (y_cuda - y_ref).abs().max().item()
    mean_abs = (y_cuda - y_ref).abs().mean().item()
    max_row_sum_err = (y_cuda.sum(dim=1) - 1.0).abs().max().item()

    print(f"max abs diff vs reference Model : {max_abs:.6e}")
    print(f"mean abs diff vs reference Model: {mean_abs:.6e}")
    print(f"max |row_sum - 1|               : {max_row_sum_err:.6e}")

@torch.no_grad()
def main():
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    correctness_check()

    print()
    print("=" * 80)
    print("Benchmark setup")
    print("=" * 80)
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Shape      : [{batch_size}, {dim}]")
    print(f"Dtype      : {DTYPE}")
    print(f"Warmup     : {WARMUP}")
    print(f"Iters      : {ITERS}")

    x = get_inputs(device=DEVICE, dtype=DTYPE)[0]

    ref_model = Model().to(DEVICE)
    cuda_model = OnlineSoftmaxCUDA().to(DEVICE)

    print()
    print("=" * 80)
    print("One-shot correctness check at benchmark shape")
    print("=" * 80)
    y_ref = ref_model(x)
    y_cuda = cuda_model(x)

    max_abs = (y_cuda - y_ref).abs().max().item()
    mean_abs = (y_cuda - y_ref).abs().mean().item()
    max_row_sum_err = (y_cuda.sum(dim=1) - 1.0).abs().max().item()

    print(f"max abs diff vs reference Model : {max_abs:.6e}")
    print(f"mean abs diff vs reference Model: {mean_abs:.6e}")
    print(f"max |row_sum - 1|               : {max_row_sum_err:.6e}")

    print()
    print("=" * 80)
    print("Latency benchmark")
    print("=" * 80)
    ref_avg, ref_min, ref_max = benchmark_ms(ref_model, x)
    print(f"reference Model (torch.softmax) : avg={ref_avg:.3f} ms  min={ref_min:.3f} ms  max={ref_max:.3f} ms")

    cuda_avg, cuda_min, cuda_max = benchmark_ms(cuda_model, x)
    print(f"custom online softmax (CUDA)    : avg={cuda_avg:.3f} ms  min={cuda_min:.3f} ms  max={cuda_max:.3f} ms")

    print()
    print("=" * 80)
    print("Relative performance")
    print("=" * 80)
    print(f"speedup over reference Model    : {ref_avg / cuda_avg:.3f}x")

if __name__ == "__main__":
    main()
