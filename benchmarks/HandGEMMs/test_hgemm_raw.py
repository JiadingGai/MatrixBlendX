import torch
from torch.utils.cpp_extension import load_inline

# Read the raw kernel source
with open("hgemm_raw.cu") as f:
    kernel_source = f.read()

# Undefine PyTorch's half-precision macros that conflict with cuda_fp16.h
undef_block = """
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
"""

wrapper = """
#include <torch/extension.h>

torch::Tensor hgemm_tn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16,
                "Inputs must be float16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    TORCH_CHECK(B.size(1) == K, "K dimension mismatch");

    // Kernel writes C in column-major (stride 1 in M, ldC=M)
    // Allocate (N, M) contiguous then transpose -> (M, N) with strides (1, M)
    auto C = torch::empty({N, M}, A.options()).t();

    gemm_tn(M, N, K,
            reinterpret_cast<const half*>(A.data_ptr<at::Half>()), K,
            reinterpret_cast<const half*>(B.data_ptr<at::Half>()), K,
            reinterpret_cast<half*>(C.data_ptr<at::Half>()), M);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return C;
}
"""

cuda_source = undef_block + kernel_source + wrapper
cpp_source = "torch::Tensor hgemm_tn(torch::Tensor A, torch::Tensor B);"

print("Compiling raw kernel...")
module = load_inline(
    name="hgemm_raw",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["hgemm_tn"],
    extra_cuda_cflags=["-O3", "-arch=sm_80", "--expt-relaxed-constexpr"],
    verbose=False,
)
print("Compilation done.\n")


def test_correctness(M, N, K, num_trials=3):
    max_err_all = 0.0
    for trial in range(num_trials):
        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B = torch.randn(N, K, dtype=torch.float16, device="cuda")

        C_ours = module.hgemm_tn(A, B)
        C_ref = torch.matmul(A, B.t())

        abs_err = (C_ours.float() - C_ref.float()).abs()
        max_err = abs_err.max().item()
        max_err_all = max(max_err_all, max_err)

        close = torch.allclose(C_ours, C_ref, rtol=1e-2, atol=1e-1)
        status = "PASS" if close else "FAIL"
        print(f"  trial {trial}: max_abs_err={max_err:.4f}  [{status}]")

    return max_err_all


def benchmark(M, N, K, warmup=10, iters=100):
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")

    for _ in range(warmup):
        module.hgemm_tn(A, B)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        module.hgemm_tn(A, B)
    end.record()
    torch.cuda.synchronize()
    ours_ms = start.elapsed_time(end) / iters

    start.record()
    for _ in range(iters):
        torch.matmul(A, B.t())
    end.record()
    torch.cuda.synchronize()
    cublas_ms = start.elapsed_time(end) / iters

    gflops = 2.0 * M * N * K / 1e9
    ours_tflops = gflops / ours_ms
    cublas_tflops = gflops / cublas_ms

    print(f"  Ours:   {ours_tflops:8.1f} TFLOPS  ({ours_ms:.4f} ms)")
    print(f"  cuBLAS: {cublas_tflops:8.1f} TFLOPS  ({cublas_ms:.4f} ms)")
    print(f"  Ratio:  {ours_tflops/cublas_tflops:.2f}x")


if __name__ == "__main__":
    print("=== Correctness Tests ===")
    for M, N, K in [(128, 128, 64), (256, 256, 128), (1024, 1024, 1024),
                     (4096, 4096, 4096), (5120, 5120, 4096)]:
        print(f"\nM={M}, N={N}, K={K}")
        test_correctness(M, N, K)

    print("\n=== Benchmark (ours vs cuBLAS) ===")
    for W in [2048, 4096, 5120, 8192]:
        print(f"\nM=N=K={W}")
        benchmark(W, W, W)
