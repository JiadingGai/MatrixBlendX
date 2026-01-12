# bench_fp8_cutlass_grouped.py
# FP8 grouped GEMM benchmark (same-NK, variable-M per expert) on H100
# - backend: cuBLAS cublasGemmGroupedBatchedEx (FP8 E4M3/E5M2)
# - includes: correctness, token sweep, warmups, timing with triton.do_bench
#
# Notes:
# - We use the standard column-major trick to compute row-major C = A@B using cuBLAS col-major GEMM:
#     C_col(N,M) = B_col(N,K) @ A_col(K,M)
#   so Aarray points to B (stored as (N,K) row-major == (K,N) col-major)
#      Barray points to A (stored as (M,K) row-major == (K,M) col-major)
#      Carray points to C (stored as (M,N) row-major == (N,M) col-major)
#
# Env knobs:
#   TOKENS_SWEEP="16,32,64,128,256,512,1024,2048,4096,8192"
#   FP8_FMT="e4m3"  or "e5m2"
#   DTYPE_BASELINE="bf16" or "fp16"   (baseline grouped GEMM)
#   G=128 K=2944 N=2944 TOPK=4
#   WARMUP=25 REP=100
#   TEST_TOKENS=128
#
# Example:
#   TORCH_CUDA_ARCH_LIST=9.0a FP8_FMT=e4m3 python bench_fp8_cutlass_grouped.py

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import triton
from torch.utils.cpp_extension import load_inline


# ----------------------------
# Shapes
# ----------------------------
@dataclass(frozen=True)
class MoESpec:
    G: int = int(os.environ.get("G", "128"))
    K: int = int(os.environ.get("K", "2944"))
    N: int = int(os.environ.get("N", "2944"))
    TOPK: int = int(os.environ.get("TOPK", "4"))


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    return default if v == "" else int(v)


def _sync():
    torch.cuda.synchronize()


def _do_bench_ms(fn, warmup: int, rep: int) -> float:
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def _pick_fp8_dtype() -> torch.dtype:
    fmt = os.environ.get("FP8_FMT", "e4m3").lower()
    if fmt in ("e4m3", "e4m3fn", "fp8e4m3"):
        # common on H100
        return torch.float8_e4m3fn
    if fmt in ("e5m2", "fp8e5m2"):
        return torch.float8_e5m2
    raise ValueError(f"Unsupported FP8_FMT={fmt}")


def _pick_baseline_dtype() -> torch.dtype:
    name = os.environ.get("DTYPE_BASELINE", "bf16").lower()
    return torch.bfloat16 if name in ("bf16", "bfloat16") else torch.float16


def _make_cumsum_M(total_M: int, G: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Random routing distribution (packed by expert boundaries).
    Returns:
      cumsum_M_gpu: [G] int32 on GPU
      cumsum_M_cpu: [G] int64 on CPU
      max_M: int
    """
    idx = torch.randint(low=0, high=G, size=(total_M,), device="cpu", dtype=torch.int64)
    counts = torch.bincount(idx, minlength=G).to(torch.int32)
    assert int(counts.sum().item()) == total_M
    cumsum_i32 = torch.cumsum(counts, dim=0)  # CPU int32
    max_M = int(counts.max().item())
    return cumsum_i32.to(device=device, non_blocking=True), cumsum_i32.to(dtype=torch.int64), max_M


# ----------------------------
# cuBLAS grouped extension (supports fp16/bf16/fp8)
# ----------------------------
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <vector>
#include <sstream>
#include <stdexcept>

static inline void cublas_check(cublasStatus_t st, const char* what) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::ostringstream oss;
    oss << "cuBLAS error: " << what << " (status=" << int(st) << ")";
    throw std::runtime_error(oss.str());
  }
}

static inline cudaDataType_t to_cuda_dtype(int code) {
  // 0: fp16, 1: bf16, 2: fp32, 3: fp8 e4m3, 4: fp8 e5m2
  switch (code) {
    case 0: return CUDA_R_16F;
    case 1: return CUDA_R_16BF;
    case 2: return CUDA_R_32F;
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
    case 3: return CUDA_R_8F_E4M3;
    case 4: return CUDA_R_8F_E5M2;
#endif
    default: throw std::runtime_error("bad dtype_code");
  }
}

void grouped_gemm_same_nk(
    torch::Tensor A_ptrs,         // int64 CUDA: pointers to B_g base (col-major view)
    torch::Tensor B_ptrs,         // int64 CUDA: pointers to A_g base (col-major view)
    torch::Tensor C_ptrs,         // int64 CUDA: pointers to C_g base (col-major view)
    torch::Tensor n_array_cpu,    // int32 CPU: n_i = M_g (varies)
    int64_t N, int64_t K,
    int dtype_code,              // input/output dtype code
    bool accumulate) {

  TORCH_CHECK(A_ptrs.is_cuda() && B_ptrs.is_cuda() && C_ptrs.is_cuda(), "ptr arrays must be CUDA");
  TORCH_CHECK(A_ptrs.scalar_type() == torch::kInt64, "A_ptrs must be int64");
  TORCH_CHECK(B_ptrs.scalar_type() == torch::kInt64, "B_ptrs must be int64");
  TORCH_CHECK(C_ptrs.scalar_type() == torch::kInt64, "C_ptrs must be int64");
  TORCH_CHECK(!n_array_cpu.is_cuda(), "n_array_cpu must be CPU");
  TORCH_CHECK(n_array_cpu.scalar_type() == torch::kInt32, "n_array_cpu must be int32");

  const int group_count = (int)n_array_cpu.numel();
  if (group_count == 0) return;

  std::vector<int> group_size(group_count, 1);

  // Column-major GEMM:
  //   C_col(m=N, n=M_g) = A_col(m=N,k=K) @ B_col(k=K,n=M_g)
  std::vector<cublasOperation_t> transa(group_count, CUBLAS_OP_N);
  std::vector<cublasOperation_t> transb(group_count, CUBLAS_OP_N);
  std::vector<int> m_array(group_count, (int)N);
  std::vector<int> n_array(group_count);
  std::vector<int> k_array(group_count, (int)K);
  std::vector<int> lda_array(group_count, (int)N);  // A is (N,K) col-major => lda=N
  std::vector<int> ldb_array(group_count, (int)K);  // B is (K,M) col-major => ldb=K
  std::vector<int> ldc_array(group_count, (int)N);  // C is (N,M) col-major => ldc=N

  auto n_acc = n_array_cpu.accessor<int,1>();
  for (int i = 0; i < group_count; ++i) n_array[i] = n_acc[i];

  std::vector<float> alpha(group_count, 1.0f);
  std::vector<float> beta(group_count, accumulate ? 1.0f : 0.0f);

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublas_check(cublasSetStream(handle, stream), "cublasSetStream");
  cublas_check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode(HOST)");
  (void)cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  const cudaDataType_t Atype = to_cuda_dtype(dtype_code);
  const cudaDataType_t Btype = to_cuda_dtype(dtype_code);
  const cudaDataType_t Ctype = to_cuda_dtype(dtype_code);

  // FP8 usually computes in FP32 accumulation; use COMPUTE_32F.
  const cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

  const void* const* Aarray = reinterpret_cast<const void* const*>(A_ptrs.data_ptr<int64_t>());
  const void* const* Barray = reinterpret_cast<const void* const*>(B_ptrs.data_ptr<int64_t>());
  void* const* Carray = reinterpret_cast<void* const*>(C_ptrs.data_ptr<int64_t>());

  cublas_check(
      cublasGemmGroupedBatchedEx(
          handle,
          transa.data(), transb.data(),
          m_array.data(), n_array.data(), k_array.data(),
          alpha.data(),
          Aarray, Atype, lda_array.data(),
          Barray, Btype, ldb_array.data(),
          beta.data(),
          Carray, Ctype, ldc_array.data(),
          group_count,
          group_size.data(),
          computeType),
      "cublasGemmGroupedBatchedEx");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grouped_gemm_same_nk", &grouped_gemm_same_nk, "cublasGemmGroupedBatchedEx grouped GEMM (same N,K)");
}
"""


_EXT = None


def _load_ext():
    # H100
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    return load_inline(
        name="ggemm_cublas_grouped_fp8_ext",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def _get_ext():
    global _EXT
    if _EXT is None:
        _EXT = _load_ext()
    return _EXT


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        return 0
    if dtype == torch.bfloat16:
        return 1
    if dtype == torch.float32:
        return 2
    if dtype == torch.float8_e4m3fn:
        return 3
    if dtype == torch.float8_e5m2:
        return 4
    raise ValueError(f"Unsupported dtype for extension: {dtype}")


def _plan_ptr_arrays_colmajor_trick(
    a: torch.Tensor,              # [total_M, K] row-major
    b_col: torch.Tensor,          # [G, N, K] row-major (== (K,N) col-major)
    c: torch.Tensor,              # [total_M, N] row-major
    cumsum_M_cpu: torch.Tensor,   # [G] int64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build pointer arrays for non-empty experts only.

    cuBLAS sees column-major GEMM:
      C_col(N,M) = A_col(N,K) @ B_col(K,M)
    We map:
      A_col := B_g (stored as (N,K) row-major)
      B_col := A_slice (stored as (M,K) row-major)
      C_col := C_slice (stored as (M,N) row-major)
    """
    assert a.is_cuda and b_col.is_cuda and c.is_cuda
    assert cumsum_M_cpu.device.type == "cpu" and cumsum_M_cpu.dtype == torch.int64

    G, N, K = b_col.shape
    elem_a = a.element_size()
    elem_b = b_col.element_size()
    elem_c = c.element_size()

    starts = torch.empty_like(cumsum_M_cpu)
    starts[0] = 0
    starts[1:] = cumsum_M_cpu[:-1]

    A_ptrs: List[int] = []
    B_ptrs: List[int] = []
    C_ptrs: List[int] = []
    n_list: List[int] = []

    a_base = a.data_ptr()
    b_base = b_col.data_ptr()
    c_base = c.data_ptr()

    for g in range(G):
        s = int(starts[g].item())
        e = int(cumsum_M_cpu[g].item())
        m_g = e - s
        if m_g <= 0:
            continue

        # byte pointers
        a_ptr = a_base + s * K * elem_a           # A_slice base
        b_ptr = b_base + g * N * K * elem_b       # B_col for expert g
        c_ptr = c_base + s * N * elem_c           # C_slice base

        # cuBLAS Aarray -> b_ptr, Barray -> a_ptr, Carray -> c_ptr
        A_ptrs.append(b_ptr)
        B_ptrs.append(a_ptr)
        C_ptrs.append(c_ptr)
        n_list.append(m_g)

    device = a.device
    A_ptrs_t = torch.tensor(A_ptrs, dtype=torch.int64, device=device)
    B_ptrs_t = torch.tensor(B_ptrs, dtype=torch.int64, device=device)
    C_ptrs_t = torch.tensor(C_ptrs, dtype=torch.int64, device=device)
    n_array_cpu = torch.tensor(n_list, dtype=torch.int32, device="cpu")
    return A_ptrs_t, B_ptrs_t, C_ptrs_t, n_array_cpu


@torch.no_grad()
def grouped_gemm_same_nk_cublas(
    a: torch.Tensor,            # [total_M, K] row-major
    b_col: torch.Tensor,        # [G, N, K] row-major
    cumsum_M_cpu: torch.Tensor, # [G] int64 CPU
    *,
    out: Optional[torch.Tensor] = None,
    accumulate: bool = False,
) -> torch.Tensor:
    """
    Output is row-major [total_M, N], same dtype as a/b_col.
    """
    ext = _get_ext()
    assert a.is_cuda and b_col.is_cuda
    assert a.dtype == b_col.dtype
    dtype = a.dtype
    if out is None:
        out = torch.empty((a.shape[0], b_col.shape[1]), device=a.device, dtype=dtype)
        accumulate = False

    A_ptrs, B_ptrs, C_ptrs, n_array_cpu = _plan_ptr_arrays_colmajor_trick(a, b_col, out, cumsum_M_cpu)
    ext.grouped_gemm_same_nk(
        A_ptrs, B_ptrs, C_ptrs, n_array_cpu,
        int(b_col.shape[1]), int(a.shape[1]),
        _dtype_code(dtype),
        bool(accumulate),
    )
    return out


# ----------------------------
# Correctness + bench
# ----------------------------
def _tokens_list() -> List[int]:
    s = os.environ.get("TOKENS_SWEEP", "")
    if s.strip():
        return [int(x) for x in s.split(",") if x.strip()]
    return [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


@torch.no_grad()
def correctness_once():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    spec = MoESpec()
    device = torch.device("cuda")

    tokens = _env_int("TEST_TOKENS", 128)
    total_M = tokens * spec.TOPK

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # baseline activation/weights in bf16 just to generate fp8
    a0 = torch.randn((total_M, spec.K), device=device, dtype=torch.bfloat16)
    b0 = torch.randn((spec.G, spec.K, spec.N), device=device, dtype=torch.bfloat16)

    fp8_dtype = _pick_fp8_dtype()
    a = a0.to(fp8_dtype)
    b = b0.to(fp8_dtype)
    b_col = b.transpose(1, 2).contiguous()  # [G, N, K] row-major

    _, cumsum_cpu, max_M = _make_cumsum_M(total_M, spec.G, device)
    # run cuBLAS grouped fp8
    _sync()
    out = grouped_gemm_same_nk_cublas(a, b_col, cumsum_cpu)
    _sync()

    # reference in fp32 from fp8 values (no extra scaling)
    starts = torch.empty_like(cumsum_cpu)
    starts[0] = 0
    starts[1:] = cumsum_cpu[:-1]
    ref = torch.empty((total_M, spec.N), device=device, dtype=torch.float32)
    for g in range(spec.G):
        s = int(starts[g].item())
        e = int(cumsum_cpu[g].item())
        if e <= s:
            continue
        ref[s:e] = a[s:e].float() @ b[g].float()

    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-1)
    print(f"[OK] correctness passed (FP8 GEMM vs fp32 ref from FP8 tensors). max_M={max_M}")


@torch.no_grad()
def bench_sweep():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    spec = MoESpec()
    device = torch.device("cuda")

    warmup = _env_int("WARMUP", 25)
    rep = _env_int("REP", 100)

    fp8_dtype = _pick_fp8_dtype()
    base_dtype = _pick_baseline_dtype()

    # allocate weights once
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    b_base = torch.randn((spec.G, spec.K, spec.N), device=device, dtype=base_dtype)
    b_fp8 = b_base.to(fp8_dtype)
    b_fp8_col = b_fp8.transpose(1, 2).contiguous()  # [G, N, K]
    b_base_col = b_base.transpose(1, 2).contiguous()  # [G, N, K]

    # compile extension once
    _ = _get_ext()

    print(f"\n[config] G={spec.G} K={spec.K} N={spec.N} TOPK={spec.TOPK} FP8={fp8_dtype} baseline={base_dtype}")
    for tokens in _tokens_list():
        total_M = tokens * spec.TOPK

        # routing
        _, cumsum_cpu, max_M = _make_cumsum_M(total_M, spec.G, device)

        # activations
        a_base = torch.randn((total_M, spec.K), device=device, dtype=base_dtype)
        a_fp8 = a_base.to(fp8_dtype)

        # outputs
        out_fp8 = torch.empty((total_M, spec.N), device=device, dtype=fp8_dtype)
        out_base = torch.empty((total_M, spec.N), device=device, dtype=base_dtype)

        # warmup (cuBLAS heuristics + caches)
        for _ in range(10):
            _ = grouped_gemm_same_nk_cublas(a_fp8, b_fp8_col, cumsum_cpu, out=out_fp8, accumulate=False)
        _sync()

        ms_fp8 = _do_bench_ms(
            lambda: grouped_gemm_same_nk_cublas(a_fp8, b_fp8_col, cumsum_cpu, out=out_fp8, accumulate=False),
            warmup=warmup, rep=rep
        )
        _sync()

        # baseline bf16/fp16 grouped
        for _ in range(10):
            _ = grouped_gemm_same_nk_cublas(a_base, b_base_col, cumsum_cpu, out=out_base, accumulate=False)
        _sync()

        ms_base = _do_bench_ms(
            lambda: grouped_gemm_same_nk_cublas(a_base, b_base_col, cumsum_cpu, out=out_base, accumulate=False),
            warmup=warmup, rep=rep
        )
        _sync()

        flops = 2.0 * float(total_M) * float(spec.K) * float(spec.N)
        t_fp8 = ms_fp8 * 1e-3
        t_base = ms_base * 1e-3
        gflops_fp8 = flops / t_fp8 / 1e9
        gflops_base = flops / t_base / 1e9
        speedup = ms_base / ms_fp8

        print(f"\n[tokens={tokens} total_M={total_M} max_M={max_M}]")
        print(f"  FP8 grouped   : {ms_fp8:.3f} ms  ({gflops_fp8/1000:.3f} TFLOP/s)")
        print(f"  Base grouped  : {ms_base:.3f} ms  ({gflops_base/1000:.3f} TFLOP/s)  dtype={base_dtype}")
        print(f"  speedup (base/FP8): {speedup:.2f}x")


def main():
    correctness_once()
    bench_sweep()


if __name__ == "__main__":
    main()

