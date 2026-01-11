# test1_fixed.py
# Standalone benchmark: VeOmni Triton group_gemm_same_nk vs cuBLAS cublasGemmGroupedBatchedEx
# + correctness + token sweep + warmups.
#
# Fix for your error:
#   cublasGemmGroupedBatchedEx does NOT support BF16/FP16 inputs with FP32 C (status=15 NOT_SUPPORTED).
#   If you pass fp32 c_init, we compute GEMM into a bf16 temp, then add into fp32 c.

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import triton
from torch.utils.cpp_extension import load_inline


# ----------------------------
# Model dims (your GPT-OSS-120B MoE guess)
# ----------------------------
@dataclass(frozen=True)
class GPTOSS120BMoESpec:
    G: int = 128
    K: int = 2880
    N: int = 2880
    TOPK: int = 4


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    return default if v == "" else int(v)


def _sync():
    torch.cuda.synchronize()


def _do_bench_ms(fn, warmup: int = 25, rep: int = 100) -> float:
    # triton.testing.do_bench already runs warmups; we also do explicit pre-warm for compilation elsewhere.
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def _bytes_per_elem(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(dtype)


def _estimate_bytes(G: int, total_M: int, K: int, N: int, dtype: torch.dtype) -> int:
    # very rough (A+B+C plus slack)
    bpe = _bytes_per_elem(dtype)
    return int((G * K * N + total_M * K + total_M * N) * bpe * 1.2)


def _make_cumsum_M(total_M: int, G: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Returns:
      cumsum_M_gpu: [G] int32 on GPU
      cumsum_M_cpu: [G] int64 on CPU (for pointer planning)
      max_M: int
    """
    idx = torch.randint(low=0, high=G, size=(total_M,), device="cpu", dtype=torch.int64)
    counts = torch.bincount(idx, minlength=G).to(torch.int32)
    assert int(counts.sum().item()) == total_M
    cumsum_cpu_i32 = torch.cumsum(counts, dim=0)  # CPU int32
    max_M = int(counts.max().item())
    cumsum_gpu = cumsum_cpu_i32.to(device=device, non_blocking=True)
    cumsum_cpu = cumsum_cpu_i32.to(dtype=torch.int64)  # CPU int64
    return cumsum_gpu, cumsum_cpu, max_M


@torch.no_grad()
def _ref_grouped_gemm_same_nk(
    a: torch.Tensor,          # [total_M, K] packed by expert boundaries
    b: torch.Tensor,          # [G, K, N]
    cumsum_M_cpu: torch.Tensor,  # [G] int64 CPU
    *,
    accumulate_to_c: bool,
    c_init: Optional[torch.Tensor],
    fp32_ref: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Reference: for each expert g, compute a_g @ b_g and write to corresponding slice.
    """
    assert a.is_cuda and b.is_cuda
    G, K, N = b.shape
    total_M = a.shape[0]

    out = torch.empty((total_M, N), device=a.device, dtype=out_dtype)

    starts = torch.empty_like(cumsum_M_cpu)
    starts[0] = 0
    starts[1:] = cumsum_M_cpu[:-1]

    for g in range(G):
        s = int(starts[g].item())
        e = int(cumsum_M_cpu[g].item())
        if e <= s:
            continue
        a_g = a[s:e]          # [M_g, K]
        b_g = b[g]            # [K, N]

        if fp32_ref:
            y = a_g.float() @ b_g.float()
        else:
            y = (a_g @ b_g).float()

        if accumulate_to_c:
            assert c_init is not None
            y = y + c_init[s:e].float()

        out[s:e] = y.to(dtype=out_dtype)
    return out


# ----------------------------
# cuBLAS GroupedBatchedEx extension
# ----------------------------
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

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

static inline cudaDataType_t to_cuda_dtype(int dtype_code) {
  // 0: fp16, 1: bf16, 2: fp32
  switch (dtype_code) {
    case 0: return CUDA_R_16F;
    case 1: return CUDA_R_16BF;
    case 2: return CUDA_R_32F;
    default: throw std::runtime_error("bad dtype_code");
  }
}

void grouped_gemm_same_nk(
    torch::Tensor A_ptrs,  // int64 CUDA: pointers to B_g base (see mapping below)
    torch::Tensor B_ptrs,  // int64 CUDA: pointers to A_g base
    torch::Tensor C_ptrs,  // int64 CUDA: pointers to C_g base (output slice)
    torch::Tensor n_array_cpu, // int32 CPU: n_i (varies per problem) = M_g
    int64_t N, int64_t K,
    int dtype_code,
    bool accumulate) {

  TORCH_CHECK(A_ptrs.is_cuda() && B_ptrs.is_cuda() && C_ptrs.is_cuda(), "ptr arrays must be CUDA");
  TORCH_CHECK(A_ptrs.scalar_type() == torch::kInt64, "A_ptrs must be int64");
  TORCH_CHECK(B_ptrs.scalar_type() == torch::kInt64, "B_ptrs must be int64");
  TORCH_CHECK(C_ptrs.scalar_type() == torch::kInt64, "C_ptrs must be int64");

  TORCH_CHECK(!n_array_cpu.is_cuda(), "n_array_cpu must be CPU");
  TORCH_CHECK(n_array_cpu.scalar_type() == torch::kInt32, "n_array_cpu must be int32");

  const int group_count = (int)n_array_cpu.numel();
  if (group_count == 0) return;

  // All group_size=1
  std::vector<int> group_size(group_count, 1);

  // For row-major C = A_row(M,K) @ B_row(K,N), we compute column-major:
  //   C_col(N,M) = B_col(N,K) @ A_col(K,M)
  // So for cuBLAS (column-major):
  //   m = N (constant), n = M_g (varies), k = K (constant)
  //   A = B_g (original), lda = N
  //   B = A_g (slice),    ldb = K
  //   C = C_g (slice),    ldc = N
  std::vector<cublasOperation_t> transa(group_count, CUBLAS_OP_N);
  std::vector<cublasOperation_t> transb(group_count, CUBLAS_OP_N);
  std::vector<int> m_array(group_count, (int)N);
  std::vector<int> n_array(group_count);
  std::vector<int> k_array(group_count, (int)K);
  std::vector<int> lda_array(group_count, (int)N);
  std::vector<int> ldb_array(group_count, (int)K);
  std::vector<int> ldc_array(group_count, (int)N);

  auto n_acc = n_array_cpu.accessor<int,1>();
  for (int i = 0; i < group_count; ++i) {
    n_array[i] = n_acc[i];
  }

  std::vector<float> alpha(group_count, 1.0f);
  std::vector<float> beta(group_count, accumulate ? 1.0f : 0.0f);

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublas_check(cublasSetStream(handle, stream), "cublasSetStream");

  // GroupedBatchedEx doesn't support device pointer mode (docs mention NOT_SUPPORTED). Keep HOST.
  cublas_check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode(HOST)");

  // Enable tensor cores where applicable
  (void)cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  const cudaDataType_t Atype = to_cuda_dtype(dtype_code);
  const cudaDataType_t Btype = to_cuda_dtype(dtype_code);
  const cudaDataType_t Ctype = to_cuda_dtype(dtype_code);

  // IMPORTANT: cublasGemmGroupedBatchedEx (compute=32F) generally requires Ctype match A/B for 16-bit paths.
  // If caller wants fp32 accumulation/output with bf16 inputs, do it outside with a bf16 temp.
  if ((dtype_code == 0 || dtype_code == 1) && Ctype == CUDA_R_32F) {
    throw std::runtime_error("GroupedBatchedEx does not support FP16/BF16 inputs with FP32 C. Use a BF16/FP16 temp then cast/add.");
  }

  const cublasComputeType_t computeType =
      (dtype_code == 2) ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F;

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


def _load_ext():
    # Speed up compilation on H100 if user didn't set it.
    # (safe no-op if already set)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")

    return load_inline(
        name="ggemm_cublas_grouped_ext",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


_EXT = None


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
    raise ValueError(dtype)


def _plan_ptr_arrays(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    cumsum_M_cpu: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build pointer arrays for non-empty experts only.

    We call cuBLAS in column-major computing:
      C_col(N,M) = B_col(N,K) @ A_col(K,M)
    So:
      A_ptrs -> pointer to B_g (original b[g]) base
      B_ptrs -> pointer to A slice base
      C_ptrs -> pointer to C slice base
      n_array -> M_g (varies)
    """
    assert not cumsum_M_cpu.is_cuda
    assert cumsum_M_cpu.dtype == torch.int64

    G, K, N = b.shape
    elem_a = a.element_size()
    elem_b = b.element_size()
    elem_c = c.element_size()
    assert elem_a == elem_b, "expect a and b same element size"
    assert elem_c == c.element_size()

    starts = torch.empty_like(cumsum_M_cpu)
    starts[0] = 0
    starts[1:] = cumsum_M_cpu[:-1]

    A_ptrs: List[int] = []
    B_ptrs: List[int] = []
    C_ptrs: List[int] = []
    n_list: List[int] = []

    a_base = a.data_ptr()
    b_base = b.data_ptr()
    c_base = c.data_ptr()

    for g in range(G):
        s = int(starts[g].item())
        e = int(cumsum_M_cpu[g].item())
        m_g = e - s
        if m_g <= 0:
            continue
        # pointers (byte addressing)
        a_ptr = a_base + s * K * elem_a
        b_ptr = b_base + g * K * N * elem_b
        c_ptr = c_base + s * N * elem_c

        # Mapping note:
        # cuBLAS Aarray points to B_g (original), Barray points to A_slice.
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
def group_gemm_same_nk_cublas_grouped(
    a: torch.Tensor,
    b: torch.Tensor,
    cumsum_M_cpu: torch.Tensor,
    *,
    c: Optional[torch.Tensor],
    accumulate_to_c: bool,
) -> torch.Tensor:
    """
    If c is fp16/bf16: we run grouped GEMM with beta=1 for accumulate.
    If c is fp32: GroupedBatchedEx doesn't support bf16/fp16->fp32 C, so:
      tmp(bf16/fp16) = A@B
      c_fp32 += tmp.float()
    """
    assert a.is_cuda and b.is_cuda
    G, K, N = b.shape
    total_M = a.shape[0]

    ext = _get_ext()
    in_dtype = a.dtype
    assert in_dtype in (torch.float16, torch.bfloat16)

    if c is None:
        c_out = torch.empty((total_M, N), device=a.device, dtype=in_dtype)
        accumulate_to_c = False
    else:
        c_out = c

    # Fast path: c_out is same dtype as inputs => supported by GroupedBatchedEx
    if c_out.dtype == in_dtype:
        A_ptrs, B_ptrs, C_ptrs, n_array_cpu = _plan_ptr_arrays(a, b, c_out, cumsum_M_cpu)
        ext.grouped_gemm_same_nk(
            A_ptrs, B_ptrs, C_ptrs, n_array_cpu,
            N, K,
            _dtype_code(in_dtype),
            bool(accumulate_to_c),
        )
        return c_out

    # If user passed fp32 c (Option B style), do GEMM into temp then add.
    if c_out.dtype == torch.float32 and in_dtype in (torch.float16, torch.bfloat16):
        tmp = torch.empty((total_M, N), device=a.device, dtype=in_dtype)
        A_ptrs, B_ptrs, C_ptrs, n_array_cpu = _plan_ptr_arrays(a, b, tmp, cumsum_M_cpu)
        # tmp = A@B  (no accumulate)
        ext.grouped_gemm_same_nk(
            A_ptrs, B_ptrs, C_ptrs, n_array_cpu,
            N, K,
            _dtype_code(in_dtype),
            False,
        )
        # c_fp32 += tmp (cast)
        c_out.add_(tmp.float())
        return c_out

    raise RuntimeError(f"Unsupported c dtype for grouped: c.dtype={c_out.dtype}, a.dtype={in_dtype}")


# ----------------------------
# Optional: VeOmni Triton kernel baseline
# ----------------------------
try:
    from veomni.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk as veomni_triton_group_gemm
except Exception as e:
    veomni_triton_group_gemm = None
    print(f"[WARN] veomni triton import failed: {e}")


# ----------------------------
# Bench + correctness
# ----------------------------
def _pick_dtype() -> torch.dtype:
    name = os.environ.get("GPTOSS_DTYPE", "bf16").lower()
    return torch.bfloat16 if name in ("bf16", "bfloat16") else torch.float16


def _check_mem(spec: GPTOSS120BMoESpec, total_M: int, dtype: torch.dtype) -> bool:
    need = _estimate_bytes(spec.G, total_M, spec.K, spec.N, dtype)
    free, _total = torch.cuda.mem_get_info()
    if need > free:
        print(f"[SKIP] Not enough GPU memory: need~{need/1e9:.2f}GB, free~{free/1e9:.2f}GB")
        return False
    return True


def correctness_once(tokens: int):
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    spec = GPTOSS120BMoESpec()
    device = torch.device("cuda")
    dtype = _pick_dtype()

    total_M = tokens * spec.TOPK
    if not _check_mem(spec, total_M, dtype):
        return

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    cumsum_M_gpu, cumsum_M_cpu, max_M = _make_cumsum_M(total_M, spec.G, device)

    a = torch.randn((total_M, spec.K), device=device, dtype=dtype)
    b = torch.randn((spec.G, spec.K, spec.N), device=device, dtype=dtype)

    # ---- cuBLAS grouped: non-accumulate ----
    _sync()
    out_cublas = group_gemm_same_nk_cublas_grouped(
        a, b, cumsum_M_cpu,
        c=None,
        accumulate_to_c=False,
    )
    _sync()

    ref = _ref_grouped_gemm_same_nk(
        a, b, cumsum_M_cpu,
        accumulate_to_c=False,
        c_init=None,
        fp32_ref=True,
        out_dtype=dtype,
    )
    torch.testing.assert_close(out_cublas, ref, rtol=1e-2, atol=2e-2)

    # ---- cuBLAS grouped: accumulate (Option B style fp32 c_init) ----
    c_init_fp32 = torch.randn((total_M, spec.N), device=device, dtype=torch.float32)
    c_init_ref = c_init_fp32.clone()

    _sync()
    out_acc = group_gemm_same_nk_cublas_grouped(
        a, b, cumsum_M_cpu,
        c=c_init_fp32,
        accumulate_to_c=True,
    )
    _sync()

    ref_acc = _ref_grouped_gemm_same_nk(
        a, b, cumsum_M_cpu,
        accumulate_to_c=True,
        c_init=c_init_ref,
        fp32_ref=True,
        out_dtype=torch.float32,
    )
    torch.testing.assert_close(out_acc, ref_acc, rtol=1e-2, atol=2e-2)

    print("[OK] cuBLAS grouped correctness passed (non-acc + accumulate-to-fp32 via temp+add)")


@torch.no_grad()
def bench_sweep():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    spec = GPTOSS120BMoESpec()
    device = torch.device("cuda")
    dtype = _pick_dtype()

    # Token sweep (override with env if you want)
    # e.g. GPTOSS_SWEEP="128,256,512,1024,2048,4096"
    sweep_env = os.environ.get("GPTOSS_SWEEP", "")
    if sweep_env:
        tokens_list = [int(x) for x in sweep_env.split(",") if x.strip()]
    else:
        tokens_list = [128, 256, 512, 1024, 2048, 4096]

    # Allocate B once (dominant memory)
    print(f"[alloc] b: [{spec.G},{spec.K},{spec.N}] dtype={dtype} ...")
    b = torch.randn((spec.G, spec.K, spec.N), device=device, dtype=dtype)

    # quick compile warmup (extension)
    _ = _get_ext()

    for tokens in tokens_list:
        total_M = tokens * spec.TOPK
        if not _check_mem(spec, total_M, dtype):
            continue

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        cumsum_M_gpu, cumsum_M_cpu, max_M = _make_cumsum_M(total_M, spec.G, device)
        a = torch.randn((total_M, spec.K), device=device, dtype=dtype)

        # ----- cuBLAS grouped bench -----
        c_out = torch.empty((total_M, spec.N), device=device, dtype=dtype)

        # Build pointer plan once per tokens (reused across timing reps)
        A_ptrs, B_ptrs, C_ptrs, n_array_cpu = _plan_ptr_arrays(a, b, c_out, cumsum_M_cpu)
        ext = _get_ext()

        # explicit warmups (compile + cache + cuBLAS heuristics)
        for _ in range(10):
            ext.grouped_gemm_same_nk(A_ptrs, B_ptrs, C_ptrs, n_array_cpu, spec.N, spec.K, _dtype_code(dtype), False)
        _sync()

        ms_cublas = _do_bench_ms(
            lambda: ext.grouped_gemm_same_nk(A_ptrs, B_ptrs, C_ptrs, n_array_cpu, spec.N, spec.K, _dtype_code(dtype), False),
            warmup=25, rep=100
        )
        _sync()

        # ----- VeOmni Triton bench (if available) -----
        ms_triton = None
        if veomni_triton_group_gemm is not None:
            for _ in range(10):
                _ = veomni_triton_group_gemm(a, b, cumsum_M_gpu, max_M, activation=None, c=None)
            _sync()
            ms_triton = _do_bench_ms(lambda: veomni_triton_group_gemm(a, b, cumsum_M_gpu, max_M, activation=None, c=None),
                                     warmup=25, rep=100)
            _sync()

        # ----- Reference per-expert loop (slow; fewer reps) -----
        def ref_loop_native():
            out = torch.empty((total_M, spec.N), device=device, dtype=dtype)
            starts = torch.empty_like(cumsum_M_cpu)
            starts[0] = 0
            starts[1:] = cumsum_M_cpu[:-1]
            for g in range(spec.G):
                s = int(starts[g].item())
                e = int(cumsum_M_cpu[g].item())
                if e <= s:
                    continue
                out[s:e] = a[s:e] @ b[g]
            return out

        for _ in range(3):
            _ = ref_loop_native()
        _sync()
        ms_ref = _do_bench_ms(ref_loop_native, warmup=10, rep=30)
        _sync()

        flops = 2.0 * float(total_M) * float(spec.K) * float(spec.N)
        gflops_cublas = flops / (ms_cublas * 1e-3) / 1e9
        gflops_ref = flops / (ms_ref * 1e-3) / 1e9
        speedup_vs_ref = ms_ref / ms_cublas

        print(f"\n[tokens={tokens} total_M={total_M} max_M={max_M} dtype={dtype}]")
        print(f"  cuBLAS grouped     : {ms_cublas:.3f} ms ({gflops_cublas:.1f} GFLOP/s)  speedup_vs_ref={speedup_vs_ref:.2f}x")
        if ms_triton is not None:
            gflops_triton = flops / (ms_triton * 1e-3) / 1e9
            print(f"  VeOmni Triton       : {ms_triton:.3f} ms ({gflops_triton:.1f} GFLOP/s)  speedup_vs_ref={ms_ref/ms_triton:.2f}x")
            print(f"  speedup (Triton/cuBLAS): {ms_cublas/ms_triton:.2f}x  ( >1 means Triton faster )")
            print(f"  speedup (cuBLAS/Triton): {ms_triton/ms_cublas:.2f}x  ( >1 means cuBLAS faster )")
        print(f"  torch per-expert    : {ms_ref:.3f} ms ({gflops_ref:.1f} GFLOP/s)")


def main():
    tokens_correct = _env_int("GPTOSS_TEST_TOKENS", 512)
    correctness_once(tokens_correct)
    bench_sweep()


if __name__ == "__main__":
    main()

