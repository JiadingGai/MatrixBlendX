import os
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline

# If not set, compilation may include many archs and be slow.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
assert torch.cuda.is_available(), "CUDA required"

# ---------------------------------------------------------------------
# Triton reduce_v1 (same as your working version)
# ---------------------------------------------------------------------
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 1


@triton.jit
def _write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    rt,
    token_mask,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    zeros = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * rt[:, None] + stride_cn * offs_n[None, :]
    c_mask = token_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, zeros, mask=c_mask)


@triton.jit
def reduce_v1_triton(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    rt = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = rt < num_valid_tokens

    e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if e == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            rt,
            token_mask,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        return

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    tok = rt // top_k
    a_ptrs = a_ptr + tok[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + e * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for kb in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_base = kb * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & ((k_base + offs_k[None, :]) < K), other=0.0)
        b = tl.load(b_ptrs, mask=((k_base + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        w = tl.load(topk_weights_ptr + rt, mask=token_mask, other=0.0).to(tl.float32)
        acc *= w[:, None]

    out = acc.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * rt[:, None] + stride_cn * offs_n[None, :]
    c_mask = token_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


@torch.no_grad()
def triton_forward(A, B, sorted_ids, expert_ids, topk_w, top_k, mul_routed_weight=False, num_warps=8):
    M, K = A.shape
    E, N, K2 = B.shape
    assert K == K2
    EM = M * top_k

    C = torch.empty((M, top_k, N), device=A.device, dtype=torch.float16)
    C2d = C.view(EM, N)

    num_tokens_post_padded = torch.tensor([sorted_ids.numel()], device=A.device, dtype=torch.int32)
    grid = (triton.cdiv(EM, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    reduce_v1_triton[grid](
        A, B, C2d,
        topk_w,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        N, K, EM, EM,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C2d.stride(0), C2d.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        num_warps=num_warps,
    )
    return C


# ---------------------------------------------------------------------
# CUDA extension: WMMA warp-tiled, plus FIX A) stage B once per (k0) in shared
# ---------------------------------------------------------------------
cuda_src = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

using half = __half;
using namespace nvcuda;

#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  exit(1); } } while(0)

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

template <bool MUL_ROUTED_WEIGHT>
__global__ void reduce_v1_wmma_stageB_once(
    const half* __restrict__ A,          // [M, K]
    const half* __restrict__ B,          // [E, N, K] row-major in (N,K)
    half* __restrict__ C,                // [EM, N]
    const half* __restrict__ topk_w,     // [EM]
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded_ptr,
    int N, int K, int EM, int num_valid_tokens,
    int stride_am, int stride_ak,
    int stride_be, int stride_bn, int stride_bk,
    int stride_cm, int stride_cn,
    int top_k
) {
  int num_pid_m = (EM + BM - 1) / BM;
  int num_pid_n = (N  + BN - 1) / BN;
  int pid = (int)blockIdx.x;
  int pid_m = pid / num_pid_n;
  int pid_n = pid - pid_m * num_pid_n;

  int num_tokens_post_padded = *num_tokens_post_padded_ptr;
  if (pid_m * BM >= num_tokens_post_padded) return;

  int e = (int)expert_ids[pid_m];

  // Shared buffers
  __shared__ half As[BM * BK];     // 64x16
  __shared__ half Bs[BN * BK];     // 64x16  <-- NEW: stage B slab once per k0
  __shared__ int rt_row[BM];
  __shared__ int tok_row[BM];
  __shared__ uint8_t row_valid[BM];

  // 16 warps = 512 threads: one warp per (mi,ni) 16x16 tile
  int warp_id = (int)threadIdx.x >> 5;
  int lane    = (int)threadIdx.x & 31;

  // Load routing rows
  for (int i = (int)threadIdx.x; i < BM; i += (int)blockDim.x) {
    int rt = (int)sorted_token_ids[pid_m * BM + i];
    rt_row[i] = rt;
    bool valid = (rt < num_valid_tokens);
    row_valid[i] = (uint8_t)valid;
    tok_row[i] = valid ? (rt / top_k) : 0;
  }
  __syncthreads();

  if (e == -1) {
    for (int idx = (int)threadIdx.x; idx < BM * BN; idx += (int)blockDim.x) {
      int mi = idx / BN;
      int ni = idx - mi * BN;
      int rt = rt_row[mi];
      if (rt < num_valid_tokens) {
        int n = pid_n * BN + ni;
        if (n < N) C[rt * stride_cm + n * stride_cn] = __float2half(0.0f);
      }
    }
    return;
  }

  bool active = (warp_id < 16);
  int mi = warp_id / 4;   // 0..3
  int ni = warp_id % 4;   // 0..3
  int n0 = pid_n * BN + ni * 16;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  if (active) wmma::fill_fragment(acc, 0.0f);

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Stage A (64x16) with gather
    for (int idx = (int)threadIdx.x; idx < BM * BK; idx += (int)blockDim.x) {
      int r  = idx / BK;
      int kk = idx - r * BK;
      half v = __float2half(0.0f);
      if (row_valid[r]) {
        int tok = tok_row[r];
        int k = k0 + kk;
        if (k < K) v = A[tok * stride_am + k * stride_ak];
      }
      As[idx] = v;
    }

    // Stage B slab (64x16) ONCE per k0 for this CTA
    // Bs is row-major [BN, BK] with ld=BK.
    for (int idx = (int)threadIdx.x; idx < BN * BK; idx += (int)blockDim.x) {
      int nn = idx / BK;       // 0..63 within BN
      int kk = idx - nn * BK;  // 0..15
      int n = pid_n * BN + nn; // global n
      int k = k0 + kk;
      half v = __float2half(0.0f);
      if (n < N && k < K) {
        v = B[e * stride_be + n * stride_bn + k * stride_bk];
      }
      Bs[idx] = v;
    }

    __syncthreads();

    if (active && (n0 + 15 < N)) {
      // A: shared row-major, ld=16
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      const half* a_tile_ptr = &As[(mi * 16) * BK];
      wmma::load_matrix_sync(a_frag, a_tile_ptr, BK);

      // B: load from shared Bs.
      // Trick: Bs is row-major [BN, BK] with ld=BK,
      // which is equivalent to B^T col-major [BK, BN] with ld=BK.
      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
      const half* b_tile_ptr = &Bs[(ni * 16) * BK];  // start at n-block, k=0
      wmma::load_matrix_sync(b_frag, b_tile_ptr, BK);

      wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    __syncthreads();
  }

  // Store warp tile to shared then scatter to C
  __shared__ float SmemAcc[16 * 16 * 16]; // 16 tiles * 256 floats

  if (active && (n0 + 15 < N)) {
    float* tile = &SmemAcc[warp_id * 256];
    wmma::store_matrix_sync(tile, acc, 16, wmma::mem_row_major);
  }
  __syncthreads();

  if (active && (n0 + 15 < N)) {
    float* tile = &SmemAcc[warp_id * 256];
    for (int idx = lane; idx < 256; idx += 32) {
      int r = idx / 16;
      int c = idx - r * 16;
      int row = mi * 16 + r;
      int col = ni * 16 + c;
      int n = pid_n * BN + col;

      int rt = rt_row[row];
      if (rt < num_valid_tokens && n < N) {
        float val = tile[idx];
        if constexpr (MUL_ROUTED_WEIGHT) val *= __half2float(topk_w[rt]);
        C[rt * stride_cm + n * stride_cn] = __float2half_rn(val);
      }
    }
  }
}

torch::Tensor reduce_v1_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor sorted_ids,
    torch::Tensor expert_ids,
    torch::Tensor topk_w,
    int64_t top_k,
    bool mul_routed_weight
) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A/B must be CUDA");
  TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16, "A/B must be fp16");
  TORCH_CHECK(sorted_ids.dtype() == torch::kInt32 && expert_ids.dtype() == torch::kInt32, "ids must be int32");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A/B must be contiguous");
  TORCH_CHECK(sorted_ids.is_contiguous() && expert_ids.is_contiguous(), "ids must be contiguous");
  TORCH_CHECK(topk_w.is_contiguous(), "topk_w must be contiguous");

  int M = (int)A.size(0);
  int K = (int)A.size(1);
  int E = (int)B.size(0);
  int N = (int)B.size(1);
  TORCH_CHECK((int)B.size(2) == K, "B.size(2) must equal K");

  int EM = M * (int)top_k;
  auto C = torch::empty({EM, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat16));

  auto ntpp = torch::empty({1}, torch::TensorOptions().device(A.device()).dtype(torch::kInt32));
  ntpp.fill_((int)sorted_ids.numel());

  int num_pid_m = (EM + BM - 1) / BM;
  int num_pid_n = (N + BN - 1) / BN;
  int grid = num_pid_m * num_pid_n;

  dim3 block(512); // 16 warps
  dim3 gridDim(grid);

  int stride_am = (int)A.stride(0), stride_ak = (int)A.stride(1);
  int stride_be = (int)B.stride(0), stride_bn = (int)B.stride(1), stride_bk = (int)B.stride(2);
  int stride_cm = (int)C.stride(0), stride_cn = (int)C.stride(1);

  const half* A_ptr = (const half*)A.data_ptr<at::Half>();
  const half* B_ptr = (const half*)B.data_ptr<at::Half>();
  half* C_ptr = (half*)C.data_ptr<at::Half>();
  const half* W_ptr = (const half*)topk_w.data_ptr<at::Half>();
  const int32_t* sorted_ptr = (const int32_t*)sorted_ids.data_ptr<int32_t>();
  const int32_t* expert_ptr = (const int32_t*)expert_ids.data_ptr<int32_t>();
  const int32_t* ntpp_ptr = (const int32_t*)ntpp.data_ptr<int32_t>();

  if (mul_routed_weight) {
    reduce_v1_wmma_stageB_once<true><<<gridDim, block>>>(
      A_ptr, B_ptr, C_ptr, W_ptr, sorted_ptr, expert_ptr, ntpp_ptr,
      N, K, EM, EM,
      stride_am, stride_ak,
      stride_be, stride_bn, stride_bk,
      stride_cm, stride_cn,
      (int)top_k
    );
  } else {
    reduce_v1_wmma_stageB_once<false><<<gridDim, block>>>(
      A_ptr, B_ptr, C_ptr, W_ptr, sorted_ptr, expert_ptr, ntpp_ptr,
      N, K, EM, EM,
      stride_am, stride_ak,
      stride_be, stride_bn, stride_bk,
      stride_cm, stride_cn,
      (int)top_k
    );
  }
  CHECK_CUDA(cudaGetLastError());
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce_v1_cuda", &reduce_v1_cuda, "reduce_v1 CUDA (WMMA, stage B once)");
}
'''

ext = load_inline(
    name="reduce_v1_ext_stageB_once",
    cpp_sources="",
    cuda_sources=cuda_src,
    functions=None,
    with_cuda=True,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    verbose=False,
)


@torch.no_grad()
def cuda_forward(A, B, sorted_ids, expert_ids, topk_w, top_k, mul_routed_weight=False):
    return ext.reduce_v1_cuda(A, B, sorted_ids, expert_ids, topk_w, int(top_k), bool(mul_routed_weight))


# ---------------------------------------------------------------------
# Bench helpers
# ---------------------------------------------------------------------
@torch.no_grad()
def bench_ms(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@torch.no_grad()
def max_abs_err(x, y):
    return (x - y).abs().max().item()


def main():
    device = "cuda"
    torch.manual_seed(0)

    # Your requested shape
    M, E, N, K, top_k = 48, 128, 2880, 2880, 4
    EM = M * top_k
    print(f"Shape: M={M} E={E} N={N} K={K} top_k={top_k} (EM={EM})")

    free, total = torch.cuda.mem_get_info()
    print(f"CUDA mem: free={free/1e9:.2f} GB total={total/1e9:.2f} GB")

    A = torch.randn(M, K, device=device, dtype=torch.float16)
    B = torch.randn(E, N, K, device=device, dtype=torch.float16)

    # Toy routing (same as before): touches only num_pid_m experts
    sorted_ids = torch.arange(EM, device=device, dtype=torch.int32).contiguous()
    num_pid_m = triton.cdiv(EM, BLOCK_SIZE_M)  # 3 for EM=192
    expert_ids = (torch.arange(num_pid_m, device=device, dtype=torch.int32) % E).contiguous()
    topk_w = torch.ones(EM, device=device, dtype=torch.float16).contiguous()

    # Correctness
    C_tri = triton_forward(A, B, sorted_ids, expert_ids, topk_w, top_k, False, num_warps=8)
    C_cu2d = cuda_forward(A, B, sorted_ids, expert_ids, topk_w, top_k, False)
    C_cu = C_cu2d.view(M, top_k, N)
    err = max_abs_err(C_tri, C_cu)
    print(f"max |triton - cuda| = {err:.6e}")

    # Speed
    tri_ms = bench_ms(lambda: triton_forward(A, B, sorted_ids, expert_ids, topk_w, top_k, False, num_warps=8))
    cu_ms = bench_ms(lambda: cuda_forward(A, B, sorted_ids, expert_ids, topk_w, top_k, False))
    print(f"Triton: {tri_ms:.6f} ms")
    print(f"CUDA  : {cu_ms:.6f} ms")
    print(f"Speedup (Triton/CUDA): {cu_ms / tri_ms:.3f}x  (>1 means Triton faster)")

    # FLOPs + TFLOPS
    flops = 2.0 * EM * N * K
    print(f"TFLOPS: Triton {flops/(tri_ms*1e-3)/1e12:.3f}, CUDA {flops/(cu_ms*1e-3)/1e12:.3f}")

    # Bytes model (unique bytes). With B staged once, CUDA should be closer to this.
    bytes_B = (num_pid_m * N * K) * 2
    bytes_A = (EM * K) * 2
    bytes_C = (EM * N) * 2
    bytes_total = bytes_A + bytes_B + bytes_C

    tri_bw = bytes_total / (tri_ms * 1e-3) / 1e9
    cu_bw = bytes_total / (cu_ms * 1e-3) / 1e9
    print(f"Assumed unique bytes/call: {bytes_total/1e6:.3f} MB (A {bytes_A/1e6:.3f}, B {bytes_B/1e6:.3f}, C {bytes_C/1e6:.3f})")
    print(f"Effective BW (unique): Triton {tri_bw:.1f} GB/s, CUDA {cu_bw:.1f} GB/s")

if __name__ == "__main__":
    main()
