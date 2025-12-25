// mma_gemm_cta_m16n8k8.cu
//
// CTA-tiled GEMM using inline PTX MMA:
//   mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
//
// Layout:
// - A: row-major [M x K], half
// - B: column-major [K x N] stored as Bcol [N x K], half
//      (i.e. B(k,n) is at Bcol[n*K + k])
// - C: row-major [M x N], float
//
// Block tile: 64x16
// - warps_m = 4 (64/16), warps_n = 2 (16/8) => 8 warps
// - blockDim = 256 threads
//
// Build examples:
//   nvcc -O3 -arch=sm_80 mma_gemm_cta_m16n8k8.cu -o mma_gemm
//   nvcc -O3 -arch=sm_90 mma_gemm_cta_m16n8k8.cu -o mma_gemm
//
// Run:
//   ./mma_gemm

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

static inline void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error %s: (%d) %s\n", msg, (int)e, cudaGetErrorString(e));
    std::abort();
  }
}

__device__ __forceinline__ unsigned lane_id() {
  unsigned id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(id));
  return id;
}

__device__ __forceinline__ unsigned warp_id_in_block() {
  return threadIdx.x >> 5;
}

__global__ void gemm_mma_m16n8k8_row_col_f32_f16_f16_f32(
    const half* __restrict__ A_rm,   // [M*K]
    const half* __restrict__ Bcol,   // [N*K] (column-major for B)
    float* __restrict__ C_rm,        // [M*N]
    int M, int N, int K)
{
  // CTA tile sizes
  constexpr int BM = 64;
  constexpr int BN = 16;
  constexpr int BK = 8;

  // Shared tiles for one K-slice
  __shared__ half As[BM * BK]; // row-major: As[(m)*BK + k]
  __shared__ half Bs[BN * BK]; // column-major by N: Bs[(n)*BK + k]  (so B is .col-friendly)

  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  // Warp mapping: 8 warps = (warp_m in [0..3], warp_n in [0..1])
  const unsigned warp = warp_id_in_block();      // 0..7
  const int warp_m = (int)(warp >> 1);           // 0..3
  const int warp_n = (int)(warp & 1u);           // 0..1

  // Each warp computes a 16x8 tile:
  const int warp_row_base = warp_m * 16;         // 0,16,32,48 within CTA
  const int warp_col_base = warp_n * 8;          // 0 or 8 within CTA

  // Lane decomposition per PTX fragment mapping
  const unsigned lid = lane_id();                // 0..31
  const unsigned g = lid >> 2;                   // groupID 0..7
  const unsigned t = lid & 3;                    // threadID_in_group 0..3

  // Accumulators: each lane holds 4 f32 outputs for its (rows, cols)
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  // Loop over K in steps of 8
  for (int k0 = 0; k0 < K; k0 += BK) {
    // ---------------------------
    // Load As tile: [BM x BK]
    // As[m, kk] = A[block_m + m, k0 + kk]
    // ---------------------------
    for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
      int m  = idx / BK;        // 0..63
      int kk = idx % BK;        // 0..7
      int gm = block_m + m;
      int gk = k0 + kk;
      half v = __float2half(0.f);
      if (gm < M && gk < K) {
        v = A_rm[gm * K + gk];
      }
      As[m * BK + kk] = v;
    }

    // ---------------------------
    // Load Bs tile: [BN x BK] stored as Bs[n, kk] (column-major for B)
    // Bs[n, kk] = B(k0+kk, block_n+n)
    // but B is stored as Bcol[n_global*K + k_global]
    // ---------------------------
    for (int idx = threadIdx.x; idx < BN * BK; idx += blockDim.x) {
      int n  = idx / BK;        // 0..15
      int kk = idx % BK;        // 0..7
      int gn = block_n + n;
      int gk = k0 + kk;
      half v = __float2half(0.f);
      if (gn < N && gk < K) {
        v = Bcol[gn * K + gk];
      }
      Bs[n * BK + kk] = v;
    }

    __syncthreads();

    // Each warp reads its fragment from shared and does mma.
    // Build A fragment (.row): a_reg0 packs A(row=g, k=2t,2t+1),
    //                          a_reg1 packs A(row=g+8, k=2t,2t+1)
    const int kA0 = 2 * (int)t;
    const int kA1 = kA0 + 1;

    const int r0 = warp_row_base + (int)g;       // 0..15 within warp tile, plus warp_row_base
    const int r1 = r0 + 8;

    half a0 = As[r0 * BK + kA0];
    half a1 = As[r0 * BK + kA1];
    half a2 = As[r1 * BK + kA0];
    half a3 = As[r1 * BK + kA1];

    __half2 a01 = __halves2half2(a0, a1);
    __half2 a23 = __halves2half2(a2, a3);

    unsigned a_reg0 = reinterpret_cast<unsigned&>(a01);
    unsigned a_reg1 = reinterpret_cast<unsigned&>(a23);

    // Build B fragment (.col): b_reg0 packs B(k=2t,2t+1, col=groupID)
    // Our Bs is indexed by n (0..15) and kk (0..7), and B.col uses col = groupID
    // For this warp tile, local col = warp_col_base + groupID
    const int n_local = warp_col_base + (int)g;  // 0..15 within CTA tile

    half b0 = Bs[n_local * BK + kA0];
    half b1 = Bs[n_local * BK + kA1];

    __half2 b01 = __halves2half2(b0, b1);
    unsigned b_reg0 = reinterpret_cast<unsigned&>(b01);

    // MMA accumulate: use c regs as both C input and D output
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5}, "
      "{%6}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a_reg0), "r"(a_reg1),
        "r"(b_reg0)
    );

    __syncthreads();
  }

  // Store this lane’s 4 outputs to global C
  // For m16n8k8 C/D mapping (f32):
  //   c0 -> (row=groupID,   col=2t)
  //   c1 -> (row=groupID,   col=2t+1)
  //   c2 -> (row=groupID+8, col=2t)
  //   c3 -> (row=groupID+8, col=2t+1)
  const int out_row0 = block_m + warp_row_base + (int)g;
  const int out_row1 = out_row0 + 8;

  const int out_col0 = block_n + warp_col_base + 2 * (int)t;
  const int out_col1 = out_col0 + 1;

  if (out_row0 < M && out_col0 < N) C_rm[out_row0 * N + out_col0] = c0;
  if (out_row0 < M && out_col1 < N) C_rm[out_row0 * N + out_col1] = c1;
  if (out_row1 < M && out_col0 < N) C_rm[out_row1 * N + out_col0] = c2;
  if (out_row1 < M && out_col1 < N) C_rm[out_row1 * N + out_col1] = c3;
}

// CPU ref: A row-major, Bcol column-major (Bcol[n*K + k])
static void cpu_ref(
    const std::vector<half>& A_rm,
    const std::vector<half>& Bcol,
    std::vector<float>& C_rm,
    int M, int N, int K)
{
  auto A = [&](int m, int k) -> float { return __half2float(A_rm[m * K + k]); };
  auto B = [&](int k, int n) -> float { return __half2float(Bcol[n * K + k]); };

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += A(m,k) * B(k,n);
      C_rm[m * N + n] = acc;
    }
  }
}

int main() {
  // Pick sizes that exercise CTA tiling & warps; must be multiples of BK=8 for full coverage
  // (code handles remainders too, but start with clean multiples)
  const int M = 128;
  const int N = 64;
  const int K = 64;

  printf("Testing GEMM: M=%d N=%d K=%d\n", M, N, K);

  std::vector<half>  hA(M * K);
  std::vector<half>  hBcol(N * K);
  std::vector<float> hCgpu(M * N, -1.f);
  std::vector<float> hCref(M * N,  0.f);

  // “Human-friendly-ish” but not trivial:
  // A(m,k) = (m%7) + (k%5) + 1
  // B(k,n) = (k%3) - (n%4)  (small signed ints)
  // Stored as Bcol[n*K + k]
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      float v = float((m % 7) + (k % 5) + 1);     // 1..11
      hA[m * K + k] = __float2half(v);
    }
  }
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      float v = float((k % 3) - (n % 4));         // small ints in [-3..2]
      hBcol[n * K + k] = __float2half(v);
    }
  }

  cpu_ref(hA, hBcol, hCref, M, N, K);

  half* dA = nullptr;
  half* dB = nullptr;
  float* dC = nullptr;

  ck(cudaMalloc(&dA, hA.size() * sizeof(half)), "malloc A");
  ck(cudaMalloc(&dB, hBcol.size() * sizeof(half)), "malloc B");
  ck(cudaMalloc(&dC, hCgpu.size() * sizeof(float)), "malloc C");

  ck(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice), "cpy A");
  ck(cudaMemcpy(dB, hBcol.data(), hBcol.size() * sizeof(half), cudaMemcpyHostToDevice), "cpy B");

  dim3 block(256, 1, 1);
  dim3 grid((N + 16 - 1) / 16, (M + 64 - 1) / 64, 1);

  gemm_mma_m16n8k8_row_col_f32_f16_f16_f32<<<grid, block>>>(dA, dB, dC, M, N, K);
  ck(cudaGetLastError(), "kernel launch");
  ck(cudaDeviceSynchronize(), "sync");

  ck(cudaMemcpy(hCgpu.data(), dC, hCgpu.size() * sizeof(float), cudaMemcpyDeviceToHost), "cpy C");

  // Compare
  float max_abs = 0.f;
  int max_i = -1;
  for (int i = 0; i < M * N; ++i) {
    float err = std::fabs(hCgpu[i] - hCref[i]);
    if (err > max_abs) { max_abs = err; max_i = i; }
  }

  printf("Max abs error: %.6g at i=%d (m=%d n=%d). GPU=%.6g REF=%.6g\n",
         max_abs, max_i, max_i / N, max_i % N,
         (max_i >= 0 ? hCgpu[max_i] : 0.f),
         (max_i >= 0 ? hCref[max_i] : 0.f));

  // Print a tiny window to eyeball
  printf("\nC[0:4, 0:8] GPU:\n");
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 8; ++n) printf("%8.1f ", hCgpu[m * N + n]);
    printf("\n");
  }
  printf("\nC[0:4, 0:8] REF:\n");
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 8; ++n) printf("%8.1f ", hCref[m * N + n]);
    printf("\n");
  }

  // Tolerance: since we’re doing exact FP16->FP32 products and deterministic sum order,
  // expect tiny differences possible but should be near 0 for these small magnitudes.
  if (max_abs > 1e-2f) {
    fprintf(stderr, "FAILED\n");
    return 1;
  }
  printf("\nPASSED\n");

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}

