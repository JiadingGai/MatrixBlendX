// mma_gemm_m512n1024k256.cu
//
// CTA-tiled GEMM using inline PTX MMA:
//   mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
//
// Computes: C[MxN] = A[MxK] * B[KxN], accumulate FP32
//
// Layout:
// - A: row-major [M x K], half   (A[m*K + k])
// - B: column-major storage by N: Bcol[n*K + k] holds B(k,n), half
// - C: row-major [M x N], float
//
// Block tile: 64x16 output per CTA
// - warps_m = 4 (64/16), warps_n = 2 (16/8) => 8 warps
// - blockDim = 256 threads
// K step BK=8
//
// Build examples:
//   nvcc -O3 -arch=sm_90  mma_gemm_m512n1024k256.cu -o mma_gemm
//   nvcc -O3 -arch=sm_120 mma_gemm_m512n1024k256.cu -o mma_gemm
//
// Run:
//   ./mma_gemm

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

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
  return (unsigned)(threadIdx.x >> 5);
}

__global__ void gemm_mma_m16n8k8_row_col_f32_f16_f16_f32(
    const half* __restrict__ A_rm,   // [M*K]
    const half* __restrict__ Bcol,   // [N*K] where B(k,n)=Bcol[n*K + k]
    float* __restrict__ C_rm,        // [M*N]
    int M, int N, int K)
{
  constexpr int BM = 64;
  constexpr int BN = 16;
  constexpr int BK = 8;

  __shared__ half As[BM * BK]; // As[m*BK + kk], row-major
  __shared__ half Bs[BN * BK]; // Bs[n*BK + kk], col-friendly (n-major)

  const int block_m = (int)blockIdx.y * BM;
  const int block_n = (int)blockIdx.x * BN;

  const unsigned warp = warp_id_in_block(); // 0..7 (since blockDim.x=256)
  const int warp_m = (int)(warp >> 1);      // 0..3
  const int warp_n = (int)(warp & 1u);      // 0..1

  const int warp_row_base = warp_m * 16;    // 0,16,32,48
  const int warp_col_base = warp_n * 8;     // 0 or 8

  const unsigned lid = lane_id();           // 0..31
  const unsigned g = lid >> 2;              // groupID 0..7
  const unsigned t = lid & 3;               // threadID_in_group 0..3

  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Load As: [BM x BK] from A_rm
    for (int idx = (int)threadIdx.x; idx < BM * BK; idx += (int)blockDim.x) {
      const int m  = idx / BK;   // 0..63
      const int kk = idx % BK;   // 0..7
      const int gm = block_m + m;
      const int gk = k0 + kk;
      half v = __float2half(0.f);
      if (gm < M && gk < K) v = A_rm[gm * K + gk];
      As[m * BK + kk] = v;
    }

    // Load Bs: [BN x BK] from Bcol (column-major by N)
    for (int idx = (int)threadIdx.x; idx < BN * BK; idx += (int)blockDim.x) {
      const int n  = idx / BK;   // 0..15
      const int kk = idx % BK;   // 0..7
      const int gn = block_n + n;
      const int gk = k0 + kk;
      half v = __float2half(0.f);
      if (gn < N && gk < K) v = Bcol[gn * K + gk];  // B(k,n)
      Bs[n * BK + kk] = v;
    }

    __syncthreads();

    // Build A fragment for .row
    const int kA0 = 2 * (int)t;
    const int kA1 = kA0 + 1;

    const int r0 = warp_row_base + (int)g;  // 0..15 within CTA's BM
    const int r1 = r0 + 8;

    half a0 = As[r0 * BK + kA0];
    half a1 = As[r0 * BK + kA1];
    half a2 = As[r1 * BK + kA0];
    half a3 = As[r1 * BK + kA1];

    __half2 a01 = __halves2half2(a0, a1);
    __half2 a23 = __halves2half2(a2, a3);

    unsigned a_reg0 = reinterpret_cast<unsigned&>(a01);
    unsigned a_reg1 = reinterpret_cast<unsigned&>(a23);

    // Build B fragment for .col
    const int n_local = warp_col_base + (int)g; // 0..15 within CTA's BN
    half b0 = Bs[n_local * BK + kA0];
    half b1 = Bs[n_local * BK + kA1];

    __half2 b01 = __halves2half2(b0, b1);
    unsigned b_reg0 = reinterpret_cast<unsigned&>(b01);

    // Accumulate: D = A*B + C (use c regs as both input and output)
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

  // Store this lane's 4 outputs (m16n8 mapping for f32 accumulators)
  const int out_row0 = block_m + warp_row_base + (int)g;
  const int out_row1 = out_row0 + 8;

  const int out_col0 = block_n + warp_col_base + 2 * (int)t;
  const int out_col1 = out_col0 + 1;

  if (out_row0 < M && out_col0 < N) C_rm[out_row0 * N + out_col0] = c0;
  if (out_row0 < M && out_col1 < N) C_rm[out_row0 * N + out_col1] = c1;
  if (out_row1 < M && out_col0 < N) C_rm[out_row1 * N + out_col0] = c2;
  if (out_row1 < M && out_col1 < N) C_rm[out_row1 * N + out_col1] = c3;
}

// CPU reference: A row-major, Bcol column-major by N
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
  const int M = 512;
  const int N = 1024;
  const int K = 256;

  printf("Testing GEMM: M=%d N=%d K=%d\n", M, N, K);

  std::vector<half>  hA(M * K);
  std::vector<half>  hBcol(N * K);
  std::vector<float> hCgpu(M * N, -1.f);
  std::vector<float> hCref(M * N,  0.f);

  // Friendly-ish values (avoid huge magnitudes):
  // A(m,k) = (m%13) - (k%7) in [-6..12]
  // B(k,n) = (k%5) - (n%9) in [-8..4]
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      float v = float((m % 13) - (k % 7));
      hA[m * K + k] = __float2half(v);
    }
  }
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      float v = float((k % 5) - (n % 9));
      hBcol[n * K + k] = __float2half(v);  // B(k,n)
    }
  }

  // CPU reference (this is ~512*1024*256 ~ 134M mul-adds; can take a bit)
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

  float max_abs = 0.f;
  int max_i = 0;
  for (int i = 0; i < M * N; ++i) {
    float err = std::fabs(hCgpu[i] - hCref[i]);
    if (err > max_abs) { max_abs = err; max_i = i; }
  }

  const int max_m = max_i / N;
  const int max_n = max_i % N;

  printf("Max abs error: %.6g at (m=%d,n=%d). GPU=%.6g REF=%.6g\n",
         max_abs, max_m, max_n, hCgpu[max_i], hCref[max_i]);

  // Print a small window
  printf("\nC[0:4, 0:8] GPU:\n");
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 8; ++n) printf("%10.1f ", hCgpu[m * N + n]);
    printf("\n");
  }
  printf("\nC[0:4, 0:8] REF:\n");
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 8; ++n) printf("%10.1f ", hCref[m * N + n]);
    printf("\n");
  }

  // Tolerance: with small magnitudes, should be exact or extremely close.
  if (max_abs > 1e-2f) {
    fprintf(stderr, "\nFAILED\n");
    return 1;
  }
  printf("\nPASSED\n");

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}

