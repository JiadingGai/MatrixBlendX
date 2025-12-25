// mma_m16n8k8_row_col_check.cu
//
// One warp computes: D(16x8) = A(16x8) * B(8x8) + C(16x8)
// using inline PTX:
//   mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
//
// This version adds a CPU reference check and uses "human-friendly" A/B:
//   A(m,k) = m + 1      (same across k)   -> rows are 1..16
//   B(k,n) = n + 1      (same across k)   -> cols are 1..8
// Then for any output:
//   D(m,n) = sum_{k=0..7} (m+1)*(n+1) = 8*(m+1)*(n+1)
//
// Storage assumptions (matching the PTX opcode row/col interpretation):
// - A is stored row-major: A_rm[m*8 + k]
// - B is stored column-major: B_cm[n*8 + k]  (so B(k,n) is at B_cm[n*8 + k])
// - D is stored row-major float: D_rm[m*8 + n]
//
// Build:
//   nvcc -O3 -arch=sm_80 mma_m16n8k8_row_col_check.cu -o mma_check
// Run:
//   ./mma_check
//
// Notes:
// - Requires Tensor Core capable GPU supporting this MMA shape (sm_70+ generally).
// - This is a correctness reference, not performance-oriented.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>

static inline void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
    std::abort();
  }
}

__device__ __forceinline__ unsigned lane_id() {
  unsigned id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(id));
  return id;
}

__global__ void mma_m16n8k8_row_col_f32_f16_f16_f32(
    const half* __restrict__ A_rm_16x8,   // row-major [16*8]
    const half* __restrict__ B_cm_8x8,    // column-major [8*8] as B_cm[n*8 + k]
    float* __restrict__ D_rm_16x8)        // row-major [16*8]
{
  // For demo simplicity, stage in shared.
  __shared__ half As[16 * 8];
  __shared__ half Bs[8 * 8];

  const unsigned tid = threadIdx.x;
  for (unsigned i = tid; i < 16u * 8u; i += blockDim.x) As[i] = A_rm_16x8[i];
  for (unsigned i = tid; i < 8u * 8u;  i += blockDim.x) Bs[i] = B_cm_8x8[i];
  __syncthreads();

  const unsigned lid = lane_id();   // 0..31
  const unsigned g   = lid >> 2;    // groupID 0..7
  const unsigned t   = lid & 3;     // threadID_in_group 0..3

  // A fragment for .row (f16): each lane packs:
  //  row=g,     k=2t,2t+1 into a_reg0
  //  row=g+8,   k=2t,2t+1 into a_reg1
  const int k0 = int(2 * t);
  const int k1 = int(2 * t + 1);
  const int r0 = int(g);
  const int r1 = int(g + 8);

  half a0 = As[r0 * 8 + k0];
  half a1 = As[r0 * 8 + k1];
  half a2 = As[r1 * 8 + k0];
  half a3 = As[r1 * 8 + k1];

  __half2 a01 = __halves2half2(a0, a1);
  __half2 a23 = __halves2half2(a2, a3);

  unsigned a_reg0 = reinterpret_cast<unsigned&>(a01);
  unsigned a_reg1 = reinterpret_cast<unsigned&>(a23);

  // B fragment for .col (f16): each lane packs:
  //  col=g (n), k=2t,2t+1 into b_reg0
  // B stored column-major: B_cm[n*8 + k]
  const int n = int(g);
  half b0 = Bs[n * 8 + k0];
  half b1 = Bs[n * 8 + k1];
  __half2 b01 = __halves2half2(b0, b1);
  unsigned b_reg0 = reinterpret_cast<unsigned&>(b01);

  // C accumulators (f32) = 0
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  float d0, d1, d2, d3;

  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5}, "
      "{%6}, "
      "{%7, %8, %9, %10};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a_reg0), "r"(a_reg1),
        "r"(b_reg0),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3)
  );

  // Store lane outputs to row-major D:
  // d0 -> (row=g,   col=2t)
  // d1 -> (row=g,   col=2t+1)
  // d2 -> (row=g+8, col=2t)
  // d3 -> (row=g+8, col=2t+1)
  D_rm_16x8[r0 * 8 + k0] = d0;
  D_rm_16x8[r0 * 8 + k1] = d1;
  D_rm_16x8[r1 * 8 + k0] = d2;
  D_rm_16x8[r1 * 8 + k1] = d3;
}

static void cpu_ref_16x8_8x8(
    const std::vector<half>& A_rm,
    const std::vector<half>& B_cm,
    std::vector<float>& D_rm_out)
{
  auto A = [&](int m, int k) -> float {
    return __half2float(A_rm[m * 8 + k]);
  };
  auto B = [&](int k, int n) -> float {
    // column-major: B_cm[n*8 + k]
    return __half2float(B_cm[n * 8 + k]);
  };
  for (int m = 0; m < 16; ++m) {
    for (int n = 0; n < 8; ++n) {
      float acc = 0.f;
      for (int k = 0; k < 8; ++k) acc += A(m, k) * B(k, n);
      D_rm_out[m * 8 + n] = acc;
    }
  }
}

int main() {
  // Human-friendly inputs:
  // A(m,k) = m+1  (rows: 1..16)
  // B(k,n) = n+1  (cols: 1..8), stored column-major
  std::vector<half>  hA(16 * 8);
  std::vector<half>  hBcm(8 * 8);
  std::vector<float> hDgpu(16 * 8, -123.f);
  std::vector<float> hDref(16 * 8, 0.f);

  for (int m = 0; m < 16; ++m) {
    for (int k = 0; k < 8; ++k) {
      hA[m * 8 + k] = __float2half(float(m + 1));
    }
  }
  for (int n = 0; n < 8; ++n) {
    for (int k = 0; k < 8; ++k) {
      // B(k,n) = n+1, stored column-major
      hBcm[n * 8 + k] = __float2half(float(n + 1));
    }
  }

  cpu_ref_16x8_8x8(hA, hBcm, hDref);

  half*  dA = nullptr;
  half*  dB = nullptr;
  float* dD = nullptr;
  ck(cudaMalloc(&dA, hA.size() * sizeof(half)), "malloc A");
  ck(cudaMalloc(&dB, hBcm.size() * sizeof(half)), "malloc B");
  ck(cudaMalloc(&dD, hDgpu.size() * sizeof(float)), "malloc D");

  ck(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice), "cpy A");
  ck(cudaMemcpy(dB, hBcm.data(), hBcm.size() * sizeof(half), cudaMemcpyHostToDevice), "cpy B");

  mma_m16n8k8_row_col_f32_f16_f16_f32<<<1, 32>>>(dA, dB, dD);
  ck(cudaGetLastError(), "launch");
  ck(cudaDeviceSynchronize(), "sync");

  ck(cudaMemcpy(hDgpu.data(), dD, hDgpu.size() * sizeof(float), cudaMemcpyDeviceToHost), "cpy D");

  // Compare
  float max_abs_err = 0.f;
  int max_i = -1;
  for (int i = 0; i < 16 * 8; ++i) {
    float err = std::fabs(hDgpu[i] - hDref[i]);
    if (err > max_abs_err) { max_abs_err = err; max_i = i; }
  }

  // Print a small slice for easy manual checking:
  // Expected: D(m,n) = 8*(m+1)*(n+1)
  printf("Expected formula: D(m,n) = 8*(m+1)*(n+1)\n\n");
  printf("Top-left 4x4 of D (GPU):\n");
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 4; ++n) printf("%8.1f ", hDgpu[m * 8 + n]);
    printf("\n");
  }
  printf("\nTop-left 4x4 of D (REF):\n");
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 4; ++n) printf("%8.1f ", hDref[m * 8 + n]);
    printf("\n");
  }

  // Show a couple of specific entries to sanity check quickly
  auto show = [&](int m, int n) {
    float expv = hDref[m * 8 + n];
    float got  = hDgpu[m * 8 + n];
    printf("D(%2d,%2d): GPU=%8.1f  REF=%8.1f\n", m, n, got, expv);
  };
  printf("\nSpot checks:\n");
  show(0, 0);   // 8*1*1 = 8
  show(0, 7);   // 8*1*8 = 64
  show(15, 0);  // 8*16*1 = 128
  show(15, 7);  // 8*16*8 = 1024
  show(3, 2);   // 8*4*3 = 96

  printf("\nMax abs err: %.6g (at linear idx %d => m=%d n=%d)\n",
         max_abs_err, max_i, max_i / 8, max_i % 8);

  // Tight tolerance: exact integer-ish sums should be exact in f32 here.
  if (max_abs_err != 0.f) {
    fprintf(stderr, "FAILED: nonzero error.\n");
    return 1;
  }
  printf("PASSED.\n");

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dD);
  return 0;
}

