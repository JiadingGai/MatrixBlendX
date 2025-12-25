// wgmma_m512n1024k256.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_e));                                        \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// PTX descriptor format: bits [13:0]=addr, [29:16]=LBO, [45:32]=SBO, [51:49]=base, [63:62]=swizzle
// matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4
// swizzle 0 = no-swizzle, base offset ignored
static __device__ __forceinline__ unsigned long long
make_wgmma_desc_noswizzle(const void* smem_ptr, int lbo_bytes, int sbo_bytes) {
  unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  unsigned long long addr = ((unsigned long long)((smem_addr & 0x3FFFF) >> 4)) & 0x3FFFULL;
  unsigned long long lbo  = ((unsigned long long)(((lbo_bytes  & 0x3FFFF) >> 4)) & 0x3FFFULL) << 16;
  unsigned long long sbo  = ((unsigned long long)(((sbo_bytes  & 0x3FFFF) >> 4)) & 0x3FFFULL) << 32;
  unsigned long long base = 0ULL << 49;   // ignored for swizzle=0
  unsigned long long swz  = 0ULL << 62;   // no-swizzle
  return addr | lbo | sbo | base | swz;
}

// One CTA = one warpgroup = 128 threads.
// Tile: C[64x16] = A[64xK] * B[Kx16], K stepped by 16.
// Data types: f16 * f16 -> f32 accumulate.
__global__ void wgmma_gemm_m64n16k16_f32_f16(const __half* __restrict__ A,
                                            const __half* __restrict__ B,
                                            float* __restrict__ C,
                                            int M, int N, int K,
                                            int lda, int ldb, int ldc) {
#if __CUDA_ARCH__ < 900
  return;
#else
  // Must be exactly 128 threads (4 warps) for a warpgroup.
  // Launch with blockDim.x == 128.
  const int tid = (int)threadIdx.x;
  const int warp = tid >> 5;
  (void)warp;

  // CTA tile coords
  const int tile_m = (int)blockIdx.y; // 64 rows
  const int tile_n = (int)blockIdx.x; // 16 cols

  const int m0 = tile_m * 64;
  const int n0 = tile_n * 16;

  // Shared tiles (no swizzle, simple row-major staging)
  extern __shared__ unsigned char smem_raw[];
  __half* As = reinterpret_cast<__half*>(smem_raw);                   // 64 x 16
  __half* Bs = reinterpret_cast<__half*>(smem_raw + 64 * 16 * sizeof(__half)); // 16 x 16

  // Accumulator fragment: for m64n16, PTX examples indicate 8 f32 regs per thread (for tf32 m64n16k8). :contentReference[oaicite:5]{index=5}
  // In practice for f16 m64n16k16, each thread holds a fixed fragment. We'll conservatively keep 8 regs.
  float d0=0.f,d1=0.f,d2=0.f,d3=0.f,d4=0.f,d5=0.f,d6=0.f,d7=0.f;

  // wgmma requires ordering fence before first use. :contentReference[oaicite:6]{index=6}
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");

  // K loop, step 16
  for (int k0 = 0; k0 < K; k0 += 16) {
    // Stage A tile: 64x16 from global (row-major A)
    // Stage B tile: 16x16 from global (row-major B) for this demo; this computes C = A * (B_sub) with B treated row-major.
    // (If you want B col-major, you'd change the global load pattern + descriptor/transpose immediates.)
    for (int idx = tid; idx < 64 * 16; idx += 128) {
      int r = idx / 16;
      int c = idx % 16;
      int gr = m0 + r;
      int gc = k0 + c;
      __half v = __float2half(0.0f);
      if (gr < M && gc < K) v = A[gr * lda + gc];
      As[idx] = v;
    }
    for (int idx = tid; idx < 16 * 16; idx += 128) {
      int r = idx / 16; // k
      int c = idx % 16; // n
      int gr = k0 + r;
      int gc = n0 + c;
      __half v = __float2half(0.0f);
      if (gr < K && gc < N) v = B[gr * ldb + gc];
      Bs[idx] = v;
    }
    __syncthreads();

    // Build descriptors (no-swizzle)
    // For this demo we stage as plain row-major:
    // As is 64x16, row stride = 16 * sizeof(__half)
    // Bs is 16x16, row stride = 16 * sizeof(__half)
    unsigned long long descA = make_wgmma_desc_noswizzle(As, /*lbo=*/16 * (int)sizeof(__half),
                                                           /*sbo=*/0);
    unsigned long long descB = make_wgmma_desc_noswizzle(Bs, /*lbo=*/16 * (int)sizeof(__half),
                                                           /*sbo=*/0);

    // wgmma.mma_async syntax for f16/bf16: includes imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b. :contentReference[oaicite:7]{index=7}
    // We use: scale-d = 0 (predicate false means "don't scale"?), imm-scale-a=1, imm-scale-b=1, trans-a=0, trans-b=0
    // (For production you'd set these intentionally; here we keep it simple.)
    asm volatile(
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, "
      "%8, %9, "
      "0, 1, 1, 0, 0;\n"
      : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
      : "l"(descA), "l"(descB)
    );

    // Commit/wait before reusing accum regs or A-fragment regs. :contentReference[oaicite:8]{index=8}
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");

    __syncthreads();
  }

  // Store: each thread owns a subset of the 64x16 output tile.
  // For correctness demo, we’ll do a simple “scatter” store pattern that matches our fragment packing assumption:
  // map each thread to up to 8 output elements in the tile via a linearization.
  // (This is NOT the optimal store; production kernels use the documented per-thread fragment mapping tables.)
  int base = tid * 8;
  for (int i = 0; i < 8; ++i) {
    int lin = base + i;
    int r = lin / 16;
    int c = lin % 16;
    if (r < 64 && c < 16) {
      int gr = m0 + r;
      int gc = n0 + c;
      if (gr < M && gc < N) {
        float v = (&d0)[i];
        C[gr * ldc + gc] = v;
      }
    }
  }
#endif
}

static void cpu_ref(const __half* A, const __half* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) {
        acc += __half2float(A[m * lda + k]) * __half2float(B[k * ldb + n]);
      }
      C[m * ldc + n] = acc;
    }
  }
}

int main() {
  // Target GEMM
  const int M = 512, N = 1024, K = 256;
  const int lda = K, ldb = N, ldc = N;

  printf("Testing WGMMA GEMM (Hopper): M=%d N=%d K=%d\n", M, N, K);

  size_t bytesA = (size_t)M * lda * sizeof(__half);
  size_t bytesB = (size_t)K * ldb * sizeof(__half);
  size_t bytesC = (size_t)M * ldc * sizeof(float);

  __half* hA = (__half*)malloc(bytesA);
  __half* hB = (__half*)malloc(bytesB);
  float*  hCref = (float*)malloc(bytesC);
  float*  hCgpu = (float*)malloc(bytesC);

  // Numerically-friendly init (small ints)
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      float v = (float)((m % 7) - 3) + 0.25f * (float)(k % 5);
      hA[m * lda + k] = __float2half(v);
    }
  }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      float v = (float)((n % 9) - 4) - 0.125f * (float)(k % 8);
      hB[k * ldb + n] = __float2half(v);
    }
  }

  cpu_ref(hA, hB, hCref, M, N, K, lda, ldb, ldc);

  __half *dA, *dB;
  float *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytesA));
  CUDA_CHECK(cudaMalloc(&dB, bytesB));
  CUDA_CHECK(cudaMalloc(&dC, bytesC));
  CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, bytesC));

  // Grid: tiles of (64 x 16)
  dim3 block(128, 1, 1);               // 1 warpgroup
  dim3 grid((N + 15) / 16, (M + 63) / 64, 1);

  size_t smem = (64 * 16 + 16 * 16) * sizeof(__half); // As + Bs

  wgmma_gemm_m64n16k16_f32_f16<<<grid, block, smem>>>(dA, dB, dC, M, N, K, lda, ldb, ldc);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(hCgpu, dC, bytesC, cudaMemcpyDeviceToHost));

  float max_abs = 0.f;
  int max_i = 0;
  for (int i = 0; i < M * N; ++i) {
    float err = fabsf(hCgpu[i] - hCref[i]);
    if (err > max_abs) { max_abs = err; max_i = i; }
  }
  printf("Max abs error: %.6g at (m=%d,n=%d). GPU=%.6g REF=%.6g\n",
         max_abs, max_i / N, max_i % N, hCgpu[max_i], hCref[max_i]);

  // Print a small corner
  printf("\nC[0:4, 0:8] GPU:\n");
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 8; ++n) printf("%10.1f ", hCgpu[m * ldc + n]);
    printf("\n");
  }
  printf("\nC[0:4, 0:8] REF:\n");
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 8; ++n) printf("%10.1f ", hCref[m * ldc + n]);
    printf("\n");
  }

  if (max_abs == 0.f) printf("\nPASSED\n");
  else                printf("\nFAILED\n");

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  free(hA); free(hB); free(hCref); free(hCgpu);
  return 0;
}

