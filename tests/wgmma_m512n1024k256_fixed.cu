// wgmma_m512n1024k256_fixed.cu
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

// PTX shared-memory descriptor (no-swizzle). See PTX "Matrix Descriptor Format".
// addr/lbo/sbo are encoded in 16-byte units.
static __device__ __forceinline__ unsigned long long
make_wgmma_desc_noswizzle(const void* smem_ptr, int lbo_bytes, int sbo_bytes) {
  unsigned int smem_addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
  unsigned long long addr = ((unsigned long long)((smem_addr & 0x3FFFF) >> 4)) & 0x3FFFULL;
  unsigned long long lbo  = ((unsigned long long)(((lbo_bytes & 0x3FFFF) >> 4)) & 0x3FFFULL) << 16;
  unsigned long long sbo  = ((unsigned long long)(((sbo_bytes & 0x3FFFF) >> 4)) & 0x3FFFULL) << 32;
  unsigned long long base = 0ULL << 49;   // ignored for swizzle=0
  unsigned long long swz  = 0ULL << 62;   // no-swizzle
  return addr | lbo | sbo | base | swz;
}

// One CTA = one warpgroup = 128 threads.
// Each warpgroup computes one output tile: C[64x16] for a given (tile_m, tile_n).
__global__ void wgmma_gemm_m64n16k16_f32_f16_f16(
    const __half* __restrict__ A,   // row-major [M x K]
    const __half* __restrict__ B,   // row-major [K x N]
    float* __restrict__ C,          // row-major [M x N]
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
#if __CUDA_ARCH__ < 900
  (void)A; (void)B; (void)C; (void)M; (void)N; (void)K; (void)lda; (void)ldb; (void)ldc;
  return;
#else
  // Must be exactly 128 threads.
  const int tid = (int)threadIdx.x;

  // CTA tile coordinates
  const int tile_m = (int)blockIdx.y; // 64 rows
  const int tile_n = (int)blockIdx.x; // 16 cols
  const int m0 = tile_m * 64;
  const int n0 = tile_n * 16;

  // Shared tiles (A: 64x16, B: 16x16) in row-major.
  extern __shared__ unsigned char smem_raw[];
  __half* As = reinterpret_cast<__half*>(smem_raw);                                 // 64*16
  __half* Bs = reinterpret_cast<__half*>(smem_raw + 64 * 16 * sizeof(__half));      // 16*16

  // Accumulator fragment for m64n16 in f32: N/2 = 8 regs per thread.
  float d0=0.f,d1=0.f,d2=0.f,d3=0.f,d4=0.f,d5=0.f,d6=0.f,d7=0.f;

  // Fence before first use
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");

  // K loop, step 16 (matches instruction k16)
  for (int k0 = 0; k0 < K; k0 += 16) {
    // Stage A tile: As[r,c] = A[m0+r, k0+c]
    for (int idx = tid; idx < 64 * 16; idx += 128) {
      int r = idx / 16;
      int c = idx % 16;
      int gr = m0 + r;
      int gc = k0 + c;
      __half v = __float2half(0.0f);
      if (gr < M && gc < K) v = A[gr * lda + gc];
      As[idx] = v;
    }

    // Stage B tile: Bs[r,c] = B[k0+r, n0+c]
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

    // Descriptors: row-major, leading-dim = 16 * sizeof(half) = 32 bytes
    // sbo = 0 for canonical contiguous
    unsigned long long descA = make_wgmma_desc_noswizzle(As, 16 * (int)sizeof(__half), 0);
    unsigned long long descB = make_wgmma_desc_noswizzle(Bs, 16 * (int)sizeof(__half), 0);

    // wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16
    // Operands: {d0..d7}, descA, descB, scaleD, scaleA, scaleB, transA, transB
    // scaleD=1 means accumulate into provided d-regs.
    asm volatile(
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, "
      "%8, %9, "
      "1, 1, 1, 0, 0;\n"
      : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3),"+f"(d4),"+f"(d5),"+f"(d6),"+f"(d7)
      : "l"(descA), "l"(descB)
    );

    // Commit + wait for this group (simple, correctness-focused)
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");

    __syncthreads();
  }

  // -------------------------
  // Correct store mapping for m64n16 (Figure 149)
  //
  // Let warp = tid/32 in {0..3}
  // Let lane = tid%32 in {0..31}
  // Decompose lane into:
  //   r_in8 = lane/4  in {0..7}
  //   c_in4 = lane%4  in {0..3}
  //
  // This thread owns rows:
  //   r0 = warp*16 + r_in8
  //   r1 = r0 + 8
  // and columns:
  //   c0 = c_in4
  //   c1 = c0 + 4
  //   c2 = c0 + 8
  //   c3 = c0 + 12
  //
  // And the regs map as:
  //   (r0,c0)=d0, (r0,c1)=d1, (r0,c2)=d4, (r0,c3)=d5
  //   (r1,c0)=d2, (r1,c1)=d3, (r1,c2)=d6, (r1,c3)=d7
  // -------------------------
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int r_in8 = lane >> 2;      // /4
  const int c_in4 = lane & 3;       // %4

  const int r0 = warp * 16 + r_in8;
  const int r1 = r0 + 8;

  const int c0 = c_in4;
  const int c1 = c0 + 4;
  const int c2 = c0 + 8;
  const int c3 = c0 + 12;

  // Global coords for this CTA tile
  const int gr0 = m0 + r0;
  const int gr1 = m0 + r1;

  const int gc0 = n0 + c0;
  const int gc1 = n0 + c1;
  const int gc2 = n0 + c2;
  const int gc3 = n0 + c3;

  if (gr0 < M) {
    if (gc0 < N) C[gr0 * ldc + gc0] = d0;
    if (gc1 < N) C[gr0 * ldc + gc1] = d1;
    if (gc2 < N) C[gr0 * ldc + gc2] = d4;
    if (gc3 < N) C[gr0 * ldc + gc3] = d5;
  }
  if (gr1 < M) {
    if (gc0 < N) C[gr1 * ldc + gc0] = d2;
    if (gc1 < N) C[gr1 * ldc + gc1] = d3;
    if (gc2 < N) C[gr1 * ldc + gc2] = d6;
    if (gc3 < N) C[gr1 * ldc + gc3] = d7;
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
  const int M = 512, N = 1024, K = 256;
  const int lda = K, ldb = N, ldc = N;

  printf("Testing WGMMA GEMM (Hopper sm_90a): M=%d N=%d K=%d\n", M, N, K);

  size_t bytesA = (size_t)M * lda * sizeof(__half);
  size_t bytesB = (size_t)K * ldb * sizeof(__half);
  size_t bytesC = (size_t)M * ldc * sizeof(float);

  __half* hA = (__half*)malloc(bytesA);
  __half* hB = (__half*)malloc(bytesB);
  float*  hCref = (float*)malloc(bytesC);
  float*  hCgpu = (float*)malloc(bytesC);

  // Same init as your failing run (small-ish numbers, signed)
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

  dim3 block(128, 1, 1);                         // 1 warpgroup
  dim3 grid((N + 15) / 16, (M + 63) / 64, 1);     // tiles of 64x16
  size_t smem = (64 * 16 + 16 * 16) * sizeof(__half);

  wgmma_gemm_m64n16k16_f32_f16_f16<<<grid, block, smem>>>(dA, dB, dC, M, N, K, lda, ldb, ldc);
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

  if (max_abs <= 1e-2f) printf("\nPASSED\n");
  else                  printf("\nFAILED\n");

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  free(hA); free(hB); free(hCref); free(hCgpu);
  return 0;
}

