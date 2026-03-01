// reduce_v1.cu
// Compile (example):
//   nvcc -O3 -std=c++17 -arch=sm_90a reduce_v1.cu -o reduce_v1
//
// Notes:
// - This is a minimal semantic port of the Triton kernel.
// - It assumes fp16 inputs/outputs and fp32 accumulation.
// - It uses WMMA (16x16x16). N and K should be multiples of 16 for full speed.
// - "sorted_token_ids" contains routed-row ids rt in [0, EM) plus padding sentinels >= num_valid_tokens.
// - C is treated as [EM, N] row-major (leading dimension N).

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

using half = __half;
using namespace nvcuda;

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  exit(1); } } while(0)
#endif

// Tile sizes (match your Triton defaults)
constexpr int BM = 64;   // BLOCK_SIZE_M
constexpr int BN = 64;   // BLOCK_SIZE_N
constexpr int BK = 16;   // WMMA K tile (must be 16 for wmma::mma_sync)

template <bool MUL_ROUTED_WEIGHT>
__global__ void reduce_v1_wmma(
    const half* __restrict__ A,          // [M, K] row-major
    const half* __restrict__ B,          // [E, N, K] row-major in (N,K) for each expert
    half* __restrict__ C,                // [EM, N] row-major
    const half* __restrict__ topk_w,     // [EM] (optional)
    const int32_t* __restrict__ sorted_token_ids, // [num_tokens_post_padded]
    const int32_t* __restrict__ expert_ids,       // [ceil(EM/BM)]
    const int32_t* __restrict__ num_tokens_post_padded_ptr, // scalar
    int N, int K, int EM, int num_valid_tokens,
    int stride_am, int stride_ak,     // A strides in elements
    int stride_be, int stride_bn, int stride_bk, // B strides in elements for [E,N,K]
    int stride_cm, int stride_cn,     // C strides in elements for [EM,N]
    int top_k
) {
  // Map linear block idx -> (pid_m, pid_n)
  int num_pid_m = (EM + BM - 1) / BM;
  int num_pid_n = (N  + BN - 1) / BN;

  int pid = (int)blockIdx.x;
  int pid_m = pid / num_pid_n;
  int pid_n = pid - pid_m * num_pid_n;

  int num_tokens_post_padded = *num_tokens_post_padded_ptr;
  if (pid_m * BM >= num_tokens_post_padded) return;

  // One expert per M-tile
  int e = (int)expert_ids[pid_m];
  if (e == -1) {
    // write zeros to C[rt, offs_n] for valid rt lanes
    int lane = (int)threadIdx.x;
    // simple parallel zeroing over tile (not tensor-core path)
    for (int i = lane; i < BM * BN; i += (int)blockDim.x) {
      int mi = i / BN;
      int ni = i - mi * BN;
      int rt = (int)sorted_token_ids[pid_m * BM + mi];
      if (rt < num_valid_tokens) {
        int n = pid_n * BN + ni;
        if (n < N) {
          C[rt * stride_cm + n * stride_cn] = __float2half(0.0f);
        }
      }
    }
    return;
  }

  // Shared memory staging for gathered A tiles (for each 16x16 A fragment)
  // We stage a 64x16 slab per K-step (BM x BK).
  __shared__ half As[BM * BK];  // row-major [BM, BK]

  // WMMA accumulators for a 64x64 output tile: 4x4 of (16x16)
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
  #pragma unroll
  for (int mi = 0; mi < 4; ++mi) {
    #pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      wmma::fill_fragment(acc[mi][ni], 0.0f);
    }
  }

  // Preload routed rows rt[] and tok[] for the BM rows
  // We'll keep them in registers for store + masks.
  int rt_row[BM];
  int tok_row[BM];
  bool row_valid[BM];

  // Load rt for each row (each thread loads multiple)
  for (int i = (int)threadIdx.x; i < BM; i += (int)blockDim.x) {
    int rt = (int)sorted_token_ids[pid_m * BM + i];
    rt_row[i] = rt;
    bool valid = (rt < num_valid_tokens);
    row_valid[i] = valid;
    tok_row[i] = valid ? (rt / top_k) : 0;
  }
  __syncthreads();

  // Main K loop in steps of 16 (WMMA K)
  for (int k0 = 0; k0 < K; k0 += BK) {
    // Stage A slab [BM, BK] into shared with gather.
    // As[i, kk] = A[tok_row[i], k0+kk] if valid & in-range else 0.
    for (int idx = (int)threadIdx.x; idx < BM * BK; idx += (int)blockDim.x) {
      int i  = idx / BK;
      int kk = idx - i * BK;
      half v = __float2half(0.0f);
      if (row_valid[i]) {
        int tok = tok_row[i];
        int k = k0 + kk;
        if (k < K) {
          v = A[tok * stride_am + k * stride_ak];
        }
      }
      As[idx] = v;
    }
    __syncthreads();

    // For each (m_tile, n_tile) do WMMA on 16x16 blocks
    // A tile: from shared As, row-major, ld = BK (16)
    // B tile: treat B^T as col-major KxN with ld = K (full K),
    //         tile origin (k0, n0) corresponds to pointer &B[e, n0, k0]
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
      const half* a_tile_ptr = &As[(mi * 16) * BK]; // row mi*16, col 0
      wmma::load_matrix_sync(a_frag, a_tile_ptr, BK);

      #pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        int n0 = pid_n * BN + ni * 16;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

        // B pointer to B[e, n0, k0] (row-major N×K),
        // interpreted as B^T col-major with ld=K.
        const half* b_tile_ptr =
            B + e * stride_be + n0 * stride_bn + k0 * stride_bk;

        // If we're in the tail (n0+15 >= N), we should mask.
        // WMMA load has no mask; simplest safe approach is to only run full tiles.
        // For your N=2880 (multiple of 16), this is fine.
        if (n0 + 15 < N) {
          wmma::load_matrix_sync(b_frag, b_tile_ptr, K);
          wmma::mma_sync(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
        }
      }
    }

    __syncthreads();
  }

  // Optional routed weight multiply (in fp32) and store results.
  // Store tile as 4x4 16x16 blocks to global C with scatter on rt.
  // We do a simple per-element store (not wmma store) to honor scatter + masks.
  for (int mi = 0; mi < 4; ++mi) {
    for (int ni = 0; ni < 4; ++ni) {
      // Dump accumulator fragment to a local array
      float tmp[16 * 16];
      #pragma unroll
      for (int t = 0; t < 16 * 16; ++t) tmp[t] = acc[mi][ni].x[t];

      // Each thread writes some elements
      for (int idx = (int)threadIdx.x; idx < 16 * 16; idx += (int)blockDim.x) {
        int r = idx / 16;
        int c = idx - r * 16;

        int row = mi * 16 + r;         // 0..63 within tile
        int col = ni * 16 + c;         // 0..63 within tile
        int n = pid_n * BN + col;       // global column

        if (n < N) {
          int rt = rt_row[row];
          if (rt < num_valid_tokens) {
            float val = tmp[idx];

            if constexpr (MUL_ROUTED_WEIGHT) {
              float w = __half2float(topk_w[rt]);
              val *= w;
            }

            C[rt * stride_cm + n * stride_cn] = __float2half_rn(val);
          }
        }
      }
    }
  }
}

void run_example(int M, int E, int N, int K, int top_k) {
  int EM = M * top_k;

  // Allocate tensors (device)
  half *A, *B, *C, *W;
  int32_t *sorted_ids, *expert_ids, *num_tokens_post_padded;

  size_t bytesA = (size_t)M * K * sizeof(half);
  size_t bytesB = (size_t)E * N * K * sizeof(half);
  size_t bytesC = (size_t)EM * N * sizeof(half);
  size_t bytesW = (size_t)EM * sizeof(half);

  CHECK_CUDA(cudaMalloc(&A, bytesA));
  CHECK_CUDA(cudaMalloc(&B, bytesB));
  CHECK_CUDA(cudaMalloc(&C, bytesC));
  CHECK_CUDA(cudaMalloc(&W, bytesW));

  CHECK_CUDA(cudaMalloc(&sorted_ids, (size_t)EM * sizeof(int32_t)));
  int num_pid_m = (EM + BM - 1) / BM;
  CHECK_CUDA(cudaMalloc(&expert_ids, (size_t)num_pid_m * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&num_tokens_post_padded, sizeof(int32_t)));

  // Initialize (toy): sorted_ids = 0..EM-1, expert_ids cycles, num_tokens_post_padded=EM
  std::vector<int32_t> h_sorted(EM);
  for (int i = 0; i < EM; ++i) h_sorted[i] = i;
  std::vector<int32_t> h_expert(num_pid_m);
  for (int i = 0; i < num_pid_m; ++i) h_expert[i] = i % E;
  int32_t h_ntpp = EM;

  CHECK_CUDA(cudaMemcpy(sorted_ids, h_sorted.data(), (size_t)EM*sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(expert_ids, h_expert.data(), (size_t)num_pid_m*sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(num_tokens_post_padded, &h_ntpp, sizeof(int32_t), cudaMemcpyHostToDevice));

  // Fill A/B/W/C (placeholder init)
  CHECK_CUDA(cudaMemset(A, 0, bytesA));
  CHECK_CUDA(cudaMemset(B, 0, bytesB));
  CHECK_CUDA(cudaMemset(W, 0, bytesW));
  CHECK_CUDA(cudaMemset(C, 0, bytesC));

  // Launch config
  int num_pid_n = (N + BN - 1) / BN;
  int grid = num_pid_m * num_pid_n;

  dim3 block(256);
  dim3 gridDim(grid);

  // Strides in elements for row-major tensors:
  int stride_am = K, stride_ak = 1;
  int stride_be = N * K, stride_bn = K, stride_bk = 1;
  int stride_cm = N, stride_cn = 1;

  // ---------------- TIMING SETUP ----------------
  cudaEvent_t ev_start, ev_stop;
  CHECK_CUDA(cudaEventCreate(&ev_start));
  CHECK_CUDA(cudaEventCreate(&ev_stop));

  // Warmup (helps avoid first-launch overhead in timing)
  reduce_v1_wmma<false><<<gridDim, block>>>(
      A, B, C, W, sorted_ids, expert_ids, num_tokens_post_padded,
      N, K, EM, EM,
      stride_am, stride_ak,
      stride_be, stride_bn, stride_bk,
      stride_cm, stride_cn,
      top_k
  );
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed runs
  const int iters = 50;
  CHECK_CUDA(cudaEventRecord(ev_start));
  for (int it = 0; it < iters; ++it) {
    reduce_v1_wmma<false><<<gridDim, block>>>(
        A, B, C, W, sorted_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, EM,
        stride_am, stride_ak,
        stride_be, stride_bn, stride_bk,
        stride_cm, stride_cn,
        top_k
    );
  }
  CHECK_CUDA(cudaEventRecord(ev_stop));
  CHECK_CUDA(cudaEventSynchronize(ev_stop));
  CHECK_CUDA(cudaGetLastError());

  float total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
  float avg_ms = total_ms / iters;

  // Report effective TFLOPS for this kernel shape
  double flops = 2.0 * (double)EM * (double)N * (double)K; // GEMM-like FLOPs
  double tflops = flops / (avg_ms * 1e-3) / 1e12;

  printf("Timing: iters=%d total=%.3f ms avg=%.6f ms\n", iters, total_ms, avg_ms);
  printf("Theoretical work: %.3f GFLOPs; Effective: %.3f TFLOP/s\n",
         flops / 1e9, tflops);

  CHECK_CUDA(cudaEventDestroy(ev_start));
  CHECK_CUDA(cudaEventDestroy(ev_stop));
  // ---------------- END TIMING ----------------

  // cleanup
  cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(W);
  cudaFree(sorted_ids); cudaFree(expert_ids); cudaFree(num_tokens_post_padded);
}

int main() {
  // Your requested shape:
  run_example(/*M=*/48, /*E=*/128, /*N=*/2880, /*K=*/2880, /*top_k=*/4);
  printf("Done.\n");
  return 0;
}
