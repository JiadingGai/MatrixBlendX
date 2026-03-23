// hgemm_raw.cu — Raw CUDA + inline PTX HGEMM kernel (TN layout, SM80)
//
// Computes C = A * B^T where:
//   A is (M, K) row-major, B is (N, K) row-major, C is (M, N) column-major
//
// CTA tile: 128x128x64, 3-stage async pipeline, 128 threads (4 warps in 2x2)
// MMA: mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
// Smem layout: Swizzle<3,3,3> with 8x(8x8) atom
//
// How to run and profile:
//   - Correctness + benchmark: python test_hgemm_raw.py
//       JIT-compiles via torch.utils.cpp_extension.load_inline,
//       runs correctness checks then times 20 iterations
//   - Standalone benchmark: inline Python script doing the same
//       load_inline + CUDA event timing (avoids compilation warmup bias)
//   - ncu profiling (needs sudo for GPU perf counters):
//       sudo /usr/local/cuda-12.8/bin/ncu --set full \
//         --kernel-name hgemm_raw_kernel --launch-skip 3 --launch-count 1 \
//         -o /tmp/hgemm_profile python3 /tmp/profile_kernel.py

#include <cuda_fp16.h>
#include <cstdint>

// =====================================================================
// PTX helpers
// =====================================================================

__device__ __forceinline__ uint32_t cvta_to_shared(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(addr) : "l"(ptr));
    return addr;
}

__device__ __forceinline__ void cp_async_ca_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t smem_addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr));
}

__device__ __forceinline__ void mma_m16n8k16_f16(
    uint32_t& d0, uint32_t& d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16\n"
        "  {%0, %1},\n"
        "  {%2, %3, %4, %5},\n"
        "  {%6, %7},\n"
        "  {%8, %9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1));
}

// =====================================================================
// Swizzled shared memory offset (in half elements, stage=0)
// =====================================================================

__device__ __forceinline__ int smem_off(int m, int k) {
    int inner = ((m & 7) << 3) | (k & 7) | ((k >> 3) << 6);
    int swizzled = inner ^ ((inner >> 3) & 0x38);
    return ((m >> 3) << 9) + swizzled;
}

// =====================================================================
// Constants
// =====================================================================

static constexpr int BM = 128;
static constexpr int BN = 128;
static constexpr int BK = 64;
static constexpr int STAGES = 3;
static constexpr int NTHREADS = 128;
static constexpr int K_BLOCKS = 4;     // BK / 16
static constexpr int MMA_M = 4;        // 64 / 16
static constexpr int MMA_N = 8;        // 64 / 8
static constexpr int STAGE_SIZE = BM * BK;  // 8192 half elements
static constexpr int SMEM_A_SIZE = (STAGES * STAGE_SIZE);
static constexpr int SMEM_BYTES = 2 * SMEM_A_SIZE * (int)sizeof(half);
static constexpr int STAGE_BYTES = STAGE_SIZE * (int)sizeof(half); // 16384 bytes

// =====================================================================
// Main kernel
// =====================================================================

__global__ __launch_bounds__(NTHREADS)
void hgemm_raw_kernel(
    int M, int N, int K,
    const half* __restrict__ A, int ldA,
    const half* __restrict__ B, int ldB,
    half* __restrict__ C, int ldC)
{
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    const int warp_m = (warp_id >> 1) << 6;
    const int warp_n = (warp_id & 1) << 6;

    // CTA-level swizzle for L2 cache locality
    const int num_m_blocks = (M + BM - 1) / BM;
    const int num_n_blocks = (N + BN - 1) / BN;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int sw = 4;
    if (sw > num_m_blocks) sw = num_m_blocks;
    int group_id = bid / (sw * num_n_blocks);
    int group_rem = bid % (sw * num_n_blocks);
    int block_m = group_id * sw + group_rem % sw;
    int block_n = group_rem / sw;
    const int cta_m = block_m * BM;
    const int cta_n = block_n * BN;

    // Copy thread layout: 16x8 k-major
    const int copy_row = tid >> 3;
    const int copy_col = tid & 7;

    // ldmatrix lane decomposition
    const int ldm_row = lane_id & 7;
    const int ldm_dm  = (lane_id >> 4) << 3;
    const int ldm_dk  = ((lane_id >> 3) & 1) << 3;

    extern __shared__ char smem_raw[];
    half* sA = reinterpret_cast<half*>(smem_raw);
    half* sB = sA + SMEM_A_SIZE;

    // ================================================================
    // Precompute shared memory addresses (stage=0 base)
    // ================================================================
    const uint32_t sA_base = cvta_to_shared(sA);
    const uint32_t sB_base = cvta_to_shared(sB);

    // ldmatrix addresses for A: [MMA_M][K_BLOCKS]
    uint32_t ldm_a_addr[MMA_M][K_BLOCKS];
    #pragma unroll
    for (int mm = 0; mm < MMA_M; mm++) {
        int m_addr = warp_m + mm * 16 + ldm_dm + ldm_row;
        #pragma unroll
        for (int kb = 0; kb < K_BLOCKS; kb++) {
            ldm_a_addr[mm][kb] = sA_base + smem_off(m_addr, (kb << 4) + ldm_dk) * 2;
        }
    }

    // ldmatrix addresses for B: [MMA_N/2][K_BLOCKS]
    uint32_t ldm_b_addr[MMA_N / 2][K_BLOCKS];
    #pragma unroll
    for (int bg = 0; bg < MMA_N / 2; bg++) {
        int n_addr = warp_n + bg * 16 + ldm_dm + ldm_row;
        #pragma unroll
        for (int kb = 0; kb < K_BLOCKS; kb++) {
            ldm_b_addr[bg][kb] = sB_base + smem_off(n_addr, (kb << 4) + ldm_dk) * 2;
        }
    }

    // cp_async smem addresses and global base pointers
    const int kcol = copy_col << 3;
    uint32_t cp_a_addr[8], cp_b_addr[8];
    const half* gmem_a_base[8];
    const half* gmem_b_base[8];
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        int row = copy_row + p * 16;
        cp_a_addr[p] = sA_base + smem_off(row, kcol) * 2;
        cp_b_addr[p] = sB_base + smem_off(row, kcol) * 2;
        gmem_a_base[p] = A + (int64_t)(cta_m + row) * ldA + kcol;
        gmem_b_base[p] = B + (int64_t)(cta_n + row) * ldB + kcol;
    }

    // Accumulators
    uint32_t frag_c[MMA_M][MMA_N][2];
    #pragma unroll
    for (int i = 0; i < MMA_M; i++)
        #pragma unroll
        for (int j = 0; j < MMA_N; j++)
            frag_c[i][j][0] = frag_c[i][j][1] = 0;

    // A/B fragments: all K_BLOCKS for full unrolling
    uint32_t frag_a[K_BLOCKS][MMA_M][4];
    uint32_t frag_b[K_BLOCKS][MMA_N][2];

    // ================================================================
    // Pipeline prefetch
    // ================================================================
    int k_tile_count = K / BK;
    int k_tile_next = 0;

    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        int k_off_base = k_tile_next * BK;
        uint32_t stage_off = s * STAGE_BYTES;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            cp_async_ca_16(cp_a_addr[p] + stage_off, gmem_a_base[p] + k_off_base);
            cp_async_ca_16(cp_b_addr[p] + stage_off, gmem_b_base[p] + k_off_base);
        }
        cp_async_commit();
        k_tile_count--;
        if (k_tile_count > 0) k_tile_next++;
    }

    cp_async_wait<STAGES - 2>();
    __syncthreads();

    int pipe_read = 0;
    int pipe_write = STAGES - 1;
    int frag_stage = 0;

    // Prefetch k_block=0 fragments
    {
        #pragma unroll
        for (int mm = 0; mm < MMA_M; mm++)
            ldmatrix_x4(frag_a[0][mm][0], frag_a[0][mm][1], frag_a[0][mm][2], frag_a[0][mm][3],
                        ldm_a_addr[mm][0]);
        #pragma unroll
        for (int bg = 0; bg < MMA_N / 2; bg++) {
            uint32_t r0, r1, r2, r3;
            ldmatrix_x4(r0, r1, r2, r3, ldm_b_addr[bg][0]);
            frag_b[0][bg * 2][0] = r0; frag_b[0][bg * 2][1] = r1;
            frag_b[0][bg * 2 + 1][0] = r2; frag_b[0][bg * 2 + 1][1] = r3;
        }
    }

    // ================================================================
    // Main loop — spread global loads across k_block iterations
    // ================================================================
    while (k_tile_count > -(STAGES - 1)) {

        // At the start of each outer iteration, determine the copy target stage
        // and update pipeline state. copy_stage holds the OLD pipe_write.
        int copy_stage = pipe_write;
        int k_off_copy = k_tile_next * BK;
        pipe_write = pipe_read;
        pipe_read = (pipe_read == STAGES - 1) ? 0 : pipe_read + 1;

        uint32_t copy_stage_off = copy_stage * STAGE_BYTES;

        #pragma unroll
        for (int kb = 0; kb < K_BLOCKS; kb++) {

            // Spread global loads: 2 copy pairs per kb
            {
                #pragma unroll
                for (int p = kb * 2; p < kb * 2 + 2; p++) {
                    cp_async_ca_16(cp_a_addr[p] + copy_stage_off, gmem_a_base[p] + k_off_copy);
                    cp_async_ca_16(cp_b_addr[p] + copy_stage_off, gmem_b_base[p] + k_off_copy);
                }
            }

            if (kb == K_BLOCKS - 1) {
                // Commit all 8 copy pairs
                cp_async_commit();
                k_tile_count--;
                if (k_tile_count > 0) k_tile_next++;

                // Wait for the stage we're about to read
                frag_stage = pipe_read;
                cp_async_wait<STAGES - 2>();
                __syncthreads();
            }

            int kb_next = (kb + 1) & (K_BLOCKS - 1);

            // Prefetch next k_block fragments
            {
                uint32_t stage_off = frag_stage * STAGE_BYTES;
                #pragma unroll
                for (int mm = 0; mm < MMA_M; mm++)
                    ldmatrix_x4(frag_a[kb_next][mm][0], frag_a[kb_next][mm][1],
                                frag_a[kb_next][mm][2], frag_a[kb_next][mm][3],
                                ldm_a_addr[mm][kb_next] + stage_off);
                #pragma unroll
                for (int bg = 0; bg < MMA_N / 2; bg++) {
                    uint32_t r0, r1, r2, r3;
                    ldmatrix_x4(r0, r1, r2, r3, ldm_b_addr[bg][kb_next] + stage_off);
                    frag_b[kb_next][bg * 2][0] = r0; frag_b[kb_next][bg * 2][1] = r1;
                    frag_b[kb_next][bg * 2 + 1][0] = r2; frag_b[kb_next][bg * 2 + 1][1] = r3;
                }
            }

            // MMA: a0=r0, a1=r2, a2=r1, a3=r3 (ldmatrix register reorder)
            #pragma unroll
            for (int mm = 0; mm < MMA_M; mm++) {
                #pragma unroll
                for (int mn = 0; mn < MMA_N; mn++) {
                    mma_m16n8k16_f16(
                        frag_c[mm][mn][0], frag_c[mm][mn][1],
                        frag_a[kb][mm][0], frag_a[kb][mm][2],
                        frag_a[kb][mm][1], frag_a[kb][mm][3],
                        frag_b[kb][mn][0], frag_b[kb][mn][1],
                        frag_c[mm][mn][0], frag_c[mm][mn][1]);
                }
            }
        }
    }

    // ================================================================
    // Epilogue: coalesced store via shared memory
    // C is column-major: C[row, col] at address C + row + col * ldC
    // Coalesced writes require threads to access consecutive rows.
    // ================================================================

    // Reuse shared memory for epilogue staging (128x128 tile, padded rows)
    constexpr int SMEM_C_STRIDE = 128 + 8; // pad to reduce bank conflicts
    half* sC = reinterpret_cast<half*>(smem_raw);

    const int c_row = lane_id >> 2;        // 0-7
    const int c_col = (lane_id & 3) << 1;  // 0,2,4,6

    // Write fragments to shared memory
    #pragma unroll
    for (int mm = 0; mm < MMA_M; mm++) {
        #pragma unroll
        for (int mn = 0; mn < MMA_N; mn++) {
            int local_m = warp_m + mm * 16;
            int local_n = warp_n + mn * 8;

            // d0: rows 0-7
            {
                int r = local_m + c_row;
                int c = local_n + c_col;
                sC[r * SMEM_C_STRIDE + c]     = reinterpret_cast<const half*>(&frag_c[mm][mn][0])[0];
                sC[r * SMEM_C_STRIDE + c + 1] = reinterpret_cast<const half*>(&frag_c[mm][mn][0])[1];
            }
            // d1: rows 8-15
            {
                int r = local_m + c_row + 8;
                int c = local_n + c_col;
                sC[r * SMEM_C_STRIDE + c]     = reinterpret_cast<const half*>(&frag_c[mm][mn][1])[0];
                sC[r * SMEM_C_STRIDE + c + 1] = reinterpret_cast<const half*>(&frag_c[mm][mn][1])[1];
            }
        }
    }

    __syncthreads();

    // Coalesced global stores: all 128 threads write consecutive rows per column
    // 128 threads × 1 element = 128 rows per column, iterate over 128 columns
    half* C_base = C + cta_m + (int64_t)cta_n * ldC;
    #pragma unroll
    for (int col = 0; col < BN; col++) {
        C_base[tid + (int64_t)col * ldC] = sC[tid * SMEM_C_STRIDE + col];
    }
    // Second half of rows (threads cover rows 0-127, tid maps to row directly)
    // Actually tid goes 0-127 which covers all 128 rows. Done.
}

// =====================================================================
// Launcher
// =====================================================================

void gemm_tn(int m, int n, int k,
             const half* A, int ldA,
             const half* B, int ldB,
             half* C, int ldC,
             cudaStream_t stream = 0)
{
    int grid_m = (m + BM - 1) / BM;
    int grid_n = (n + BN - 1) / BN;
    dim3 grid(grid_m * grid_n, 1);
    dim3 block(NTHREADS);

    cudaFuncSetAttribute(hgemm_raw_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(hgemm_raw_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    hgemm_raw_kernel<<<grid, block, SMEM_BYTES, stream>>>(
        m, n, k, A, ldA, B, ldB, C, ldC);
}
