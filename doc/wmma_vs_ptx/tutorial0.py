import os
import torch
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# Full, self-contained e2e script:
#   - builds a CUDA extension with 2 kernels:
#       (1) WMMA version (wmma::matrix_b col_major trick => reads B^T)
#       (2) PTX version (ldmatrix.trans + 2x mma.sync m16n8k16 => full 16x16)
#   - compares against PyTorch reference: C = A @ B.T
# -----------------------------------------------------------------------------

cuda_src = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

static __device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    uint32_t smem_u32;
    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }"
        : "=r"(smem_u32) : "l"(ptr));
    return smem_u32;
}

// ============================================================================
// 1) WMMA kernel: C = A @ B.T
//    - A is staged in shared (row-major)
//    - B is in global row-major but loaded as col_major => effectively B^T
// ============================================================================
__global__ void wmma_kernel(const half* A, const half* B, float* C) {
    __shared__ __align__(16) half As[16][16];

    int tid = threadIdx.x; // 0..31
    for (int i = tid; i < 256; i += 32) {
        reinterpret_cast<half*>(As)[i] = A[i];
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    // B is row-major in memory, but we *interpret* it as col_major => transpose trick.
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, &As[0][0], 16);
    wmma::load_matrix_sync(b_frag, B, 16);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

// ============================================================================
// 2) PTX kernel: C = A @ B.T using ldmatrix.trans + mma.sync.m16n8k16 twice
//    - A staged in shared row-major
//    - B staged in shared row-major
//    - ldmatrix.trans makes B behave like col-major operand => transpose trick
// ============================================================================
__global__ void ptx_kernel(const half* A, const half* B, float* C) {
    __shared__ __align__(16) half As[16][16];
    __shared__ __align__(16) half Bs[16][16];

    int lane = threadIdx.x; // 0..31

    for (int i = lane; i < 256; i += 32) {
        reinterpret_cast<half*>(As)[i] = A[i];
        reinterpret_cast<half*>(Bs)[i] = B[i];
    }
    __syncthreads();

    // ---- Load A as 4x (8x8) using ldmatrix.x4 ----
    // x4 uses 4 groups of 8 threads; each group provides 8 row addresses:
    //   group = lane / 8, row_in = lane % 8
    //   base_row = (group & 1) * 8, base_col = (group >> 1) * 8
    uint32_t a_reg[4];
    int gA = lane >> 3;        // 0..3
    int rA = lane & 7;         // 0..7
    int base_rowA = (gA & 1) * 8;
    int base_colA = (gA >> 1) * 8;
    uint32_t a_smem = cvta_to_shared_u32(&As[base_rowA + rA][base_colA]);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])
        : "r"(a_smem)
    );

    // ---- Load B for two output tiles (N=8 left + N=8 right) ----
    // For mma.sync.m16n8k16.row.col, B operand is 16x8 col-major.
    // We want C = A @ B^T. With B stored row-major, we load an 8x16 "strip" of rows
    // and use ldmatrix.trans to transpose it into the registers as col-major.
    //
    // ldmatrix.x2 uses 2 groups of 8 threads (16 lanes). Lanes 16..31 can mirror lanes 0..15.
    auto b_smem_addr = [&](int base_row /*0 or 8*/) -> uint32_t {
        int l16 = lane & 15;   // 0..15
        int grp = l16 >> 3;    // 0 or 1  -> selects left/right half (col 0 or 8) of the 8x16 strip
        int r   = l16 & 7;     // 0..7    -> row within strip
        int c   = grp * 8;     // 0 or 8
        return cvta_to_shared_u32(&Bs[base_row + r][c]);
    };

    uint32_t b0_reg[2], b1_reg[2];

    // Output cols 0..7 come from B rows 0..7 (since B^T)
    uint32_t b0_smem = b_smem_addr(0);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(b0_reg[0]), "=r"(b0_reg[1])
        : "r"(b0_smem)
    );

    // Output cols 8..15 come from B rows 8..15
    uint32_t b1_smem = b_smem_addr(8);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(b1_reg[0]), "=r"(b1_reg[1])
        : "r"(b1_smem)
    );

    // ---- Two mma.sync ops: each produces a 16x8 tile => together 16x16 ----
    float d0_0=0.f, d0_1=0.f, d0_2=0.f, d0_3=0.f;
    float d1_0=0.f, d1_1=0.f, d1_2=0.f, d1_3=0.f;

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d0_0), "+f"(d0_1), "+f"(d0_2), "+f"(d0_3)
        : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
          "r"(b0_reg[0]), "r"(b0_reg[1])
    );

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d1_0), "+f"(d1_1), "+f"(d1_2), "+f"(d1_3)
        : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
          "r"(b1_reg[0]), "r"(b1_reg[1])
    );

    // ---- Store mapping (per warp) for each 16x8 tile ----
    // Each lane owns 4 outputs in a 16x8 tile:
    //   rows: r0 and r0+8
    //   cols: c0 and c0+1
    int r0 = lane / 4;          // 0..7
    int c0 = (lane % 4) * 2;    // 0,2,4,6

    // left tile -> cols 0..7
    C[(r0 + 0) * 16 + (c0 + 0)] = d0_0;
    C[(r0 + 0) * 16 + (c0 + 1)] = d0_1;
    C[(r0 + 8) * 16 + (c0 + 0)] = d0_2;
    C[(r0 + 8) * 16 + (c0 + 1)] = d0_3;

    // right tile -> cols 8..15
    C[(r0 + 0) * 16 + (c0 + 8)] = d1_0;
    C[(r0 + 0) * 16 + (c0 + 9)] = d1_1;
    C[(r0 + 8) * 16 + (c0 + 8)] = d1_2;
    C[(r0 + 8) * 16 + (c0 + 9)] = d1_3;
}

// ----------------------------------------------------------------------------
// PyTorch entry points
// ----------------------------------------------------------------------------
static inline void check_inputs(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16, "A and B must be float16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A and B must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(0)==16 && A.size(1)==16 && B.size(0)==16 && B.size(1)==16,
                "A and B must be [16,16]");
}

torch::Tensor run_wmma(torch::Tensor A, torch::Tensor B) {
    check_inputs(A, B);
    auto C = torch::zeros({16, 16}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));
    wmma_kernel<<<1, 32>>>(reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
                           reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
                           C.data_ptr<float>());
    return C;
}

torch::Tensor run_ptx(torch::Tensor A, torch::Tensor B) {
    check_inputs(A, B);
    auto C = torch::zeros({16, 16}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));
    ptx_kernel<<<1, 32>>>(reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
                          reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
                          C.data_ptr<float>());
    return C;
}
'''

cpp_src = r'''
#include <torch/extension.h>

torch::Tensor run_wmma(torch::Tensor A, torch::Tensor B);
torch::Tensor run_ptx(torch::Tensor A, torch::Tensor B);
'''

def build_ext():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    major, minor = torch.cuda.get_device_capability()
    # This path is really intended for tensor-core capable GPUs.
    if major < 8:
        raise RuntimeError(f"Need compute capability >= 8.0 for this script; got {major}.{minor}")

    arch = f"sm_{major}{minor}"

    # Optional: make builds deterministic-ish and more verbose
    # os.environ["TORCH_EXTENSIONS_DIR"] = "./torch_extensions"

    ext = load_inline(
        name="wmma_vs_ptx_ldmatrix_trans",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["run_wmma", "run_ptx"],
        with_cuda=True,
        extra_cuda_cflags=[
            "-O3",
            f"-arch={arch}",
            "--expt-relaxed-constexpr",
        ],
        verbose=True,
    )
    return ext

def main():
    torch.manual_seed(42)
    device = torch.device("cuda")

    ext = build_ext()

    # Both matrices row-major in memory
    A = torch.randn(16, 16, dtype=torch.float16, device=device).contiguous()
    B = torch.randn(16, 16, dtype=torch.float16, device=device).contiguous()

    # Both kernels implement C = A @ B.T
    C_ref = torch.matmul(A.to(torch.float32), B.t().to(torch.float32))

    C_wmma = ext.run_wmma(A, B)
    C_ptx  = ext.run_ptx(A, B)
    torch.cuda.synchronize()

    print("--- 16x16 Tile Results (C = A @ B.T) ---")
    print(f"Max Diff (PyTorch vs WMMA): {(C_ref - C_wmma).abs().max().item():.6e}")
    print(f"Max Diff (PyTorch vs PTX):  {(C_ref - C_ptx ).abs().max().item():.6e}")
    print(f"Max Diff (WMMA vs PTX):     {(C_wmma - C_ptx).abs().max().item():.6e}")

    ok = torch.allclose(C_wmma, C_ptx, rtol=0.0, atol=0.0)
    print("\nWMMA == PTX bitwise:", ok)
    if not ok:
        ok2 = torch.allclose(C_wmma, C_ptx, rtol=1e-5, atol=1e-4)
        print("WMMA ~= PTX (tolerant):", ok2)

if __name__ == "__main__":
    main()
