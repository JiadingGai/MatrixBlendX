import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================================
// 1. WMMA Implementation (Matching test_reduce_v1_e2e.py)
// ============================================================================
__global__ void wmma_kernel(const half* A, const half* B, float* C) {
    __shared__ half As[16][16];
    int tid = threadIdx.x;
    
    // Stage A in shared memory (like in your script)
    for(int i = tid; i < 256; i += 32) {
        ((half*)As)[i] = A[i];
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    // B is row-major in memory, but we load as col_major to transpose it
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Load A from shared, load B directly from global
    wmma::load_matrix_sync(a_frag, &As[0][0], 16);
    wmma::load_matrix_sync(b_frag, B, 16); // B is read from global memory!

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

// ============================================================================
// 2. Inline PTX Implementation
// ============================================================================
__global__ void ptx_kernel(const half* A, const half* B, float* C) {
    __shared__ half As[16][16];
    __shared__ half Bs[16][16]; // We MUST stage B in shared memory for ldmatrix!

    int tid = threadIdx.x;
    
    for(int i = tid; i < 256; i += 32) {
        ((half*)As)[i] = A[i];
        ((half*)Bs)[i] = B[i];
    }
    __syncthreads();

    uint32_t reg_a[4];
    uint32_t reg_b[4];
    float reg_c[8] = {0.0f};

    uint32_t smem_a, smem_b;
    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }"
        : "=r"(smem_a) : "l"(&As[tid % 16][0]));
    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }"
        : "=r"(smem_b) : "l"(&Bs[tid % 16][0]));

    // 1. Load A normally (Row-Major)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(reg_a[0]), "=r"(reg_a[1]), "=r"(reg_a[2]), "=r"(reg_a[3]) : "r"(smem_a));

    // 2. MAGIC: Load B with .trans! 
    // This reads the row-major shared memory and transposes it into the registers, 
    // perfectly mimicking the wmma::col_major trick.
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(reg_b[0]), "=r"(reg_b[1]), "=r"(reg_b[2]), "=r"(reg_b[3]) : "r"(smem_b));

    // 3. Tensor Core Math
    asm volatile(
        "mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, "
        "{%16, %17, %18, %19, %20, %21, %22, %23};\n"
        : "=f"(reg_c[0]), "=f"(reg_c[1]), "=f"(reg_c[2]), "=f"(reg_c[3]),
          "=f"(reg_c[4]), "=f"(reg_c[5]), "=f"(reg_c[6]), "=f"(reg_c[7])
        : "r"(reg_a[0]), "r"(reg_a[1]), "r"(reg_a[2]), "r"(reg_a[3]),
          "r"(reg_b[0]), "r"(reg_b[1]), "r"(reg_b[2]), "r"(reg_b[3]),
          "f"(reg_c[0]), "f"(reg_c[1]), "f"(reg_c[2]), "f"(reg_c[3]),
          "f"(reg_c[4]), "f"(reg_c[5]), "f"(reg_c[6]), "f"(reg_c[7])
    );

    // 4. Manual Store
    int r0 = tid / 4;
    int c0 = (tid % 4) * 2;

    C[r0 * 16 + c0]     = reg_c[0];
    C[r0 * 16 + c0 + 1] = reg_c[1];
    C[r0 * 16 + c0 + 8] = reg_c[2];
    C[r0 * 16 + c0 + 9] = reg_c[3];

    int r1 = r0 + 8;
    C[r1 * 16 + c0]     = reg_c[4];
    C[r1 * 16 + c0 + 1] = reg_c[5];
    C[r1 * 16 + c0 + 8] = reg_c[6];
    C[r1 * 16 + c0 + 9] = reg_c[7];
}

torch::Tensor run_wmma(torch::Tensor A, torch::Tensor B) {
    auto C = torch::zeros({16, 16}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));
    wmma_kernel<<<1, 32>>>((const half*)A.data_ptr<at::Half>(), (const half*)B.data_ptr<at::Half>(), C.data_ptr<float>());
    return C;
}

torch::Tensor run_ptx(torch::Tensor A, torch::Tensor B) {
    auto C = torch::zeros({16, 16}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));
    ptx_kernel<<<1, 32>>>((const half*)A.data_ptr<at::Half>(), (const half*)B.data_ptr<at::Half>(), C.data_ptr<float>());
    return C;
}
'''

cpp_source = '''
torch::Tensor run_wmma(torch::Tensor A, torch::Tensor B);
torch::Tensor run_ptx(torch::Tensor A, torch::Tensor B);
'''

ext = load_inline(
    name="wmma_moe_ptx",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["run_wmma", "run_ptx"],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-arch=sm_90a"], # Targeting your SM90a architecture
)

def main():
    torch.manual_seed(42)
    device = torch.device("cuda")

    # Both matrices are Row-Major in memory
    A = torch.randn(16, 16, dtype=torch.float16, device=device)
    B = torch.randn(16, 16, dtype=torch.float16, device=device)

    # Because both kernels transpose B on the fly, the math is C = A @ B.T
    C_ref = torch.matmul(A.to(torch.float32), B.t().to(torch.float32))

    C_wmma = ext.run_wmma(A, B)
    C_ptx = ext.run_ptx(A, B)

    print("--- MoE Tile Results ---")
    print(f"Max Diff (PyTorch vs WMMA): {(C_ref - C_wmma).abs().max().item():.6f}")
    print(f"Max Diff (PyTorch vs PTX):  {(C_ref - C_ptx).abs().max().item():.6f}")
    print(f"Max Diff (WMMA vs PTX):     {(C_wmma - C_ptx).abs().max().item():.6f}")
    
    if torch.allclose(C_wmma, C_ptx):
        print("\n✅ SUCCESS: The PTX ldmatrix.trans implementation matches your wmma::col_major global load trick perfectly!")

if __name__ == "__main__":
    main()
