#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <ATen/ATen.h>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

using namespace cute;

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  ColumnMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  ColumnMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  ColumnMajor>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //
  
  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

__global__ void gemm_nt_kernel_naive(float *C, const float *A, const float *B, int M, int N, int K) {
  int row = blockIdx.x, col = blockIdx.y;
  for (int i = 0; i < K; i++) {
    C[row * N + col] += A[row * K + i] * B[col * K + i];
  }
}

torch::Tensor gemm_main(torch::Tensor A, torch::Tensor B) {
  auto shape_A = A.sizes();
  auto shape_B = B.sizes();
  int M = shape_A[0];
  int K = shape_A[1];
  int N = shape_B[0];
  assert(shape_B[1] == K);

  auto C = torch::zeros({M, N});
  torch::Device device(torch::kCUDA);
  C = C.to(device);

  float *A_ptr = A.data_ptr<float>();
  float *B_ptr = B.data_ptr<float>();
  float *C_ptr = C.data_ptr<float>();

  dim3 grid_dim(M, N);
  dim3 block_dim(1);

#if 0
  gemm_nt_kernel_naive<<<grid_dim, block_dim>>>(C_ptr, A_ptr, B_ptr, M, N, K);
#elif 1
  //FIXME: CutlassSgemmNN is NN, but you are passing in arguments for NT => numerical error.
  CutlassSgemmNN(M, N, K, /*alpha*/1.0, A_ptr, /*ldA*/M, B_ptr, /*ldB*/N, /*beta*/0.0, C_ptr, /**/M);
#endif
  return C;
}
