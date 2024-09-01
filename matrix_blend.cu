#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <ATen/ATen.h>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;

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

  gemm_nt_kernel_naive<<<grid_dim, block_dim>>>(C_ptr, A_ptr, B_ptr, M, N, K);
  return C;
}
