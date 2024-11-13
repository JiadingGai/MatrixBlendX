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
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include <nameof.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace cute;

/// Reference: basic_gemm.cu from cutlass
/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
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

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
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

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

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

  // showcase how to use nameof for type info querying
  std::cout << "Type of Shape_A is " << nameof::nameof_type<decltype(shape_A)>()  << std::endl;

  auto C = torch::zeros({M, N});
  torch::Device device(torch::kCUDA);
  C = C.to(device);

  float *A_ptr = A.data_ptr<float>();
  float *B_ptr = B.data_ptr<float>();
  float *C_ptr = C.data_ptr<float>();

  dim3 grid_dim(M, N);
  dim3 block_dim(1);

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<float, float, float>{}, Layout<Shape<_16, _16, _1>>{});
  print(size(mmaC));

#if 1
  gemm_nt_kernel_naive<<<grid_dim, block_dim>>>(C_ptr, A_ptr, B_ptr, M, N, K);
#elif 0
  //FIXME: CutlassSgemmNN is NN, but you are passing in arguments for NT => numerical error.
  CutlassSgemmNN(M, N, K, /*alpha*/1.0, A_ptr, /*ldA*/M, B_ptr, /*ldB*/N, /*beta*/0.0, C_ptr, /**/M);
#endif
  return C;
}

//gemm_universal_f16t_s8n_f16t_mixed_input_tensor_op_f32_sm80.cu
torch::Tensor gemm_universal_f16t_s8n_f16t_mixed_input_tensor_op_f32_sm80(torch::Tensor A, torch::Tensor B, torch::Tensor C) {

  using ElementA = cutlass::half_t;
  using ElementB = int8_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using Gemm = cutlass::gemm::device::GemmUniversal<
    ElementA,
    cutlass::layout::RowMajor,
    ElementB,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,  // Stages
    8,  // AlignmentA
    16, // AlignmentB
    cutlass::arch::OpMultiplyAddMixedInputUpcast,
    cutlass::ComplexTransform::kNone,
    cutlass::ComplexTransform::kNone
  >;

  auto shape_A = A.sizes();
  auto shape_B = B.sizes();
  int K = shape_A[0];
  int M = shape_A[1];
  int N = shape_B[1];
  assert(shape_B[0] == K);

  int lda = K;
  int ldb = N;
  int ldc = N;
  int ldd = N;
  float alpha = 1.0f, beta = 0.0f;

  // ref: https://pytorch.org/cppdocs/notes/tensor_creation.html
  auto D_options =
    torch::TensorOptions()
      .dtype(torch::kFloat16)
      .layout(torch::kStrided)
      .device(torch::kCUDA, 1)
      .requires_grad(false);
  auto D = torch::zeros({M, N}, D_options);

  // ref:
  auto *A_ptr = A.data_ptr<at::Half>();
  auto *B_ptr = B.data_ptr<int8_t>();
  auto *C_ptr = C.data_ptr<at::Half>();
  auto *D_ptr = D.data_ptr<at::Half>();

  // Initialize the GEMM operator

  // ref: cutlass/examples/47_ampere_gemm_universal_streamk
  cutlass::gemm::GemmCoord problem_size({M, N, K});
  typename Gemm::Arguments arguments(
      cutlass::gemm::GemmUniversalMode::kGemm, //universal mode
      problem_size, // problem_size
      1, // batch count
      {
        ElementAccumulator(alpha),
        ElementAccumulator(beta)
      },
      A_ptr,
      B_ptr,
      C_ptr,
      D_ptr,
      problem_size.mk().product(), // batch_stride_A
      problem_size.nk().product(), // batch_stride_B
      problem_size.mn().product(), // batch_stride_C
      problem_size.mn().product(), // batch_stride_D
      lda, // stride_a
      ldb, // stride_b
      ldc, // stride_c
      ldd  // stride_d
   );

   Gemm gemm_op;
   auto status = gemm_op();
   assert(status == cutlass::Status::kSuccess);
}


torch::Tensor gemm_main_nn_column_major(torch::Tensor A, torch::Tensor B) {
  // GEMM_NN with A, B, C column major:
  // A: M x K column major
  // B: K x N column major
  // C: M x N column major
  auto shape_A = A.sizes();
  auto shape_B = B.sizes();
  int M = shape_A[0];
  int K = shape_A[1];
  int N = shape_B[1];
  assert(shape_B[0] == K);

  auto C = torch::zeros({M, N});
  torch::Device device(torch::kCUDA);
  C = C.to(device);

  /* float *A_ptr = A.data_ptr<float>(); */
  /* float *B_ptr = B.data_ptr<float>(); */
  /* float *C_ptr = C.data_ptr<float>(); */
  float *A_ptr = reinterpret_cast<float *>(A.data_ptr());
  float *B_ptr = reinterpret_cast<float *>(B.data_ptr());
  float *C_ptr = reinterpret_cast<float *>(C.data_ptr());

  dim3 grid_dim(M, N);
  dim3 block_dim(1);

  int gemm_selector = 1;
  if (gemm_selector == 0) {
    gemm_nt_kernel_naive<<<grid_dim, block_dim>>>(C_ptr, A_ptr, B_ptr, M, N, K);
  } else if (gemm_selector == 1) {
    //FIXME: CutlassSgemmNN is NN, but you are passing in arguments for NT => numerical error.
    CutlassSgemmNN(M, N, K, /*alpha*/1.0, A_ptr, /*ldA*/M, B_ptr, /*ldB*/K, /*beta*/0.0, C_ptr, /**/M);
  } else if (gemm_selector == 2) {
    ReferenceGemm(M, N, K, /*alpha*/1.0, A_ptr, /*ldA*/M, B_ptr, /*ldB*/K, /*beta*/0.0, C_ptr, /**/M);
  }

  return C;
}

template<typename elem_type>
__global__ void apply_mask_from_flash_attention() {
  constexpr int kNWarps = 4, kBlockM = 64, kBlockN = 128;
  using MMA_Atom_Arch = std::conditional_t<
    std::is_same_v<elem_type, cutlass::half_t>,
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
    MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
  >;
  using TiledMma = TiledMMA<
    MMA_Atom_Arch,
    Layout<Shape<Int<kNWarps>, _1, _1>>, // 4x1x1 or 8x1x1 thread group
    Tile<Int<16 * kNWarps>, _16, _16>>;

  TiledMma tiled_mma;
  Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); // (MMA=4, MMA_M, MMA_N)

  printf("\nacc_s:\n");
  print(acc_s);
  printf("\n\n");
  for (int z = 0; z < size<1>(acc_s.layout()); z++) {
    for (int w = 0; w < size<2>(acc_s.layout()); w++) {
      for (int x = 0; x < size<0,0>(acc_s.layout()); x++) {
        for (int y = 0; y < size<0,1>(acc_s.layout()); y++) {
          printf("%d ", acc_s.layout()(make_coord(make_coord(x, y), z, w)));
        }
      }
      printf("\n");
    }
  }
  printf("\n\n");
}

torch::Tensor flash_apply_mask(torch::Tensor A) {
  int M = 64, N = 128;
  auto C = torch::zeros({M, N});
  dim3 grid_dim(1);
  dim3 block_dim(1);
  apply_mask_from_flash_attention<cutlass::half_t><<<grid_dim, block_dim>>>();
  return C;
}

