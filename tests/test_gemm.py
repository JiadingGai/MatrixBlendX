import math
import pytest
import numpy as np
import torch
from torch.utils.cpp_extension import load

# FIXME: do not use relative path for cutlass dir
# relative path is set to ./cutlass instead of ../cutlass because
# this test is usually invoked from the top level dir:
# pytest -q -s tests/test_gemm.py
blend_cpp = load(name="blend_cpp",
                 sources=["blend.cpp", "matrix_blend.cu"],
                 extra_cuda_cflags=['-O2'],
                 extra_include_paths=['./cutlass/include',
                                      './cutlass/tools/util/include',
                                      './nameof/include',
                                      ])

class TestCutlassGemm:
    def test_0(self):
      # A, B are row major (default in torch tensor)
      A = torch.from_numpy(np.array([[1,1,1,1], [2,2,2,2]], dtype=np.float32)).cuda()
      B = torch.from_numpy(np.array([[1,1,1,1], [3,3,3,3]], dtype=np.float32)).cuda()
      # print(A.stride())
      # print(B.stride())

      # Turn A and B into column major
      A = A.t().contiguous().t()
      BT = B.transpose(0,1).t().contiguous().t()

      C = blend_cpp.gemm_main_nn_column_major(A, BT)
      C_gold = A @ B.transpose(0, 1)
      # C_gold is a transpose of C due to C is in column major and C_gold is in row major
      assert(torch.allclose(C.transpose(0,1), C_gold, rtol=0.0, atol=0.0))

    def test_gemm_main_0(self):
      M = 128
      N = 256
      K = 1024

      A = torch.randn(M, K).cuda()
      B = torch.randn(N, K).cuda()
      C_gold = A @ B.transpose(0, 1)
      C = blend_cpp.gemm_main(A, B)
      print(C[0, :16])
      if torch.allclose(C, C_gold, rtol=0, atol=1e-02):
          print('PASS')
      else:
          diff = (C - C_gold).abs()
          xxx = torch.argwhere(diff>=diff.max())
          print("C_gold@max_diff = ", C_gold[xxx[0][0], xxx[0][1]])
          print("C@max_diff      = ", C[xxx[0][0], xxx[0][1]])
          print('FAILED')

class TestCute:
    def test_0(self):
      M = 64
      N = 128
      X = torch.randn(M, N).cuda()
      Y = blend_cpp.flash_apply_mask(X)
      pass
