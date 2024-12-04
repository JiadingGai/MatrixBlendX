import torch
from torch.utils.cpp_extension import load

blend_cpp = load(name="blend_cpp",
                 sources=[
                     "blend.cpp",
                     "sgemm_sm80.cu",
                     "matrix_blend.cu",
                 ],
                 extra_cuda_cflags=[
                     '-O2',
                     '--keep',
                     '--verbose',
                 ],
                 extra_include_paths=['./cutlass/include',
                                      './cutlass/tools/util/include',
                                      './nameof/include',
                                      ])

def check_allclose(x, y):
    if torch.allclose(x, y, rtol=0, atol=1e-02):
        print('PASS')
    else:
        diff = (x - y).abs()
        xxx = torch.argwhere(diff>=diff.max())
        print("C_gold@max_diff = ", y[xxx[0][0], xxx[0][1]])
        print("C@max_diff      = ", x[xxx[0][0], xxx[0][1]])
        print('FAILED')


M = 128
N = 256
K = 1024
X = torch.randn(M, N)
y = blend_cpp.d_sigmoid(X)

# compute gold
s = torch.sigmoid(X)
y_gold = (1 - s) * s

if torch.allclose(y, y_gold, rtol=0, atol=1e-05):
  print("PASS")
else:
  print("FAIL")

# test gemm_main
A = torch.randn(M, K).cuda()
B = torch.randn(N, K).cuda()
C_gold = A @ B.transpose(0, 1)
C = blend_cpp.gemm_main(A, B)
check_allclose(C, C_gold)

print("========== sgemm_sm80 ===========")
C = blend_cpp.sgemm_sm80_from_cute(A, B);
check_allclose(C, C_gold)

