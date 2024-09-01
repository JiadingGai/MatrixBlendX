import torch
from torch.utils.cpp_extension import load

blend_cpp = load(name="blend_cpp", sources=["blend.cpp"])


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
