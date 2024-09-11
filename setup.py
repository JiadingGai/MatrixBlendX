from setuptools import setup, Extension
from torch.utils import cpp_extension
from pathlib import Path
import os

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

setup(name='blend_cpp',
      ext_modules=[
          cpp_extension.CUDAExtension(
              name='blend_cpp',
              sources=["blend.cpp", "matrix_blend.cu"],
              extra_cuda_cflags=['-O2'],
              include_dirs=[Path(this_dir) / "cutlass" / "include", Path(this_dir) / "cutlass" / "tools" / "util" / "include"])
      ],
      cmdclass={
          'build_ext': cpp_extension.BuildExtension
      })
