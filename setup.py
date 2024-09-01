from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='blend_cpp',
            ext_modules=[cpp_extension.CppExtension('blend_cpp', ['blend.cpp'])],
            cmdclass={'build_ext': cpp_extension.BuildExtension})
