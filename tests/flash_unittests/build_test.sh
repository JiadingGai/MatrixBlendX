# TODO: use cmake to drive all the tests
# try cuda 11.7 if has any problem compiling
/usr/local/cuda-11.7/bin/nvcc tiled_copy.cu -I../../cutlass/include -I../../cutlass/tools/util/include --expt-relaxed-constexpr
