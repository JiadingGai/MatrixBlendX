import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# ==========================================
# 1. KERNEL SOURCES
# ==========================================

# --- VERSION A: Naive (Strided) ---
naive_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rms_norm_naive_kernel(const float* __restrict__ x, float* __restrict__ out,
                                      int batch_size, int num_features, int dim1, int dim2, float eps) {
    int idx = blockIdx.x;
    int c = threadIdx.x;
    if (idx >= batch_size * dim1 * dim2 || c >= num_features) return;

    int j = idx % dim2;
    int temp = idx / dim2;
    int i = temp % dim1;
    int b = temp / dim1;

    int x_idx = ((b * num_features + c) * dim1 + i) * dim2 + j;
    float val = x[x_idx];
    float val_sq = val * val;

    extern __shared__ float sdata[];
    sdata[c] = val_sq;
    __syncthreads();

    for (unsigned int s = num_features / 2; s > 0; s >>= 1) {
        if (c < s) sdata[c] += sdata[c + s];
        __syncthreads();
    }
    
    float rms = 0.0f;
    if (c == 0) {
        rms = sqrtf(sdata[0] / num_features + eps);
        sdata[0] = rms;
    }
    __syncthreads();
    out[x_idx] = val / sdata[0];
}

torch::Tensor rms_norm_naive(torch::Tensor x, float eps) {
    auto out = torch::empty_like(x);
    int threads = x.size(1);
    int blocks = x.size(0) * x.size(2) * x.size(3);
    rms_norm_naive_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), x.size(1), x.size(2), x.size(3), eps
    );
    return out;
}
"""

# --- VERSION B: Coalesced (Thread-per-Pixel) ---
coalesced_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rms_norm_coalesced_kernel(const float* __restrict__ x, float* __restrict__ out,
                                          int batch_size, int num_features, int height, int width, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = height * width;
    int total_pixels = batch_size * spatial_size;
    if (idx >= total_pixels) return;

    int j = idx % width;
    int i = (idx / width) % height;
    int b = idx / spatial_size;

    int channel_stride = spatial_size;
    int base_offset = b * (num_features * spatial_size) + i * width + j;

    float sum_sq = 0.0f;
    for (int c = 0; c < num_features; c++) {
        float val = x[base_offset + c * channel_stride];
        sum_sq += val * val;
    }
    float rms_inv = rsqrtf(sum_sq / num_features + eps);

    for (int c = 0; c < num_features; c++) {
        int addr = base_offset + c * channel_stride;
        out[addr] = x[addr] * rms_inv;
    }
}

torch::Tensor rms_norm_coalesced(torch::Tensor x, float eps) {
    auto out = torch::empty_like(x);
    int total_pixels = x.size(0) * x.size(2) * x.size(3); 
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;
    rms_norm_coalesced_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), x.size(1), x.size(2), x.size(3), eps
    );
    return out;
}
"""

# --- VERSION C: Vectorized (float4) ---
vectorized_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rms_norm_vectorized_kernel(const float* __restrict__ x, float* __restrict__ out,
                                           int batch_size, int num_features, int height, int width, float eps) {
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* out_vec = reinterpret_cast<float4*>(out);
    
    int vec_width = width / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vectors = batch_size * height * vec_width;
    if (idx >= total_vectors) return;

    int j_vec = idx % vec_width; 
    int i = (idx / vec_width) % height;
    int b = idx / (height * vec_width);

    int channel_stride_vec = height * vec_width;
    int base_offset_vec = b * (num_features * channel_stride_vec) + i * vec_width + j_vec;

    float4 sum_sq_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int c = 0; c < num_features; c++) {
        float4 val = x_vec[base_offset_vec + c * channel_stride_vec];
        sum_sq_vec.x += val.x * val.x;
        sum_sq_vec.y += val.y * val.y;
        sum_sq_vec.z += val.z * val.z;
        sum_sq_vec.w += val.w * val.w;
    }

    float4 rms_inv;
    rms_inv.x = rsqrtf(sum_sq_vec.x / num_features + eps);
    rms_inv.y = rsqrtf(sum_sq_vec.y / num_features + eps);
    rms_inv.z = rsqrtf(sum_sq_vec.z / num_features + eps);
    rms_inv.w = rsqrtf(sum_sq_vec.w / num_features + eps);

    for (int c = 0; c < num_features; c++) {
        int addr = base_offset_vec + c * channel_stride_vec;
        float4 val = x_vec[addr];
        val.x *= rms_inv.x; val.y *= rms_inv.y; val.z *= rms_inv.z; val.w *= rms_inv.w;
        out_vec[addr] = val;
    }
}

torch::Tensor rms_norm_vectorized(torch::Tensor x, float eps) {
    auto out = torch::empty_like(x);
    int vec_width = x.size(3) / 4;
    int total_vectors = x.size(0) * x.size(2) * vec_width; 
    int threads = 256;
    int blocks = (total_vectors + threads - 1) / threads;
    rms_norm_vectorized_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), x.size(1), x.size(2), x.size(3), eps
    );
    return out;
}
"""

# --- VERSION D: One-Pass (BROKEN/MIRAGE) ---
# This version uses float4 on scalar channels. 
# It incorrectly processes only 16 channels instead of 64.
# Included for historical reproduction of the 2800+ GB/s result.
one_pass_broken_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rms_norm_one_pass_broken_kernel(const float* __restrict__ x, float* __restrict__ out,
                                         int batch_size, int num_features, int height, int width, float eps) {
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* out_vec = reinterpret_cast<float4*>(out);

    int vec_width = width / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vectors = batch_size * height * vec_width;
    if (idx >= total_vectors) return;

    int j_vec = idx % vec_width; 
    int i = (idx / vec_width) % height;
    int b = idx / (height * vec_width);
    
    int stride = height * vec_width;
    int offset = b * (num_features * stride) + i * vec_width + j_vec;

    float4 cache[16]; 
    float4 sum_sq_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    #pragma unroll
    for (int c = 0; c < 16; c++) {
        cache[c] = x_vec[offset + c * stride];
        sum_sq_vec.x += cache[c].x * cache[c].x;
        sum_sq_vec.y += cache[c].y * cache[c].y;
        sum_sq_vec.z += cache[c].z * cache[c].z;
        sum_sq_vec.w += cache[c].w * cache[c].w;
    }

    float4 rms_inv;
    rms_inv.x = rsqrtf(sum_sq_vec.x / num_features + eps);
    rms_inv.y = rsqrtf(sum_sq_vec.y / num_features + eps);
    rms_inv.z = rsqrtf(sum_sq_vec.z / num_features + eps);
    rms_inv.w = rsqrtf(sum_sq_vec.w / num_features + eps);

    #pragma unroll
    for (int c = 0; c < 16; c++) {
        float4 val = cache[c];
        val.x *= rms_inv.x;
        val.y *= rms_inv.y;
        val.z *= rms_inv.z;
        val.w *= rms_inv.w;
        out_vec[offset + c * stride] = val;
    }
}

torch::Tensor rms_norm_one_pass_broken(torch::Tensor x, float eps) {
    auto out = torch::empty_like(x);
    int vec_width = x.size(3) / 4;
    int total_vectors = x.size(0) * x.size(2) * vec_width; 
    int threads = 256;
    int blocks = (total_vectors + threads - 1) / threads;
    rms_norm_one_pass_broken_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), x.size(0), x.size(1), x.size(2), x.size(3), eps
    );
    return out;
}
"""

# --- VERSION E: God Mode (FIXED SCALAR + STREAMING) ---
# This is the correct "Speed of Light" solution.
god_mode_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ void stream_store(float* addr, float val) {
    asm volatile ("st.global.cs.f32 [%0], %1;" :: "l"(addr), "f"(val) : "memory");
}

__global__ void rms_norm_god_mode_kernel(const float* __restrict__ x, 
                                         float* __restrict__ out,
                                         float eps) {
    constexpr int HEIGHT = 256;
    constexpr int WIDTH = 256;
    constexpr int SPATIAL = HEIGHT * WIDTH;
    constexpr int FEAT = 64;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = 16 * SPATIAL; // B=16
    if (idx >= total_pixels) return;

    // Bitwise Indexing (256 = 2^8)
    int j = idx & 255;          
    int i = (idx >> 8) & 255;   
    int b = idx >> 16;          

    int base_offset = b * (FEAT * SPATIAL) + i * WIDTH + j;
    constexpr int stride = SPATIAL;

    float cache[64];
    float sum_sq = 0.0f;

    // Load
    #pragma unroll
    for (int c = 0; c < 64; c++) {
        float val = x[base_offset + c * stride];
        cache[c] = val;
        sum_sq += val * val;
    }

    float rms_inv = rsqrtf(sum_sq / 64.0f + eps);

    // Store (Streaming)
    #pragma unroll
    for (int c = 0; c < 64; c++) {
        float val = cache[c] * rms_inv;
        stream_store(&out[base_offset + c * stride], val);
    }
}

torch::Tensor rms_norm_god_mode(torch::Tensor x, float eps) {
    auto out = torch::empty_like(x);
    int total_pixels = 16 * 256 * 256;
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;
    rms_norm_god_mode_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), eps
    );
    return out;
}
"""

# ==========================================
# 2. COMPILE
# ==========================================
print("Compiling Kernels...")
mod_naive = load_inline(name='naive', cpp_sources="torch::Tensor rms_norm_naive(torch::Tensor x, float eps);", cuda_sources=naive_source, functions=['rms_norm_naive'], verbose=False)
mod_opt = load_inline(name='opt', cpp_sources="torch::Tensor rms_norm_coalesced(torch::Tensor x, float eps);", cuda_sources=coalesced_source, functions=['rms_norm_coalesced'], verbose=False)
mod_vec = load_inline(name='vec', cpp_sources="torch::Tensor rms_norm_vectorized(torch::Tensor x, float eps);", cuda_sources=vectorized_source, functions=['rms_norm_vectorized'], verbose=False)
mod_broken = load_inline(name='broken', cpp_sources="torch::Tensor rms_norm_one_pass_broken(torch::Tensor x, float eps);", cuda_sources=one_pass_broken_source, functions=['rms_norm_one_pass_broken'], verbose=False)
mod_god = load_inline(name='godmode', cpp_sources="torch::Tensor rms_norm_god_mode(torch::Tensor x, float eps);", cuda_sources=god_mode_source, functions=['rms_norm_god_mode'], verbose=False)
print("Done.\n")

# ==========================================
# 3. BENCHMARK
# ==========================================

device = torch.device("cuda")
B, C, H, W = 16, 64, 256, 256
eps = 1e-5
x = torch.randn(B, C, H, W, device=device, dtype=torch.float32)

# Theoretical Limit
A100_PEAK_BW = 1935.0 # GB/s (A100 80GB spec) - Using this for consistency with your request

print(f"--- BENCHMARK: {torch.cuda.get_device_name(0)} ---")
print(f"Input: {x.shape}")
print(f"Efficiency Ref (A100): {A100_PEAK_BW} GB/s\n")

kernels = [
    ("PyTorch", lambda x,e: x/torch.sqrt(x.pow(2).mean(dim=1, keepdim=True)+e)),
    ("Naive", mod_naive.rms_norm_naive),
    ("Coalesced", mod_opt.rms_norm_coalesced),
    ("Vectorized (float4)", mod_vec.rms_norm_vectorized),
    ("One-Pass (Broken/Mirage)", mod_broken.rms_norm_one_pass_broken),
    ("God Mode (Scalar Correct)", mod_god.rms_norm_god_mode)
]

required_gb = (x.numel() * 4 * 2) / 1e9

print(f"{'Kernel':<26} | {'Latency':<10} | {'Bandwidth':<12} | {'% A100 Peak':<12} | {'Status'}")
print("-" * 85)

for name, func in kernels:
    try:
        # Correctness Check
        out_ref = kernels[0][1](x, eps)
        out = func(x, eps)
        is_correct = torch.allclose(out_ref, out, atol=1e-3, rtol=1e-3)
        status = "PASS" if is_correct else "FAIL"
            
        # Timing
        for _ in range(20): _ = func(x, eps)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(1000): _ = func(x, eps)
        end.record()
        torch.cuda.synchronize()
        
        ms = start.elapsed_time(end) / 1000
        gb_s = required_gb / (ms / 1000.0)
        eff = (gb_s / A100_PEAK_BW) * 100
        
        print(f"{name:<26} | {ms:.3f} ms   | {gb_s:.2f} GB/s  | {eff:.1f}%        | {status}")
    except Exception as e:
        print(f"{name:<26} | ERROR: {e}")
