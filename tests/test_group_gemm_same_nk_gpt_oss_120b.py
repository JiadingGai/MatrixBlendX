# test2.py
# Benchmark + correctness for VeOmni group_gemm_same_nk using GPT-OSS-120B MoE dims.
#
# Option B: for ACCUMULATE_TO_C correctness check, pass fp32 `c_init`
# so Triton `tl.dot(a, b, c)` sees an fp32 accumulator even if veomni_patch
# isn't installed.
#
# IMPORTANT FIX: group_gemm_same_nk writes into `c` in-place, so we must clone
# c_init BEFORE calling the kernel, and use the clone for the reference.

import os
from dataclasses import dataclass

import torch
import triton

from veomni.ops.group_gemm.kernel.group_gemm import group_gemm_same_nk


@dataclass(frozen=True)
class GPTOSS120BMoESpec:
    G: int = 128
    K: int = 2880
    N: int = 2880
    TOPK: int = 4


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    return default if v == "" else int(v)


def _bytes_per_elem(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype: {dtype}")


def _estimate_bytes(G: int, total_M: int, K: int, N: int, dtype: torch.dtype) -> int:
    bpe = _bytes_per_elem(dtype)
    return int((G * K * N + total_M * K + total_M * N) * bpe * 1.15)


def _make_cumsum_M(total_M: int, G: int, device: torch.device) -> tuple[torch.Tensor, int]:
    """
    Create a random distribution of routed rows across experts.
    Assumes A is already packed by expert in expert-id order [0..G-1],
    with boundaries given by cumsum_M.
    """
    idx = torch.randint(low=0, high=G, size=(total_M,), device="cpu", dtype=torch.int64)
    counts = torch.bincount(idx, minlength=G).to(torch.int32)
    assert int(counts.sum().item()) == total_M
    cumsum = torch.cumsum(counts, dim=0)  # [G], CPU int32
    max_M = int(counts.max().item())
    return cumsum.to(device=device, non_blocking=True), max_M


@torch.no_grad()
def _ref_grouped_gemm_same_nk(
    a: torch.Tensor,          # [total_M, K] packed by expert
    b: torch.Tensor,          # [G, K, N]
    cumsum_M: torch.Tensor,   # [G] on GPU
    *,
    accumulate_to_c: bool,
    c_init: torch.Tensor | None,
    fp32_ref: bool,
) -> torch.Tensor:
    """
    Reference: for each expert g, compute a_g @ b_g and write to corresponding slice.
    If accumulate_to_c, adds c_init slice.
    """
    assert a.is_cuda and b.is_cuda and cumsum_M.is_cuda
    G, K, N = b.shape
    total_M = a.shape[0]

    if accumulate_to_c:
        assert c_init is not None
        out = torch.empty((total_M, N), device=a.device, dtype=c_init.dtype)
    else:
        out = torch.empty((total_M, N), device=a.device, dtype=a.dtype)

    cumsum_cpu = cumsum_M.to("cpu", dtype=torch.int64)
    starts = torch.empty_like(cumsum_cpu)
    starts[0] = 0
    starts[1:] = cumsum_cpu[:-1]

    for g in range(G):
        s = int(starts[g].item())
        e = int(cumsum_cpu[g].item())
        if e <= s:
            continue

        a_g = a[s:e]    # [M_g, K]
        b_g = b[g]      # [K, N]

        if fp32_ref:
            y = a_g.float() @ b_g.float()   # fp32
        else:
            y = (a_g @ b_g).float()         # run in dtype then cast to fp32 for stable compare

        if accumulate_to_c:
            y = y + c_init[s:e].float()
            out[s:e] = y.to(dtype=c_init.dtype)
        else:
            out[s:e] = y.to(dtype=a.dtype)

    return out


def _do_bench_ms(fn, warmup: int = 50, rep: int = 200) -> float:
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def _sync():
    torch.cuda.synchronize()


def test_group_gemm_same_nk_gpt_oss_120b_correctness():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    spec = GPTOSS120BMoESpec()
    device = torch.device("cuda")

    tokens = _env_int("GPTOSS_TEST_TOKENS", 512)
    total_M = tokens * spec.TOPK

    dtype_name = os.environ.get("GPTOSS_DTYPE", "bf16").lower()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16

    need = _estimate_bytes(spec.G, total_M, spec.K, spec.N, dtype)
    free, _total = torch.cuda.mem_get_info()
    if need > free:
        print(f"[SKIP] Not enough GPU memory: need~{need/1e9:.2f}GB, free~{free/1e9:.2f}GB")
        return

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    cumsum_M, max_M = _make_cumsum_M(total_M=total_M, G=spec.G, device=device)

    a = torch.randn((total_M, spec.K), device=device, dtype=dtype)
    b = torch.randn((spec.G, spec.K, spec.N), device=device, dtype=dtype)

    # ---- Non-accumulating path ----
    _sync()
    out = group_gemm_same_nk(
        a=a,
        b=b,
        cumsum_M=cumsum_M,
        max_M=max_M,
        transpose_a=False,
        transpose_b=False,
        activation=None,
        save_activation=False,
        c=None,
    )
    _sync()

    ref = _ref_grouped_gemm_same_nk(
        a, b, cumsum_M,
        accumulate_to_c=False,
        c_init=None,
        fp32_ref=True,
    )

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)

    # ---- Accumulating path (Option B + in-place safe) ----
    # Use fp32 c to satisfy Triton's tl.dot accumulator type checks.
    c_init = torch.randn((total_M, spec.N), device=device, dtype=torch.float32)
    c_init_ref = c_init.clone()  # <<< critical: preserve original for reference

    _sync()
    out2 = group_gemm_same_nk(
        a=a,
        b=b,
        cumsum_M=cumsum_M,
        max_M=max_M,
        transpose_a=False,
        transpose_b=False,
        activation=None,
        save_activation=False,
        c=c_init,  # in-place updated, fp32
    )
    _sync()

    ref2 = _ref_grouped_gemm_same_nk(
        a, b, cumsum_M,
        accumulate_to_c=True,
        c_init=c_init_ref,  # <<< use the original values
        fp32_ref=True,
    )

    assert out2.dtype == torch.float32, out2.dtype
    torch.testing.assert_close(out2, ref2, rtol=1e-2, atol=5e-2)

    print("[OK] correctness passed (including accumulate-to-C with fp32 c_init)")


def test_group_gemm_same_nk_gpt_oss_120b_benchmark():
    """
    Run:
      GPTOSS_BENCH_TOKENS=2048 GPTOSS_DTYPE=bf16 python test2.py
    """
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    spec = GPTOSS120BMoESpec()
    device = torch.device("cuda")

    tokens = _env_int("GPTOSS_BENCH_TOKENS", 2048)
    total_M = tokens * spec.TOPK

    dtype_name = os.environ.get("GPTOSS_DTYPE", "bf16").lower()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16

    need = _estimate_bytes(spec.G, total_M, spec.K, spec.N, dtype)
    free, _total = torch.cuda.mem_get_info()
    if need > free:
        print(f"[SKIP] Not enough GPU memory: need~{need/1e9:.2f}GB, free~{free/1e9:.2f}GB")
        return

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    cumsum_M, max_M = _make_cumsum_M(total_M=total_M, G=spec.G, device=device)
    a = torch.randn((total_M, spec.K), device=device, dtype=dtype)
    b = torch.randn((spec.G, spec.K, spec.N), device=device, dtype=dtype)

    # Warmups
    for _ in range(10):
        _ = group_gemm_same_nk(a, b, cumsum_M, max_M, activation=None, c=None)
    _sync()

    ms_kernel = _do_bench_ms(lambda: group_gemm_same_nk(a, b, cumsum_M, max_M, activation=None, c=None))
    _sync()

    @torch.no_grad()
    def ref_loop_native():
        G, K, N = b.shape
        out = torch.empty((a.shape[0], N), device=device, dtype=dtype)

        cumsum_cpu = cumsum_M.to("cpu", dtype=torch.int64)
        starts = torch.empty_like(cumsum_cpu)
        starts[0] = 0
        starts[1:] = cumsum_cpu[:-1]

        for g in range(G):
            s = int(starts[g].item())
            e = int(cumsum_cpu[g].item())
            if e <= s:
                continue
            out[s:e] = a[s:e] @ b[g]
        return out

    for _ in range(3):
        _ = ref_loop_native()
    _sync()

    ms_ref = _do_bench_ms(ref_loop_native, warmup=10, rep=50)
    _sync()

    flops = 2.0 * float(total_M) * float(spec.K) * float(spec.N)
    gflops_kernel = flops / (ms_kernel * 1e-3) / 1e9
    gflops_ref = flops / (ms_ref * 1e-3) / 1e9
    speedup = ms_ref / ms_kernel

    print(
        f"\n[gpt-oss-120b dims] G={spec.G} K={spec.K} N={spec.N} TOPK={spec.TOPK} "
        f"tokens={tokens} total_M={total_M} max_M={max_M} dtype={dtype}\n"
        f"  group_gemm_same_nk: {ms_kernel:.3f} ms  ({gflops_kernel:.1f} GFLOP/s)\n"
        f"  torch per-expert   : {ms_ref:.3f} ms  ({gflops_ref:.1f} GFLOP/s)\n"
        f"  speedup: {speedup:.2f}x\n"
    )


if __name__ == "__main__":
    test_group_gemm_same_nk_gpt_oss_120b_correctness()
    test_group_gemm_same_nk_gpt_oss_120b_benchmark()

