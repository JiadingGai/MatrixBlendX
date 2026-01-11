# test_sweep_group_gemm.py
# Adds a sweep over tokens (e.g., 256 -> 16384) and prints:
#   (tokens, total_M, max_M, ms, TFLOP/s, est TB/s, BW%peak, baseline ms, speedup)
#
# IMPORTANT:
# - Uses explicit warmup iterations BEFORE timing measurement (as requested).
# - Keeps Option B for accumulate correctness (fp32 c_init) + in-place safety.
#
# Run:
#   python test_sweep_group_gemm.py
# Or:
#   GPTOSS_SWEEP_MAX_TOKENS=16384 GPTOSS_WARMUP_ITERS=10 GPTOSS_REP=200 python test_sweep_group_gemm.py
# Optional:
#   GPTOSS_DTYPE=bf16|fp16
#   GPTOSS_PEAK_TBPS=3.35        (override peak HBM BW)
#   GPTOSS_BASELINE=0|1          (disable/enable per-expert torch baseline in sweep)
#   GPTOSS_SWEEP_TOKENS="256,512,1024,2048,4096,8192,16384"

import os
from dataclasses import dataclass
from typing import List, Optional

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


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    return default if v == "" else float(v)


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "")
    return default if v == "" else v


def _bytes_per_elem(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype: {dtype}")


def _estimate_bytes_streamed(G: int, total_M: int, K: int, N: int, dtype_a: torch.dtype, dtype_b: torch.dtype) -> int:
    """
    Rough "one pass" estimate:
      read A + read B + write C
    """
    a_bpe = _bytes_per_elem(dtype_a)
    b_bpe = _bytes_per_elem(dtype_b)
    # A: [total_M, K], B: [G, K, N], C: [total_M, N]
    return (total_M * K * a_bpe) + (G * K * N * b_bpe) + (total_M * N * a_bpe)


def _estimate_bytes_total(G: int, total_M: int, K: int, N: int, dtype: torch.dtype) -> int:
    # a=b=c dtype in our test; add a small overhead factor.
    return int(_estimate_bytes_streamed(G, total_M, K, N, dtype, dtype) * 1.15)


def _make_cumsum_M(total_M: int, G: int, device: torch.device) -> tuple[torch.Tensor, int]:
    """
    Make random expert counts that sum to total_M.
    The contract with the kernel is: A is packed by expert in order [0..G-1],
    with boundaries given by cumsum_M.
    For benchmarking, it's fine that values are random; we just need a valid partition.
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
    c_init: Optional[torch.Tensor],
    fp32_ref: bool,
) -> torch.Tensor:
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

        a_g = a[s:e]
        b_g = b[g]

        if fp32_ref:
            y = a_g.float() @ b_g.float()
        else:
            y = (a_g @ b_g).float()

        if accumulate_to_c:
            y = y + c_init[s:e].float()
            out[s:e] = y.to(dtype=c_init.dtype)
        else:
            out[s:e] = y.to(dtype=a.dtype)

    return out


def _sync():
    torch.cuda.synchronize()


def _parse_sweep_tokens(spec: GPTOSS120BMoESpec) -> List[int]:
    s = _env_str("GPTOSS_SWEEP_TOKENS", "")
    if s.strip():
        toks = [int(x.strip()) for x in s.split(",") if x.strip()]
        return toks

    # Default: powers of two from 256 up to env max
    max_toks = _env_int("GPTOSS_SWEEP_MAX_TOKENS", 16384)
    toks = []
    t = 256
    while t <= max_toks:
        toks.append(t)
        t *= 2
    return toks


def _infer_peak_tbps(device_name: str) -> Optional[float]:
    # Heuristic default. Override with GPTOSS_PEAK_TBPS to be exact for your SKU.
    # For H100 SXM (HBM3), common spec is 3.35 TB/s.
    # For other GPUs/SKUs, return None (we'll just omit BW%peak).
    if "H100" in device_name.upper():
        return 3.35
    return None


def test_correctness_once_option_b():
    """
    One correctness run, including:
      - non-accumulating path
      - accumulate-to-C path with fp32 c_init (Option B) and in-place safe reference
    """
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    spec = GPTOSS120BMoESpec()
    device = torch.device("cuda")

    tokens = _env_int("GPTOSS_TEST_TOKENS", 512)
    total_M = tokens * spec.TOPK

    dtype_name = _env_str("GPTOSS_DTYPE", "bf16").lower()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16

    need = _estimate_bytes_total(spec.G, total_M, spec.K, spec.N, dtype)
    free, _total = torch.cuda.mem_get_info()
    if need > free:
        print(f"[SKIP] Not enough GPU memory: need~{need/1e9:.2f}GB, free~{free/1e9:.2f}GB")
        return

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    cumsum_M, max_M = _make_cumsum_M(total_M=total_M, G=spec.G, device=device)
    a = torch.randn((total_M, spec.K), device=device, dtype=dtype)
    b = torch.randn((spec.G, spec.K, spec.N), device=device, dtype=dtype)

    # Non-accumulate
    _sync()
    out = group_gemm_same_nk(a=a, b=b, cumsum_M=cumsum_M, max_M=max_M, activation=None, c=None)
    _sync()

    ref = _ref_grouped_gemm_same_nk(a, b, cumsum_M, accumulate_to_c=False, c_init=None, fp32_ref=True)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-2)

    # Accumulate: Option B (fp32 c_init) + in-place safe
    c_init = torch.randn((total_M, spec.N), device=device, dtype=torch.float32)
    c_init_ref = c_init.clone()

    _sync()
    out2 = group_gemm_same_nk(a=a, b=b, cumsum_M=cumsum_M, max_M=max_M, activation=None, c=c_init)
    _sync()

    ref2 = _ref_grouped_gemm_same_nk(a, b, cumsum_M, accumulate_to_c=True, c_init=c_init_ref, fp32_ref=True)
    assert out2.dtype == torch.float32
    torch.testing.assert_close(out2, ref2, rtol=1e-2, atol=5e-2)

    print("[OK] correctness passed (including accumulate-to-C with fp32 c_init)")


def bench_sweep_tokens():
    """
    Sweep tokens and benchmark:
      - group_gemm_same_nk kernel (c=None)
      - optional per-expert torch baseline
    Includes explicit warmup iterations before timing.
    """
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return

    spec = GPTOSS120BMoESpec()
    device = torch.device("cuda")

    dtype_name = _env_str("GPTOSS_DTYPE", "bf16").lower()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16

    warmup_iters = _env_int("GPTOSS_WARMUP_ITERS", 10)
    rep = _env_int("GPTOSS_REP", 200)

    baseline_enabled = _env_int("GPTOSS_BASELINE", 1) != 0
    baseline_rep = _env_int("GPTOSS_BASELINE_REP", 50)

    toks_list = _parse_sweep_tokens(spec)

    device_name = torch.cuda.get_device_name(0)
    peak_tbps = _env_float("GPTOSS_PEAK_TBPS", -1.0)
    if peak_tbps <= 0:
        peak_tbps = _infer_peak_tbps(device_name) or 0.0  # 0 means unknown

    print(f"\n[device] {device_name} | triton={triton.__version__} | dtype={dtype}")
    if peak_tbps > 0:
        print(f"[info] Using peak HBM BW = {peak_tbps:.2f} TB/s (override with GPTOSS_PEAK_TBPS)")
    else:
        print("[info] Peak HBM BW unknown (set GPTOSS_PEAK_TBPS to compute BW%peak)")

    # Allocate weights once (dominant memory).
    # Note: this is huge: G*K*N*2 bytes ~ 2.12GB for bf16.
    need_b = spec.G * spec.K * spec.N * _bytes_per_elem(dtype)
    free, _total = torch.cuda.mem_get_info()
    if int(need_b * 1.05) > free:
        print(f"[SKIP] Not enough GPU memory for weights: need~{need_b/1e9:.2f}GB, free~{free/1e9:.2f}GB")
        return

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    b = torch.randn((spec.G, spec.K, spec.N), device=device, dtype=dtype)

    # Allocate max A once and slice (avoids realloc).
    max_tokens = max(toks_list)
    max_total_M = max_tokens * spec.TOPK
    need_total = _estimate_bytes_total(spec.G, max_total_M, spec.K, spec.N, dtype)
    free, _total = torch.cuda.mem_get_info()
    if need_total > free:
        print(f"[SKIP] Not enough GPU memory for sweep max workload: need~{need_total/1e9:.2f}GB, free~{free/1e9:.2f}GB")
        return

    a_big = torch.randn((max_total_M, spec.K), device=device, dtype=dtype)

    # Helper baseline (per-expert loop). Keep it as close as possible to the same contract.
    @torch.no_grad()
    def ref_loop_native(a: torch.Tensor, cumsum_M: torch.Tensor) -> torch.Tensor:
        out = torch.empty((a.shape[0], spec.N), device=device, dtype=dtype)

        cumsum_cpu = cumsum_M.to("cpu", dtype=torch.int64)
        starts = torch.empty_like(cumsum_cpu)
        starts[0] = 0
        starts[1:] = cumsum_cpu[:-1]

        for g in range(spec.G):
            s = int(starts[g].item())
            e = int(cumsum_cpu[g].item())
            if e <= s:
                continue
            out[s:e] = a[s:e] @ b[g]
        return out

    # Print header
    header = (
        "\n"
        "tokens  total_M  max_M   ms_kernel   TFLOP/s   est_TB/s  BW%peak"
        + ("   ms_torch   speedup" if baseline_enabled else "")
    )
    print(header)
    print("-" * len(header))

    # Compile once (first call) with a representative size to remove JIT cost from sweep
    rep_tokens = toks_list[len(toks_list) // 2]
    rep_total_M = rep_tokens * spec.TOPK
    cumsum_rep, max_M_rep = _make_cumsum_M(rep_total_M, spec.G, device)
    _ = group_gemm_same_nk(a_big[:rep_total_M], b, cumsum_rep, max_M_rep, activation=None, c=None)
    _sync()

    for tokens in toks_list:
        total_M = tokens * spec.TOPK
        a = a_big[:total_M]

        cumsum_M, max_M = _make_cumsum_M(total_M, spec.G, device)

        # ---- Explicit warmup (requested) ----
        for _ in range(warmup_iters):
            _ = group_gemm_same_nk(a, b, cumsum_M, max_M, activation=None, c=None)
        _sync()

        # ---- Timed measurement (no additional warmup inside do_bench) ----
        ms_kernel = float(triton.testing.do_bench(
            lambda: group_gemm_same_nk(a, b, cumsum_M, max_M, activation=None, c=None),
            warmup=0,
            rep=rep,
        ))
        _sync()

        flops = 2.0 * float(total_M) * float(spec.K) * float(spec.N)
        tflops = flops / (ms_kernel * 1e-3) / 1e12

        est_bytes = float(_estimate_bytes_streamed(spec.G, total_M, spec.K, spec.N, dtype, dtype))
        tbps = est_bytes / (ms_kernel * 1e-3) / 1e12  # TB/s

        bw_pct = (tbps / peak_tbps * 100.0) if peak_tbps > 0 else 0.0

        if baseline_enabled:
            # Warm baseline a bit (fewer iters so it doesn't dominate runtime)
            for _ in range(max(1, warmup_iters // 2)):
                _ = ref_loop_native(a, cumsum_M)
            _sync()

            ms_torch = float(triton.testing.do_bench(
                lambda: ref_loop_native(a, cumsum_M),
                warmup=0,
                rep=baseline_rep,
            ))
            _sync()

            speedup = ms_torch / ms_kernel
            print(
                f"{tokens:5d}  {total_M:7d}  {max_M:5d}  "
                f"{ms_kernel:9.3f}  {tflops:7.2f}  {tbps:8.2f}  "
                f"{(bw_pct if peak_tbps > 0 else 0.0):6.1f}  "
                f"{ms_torch:8.3f}  {speedup:7.2f}x"
            )
        else:
            print(
                f"{tokens:5d}  {total_M:7d}  {max_M:5d}  "
                f"{ms_kernel:9.3f}  {tflops:7.2f}  {tbps:8.2f}  "
                f"{(bw_pct if peak_tbps > 0 else 0.0):6.1f}"
            )


if __name__ == "__main__":
    # Keep your one-shot correctness + single-point benchmark if you want,
    # but now you also get a sweep.
    test_correctness_once_option_b()
    bench_sweep_tokens()

