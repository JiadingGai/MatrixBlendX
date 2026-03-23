import numpy as np

M = N = K = 256
bm, bn, bk = 256, 256, 32
wm, wn, wk = 128, 64, 8
rng = np.random.default_rng(0)
A = rng.standard_normal((M, K)).astype(np.float16)
B = rng.standard_normal((N, K)).astype(np.float16)

def load_stage(mo, no, ko):
    return A[mo:mo + bm, ko:ko + bk].astype(np.float32), B[no:no + bn, ko:ko + bk].astype(np.float32)

def mma_block(Cblk, As, Bs):
    for wo_m in range(0, bm, wm):
        for wo_n in range(0, bn, wn):
            Cfrag = np.zeros((wm, wn), np.float32)
            for kk in range(0, bk, wk):
                Cfrag += As[wo_m:wo_m + wm, kk:kk + wk] @ Bs[wo_n:wo_n + wn, kk:kk + wk].T
            Cblk[wo_m:wo_m + wm, wo_n:wo_n + wn] += Cfrag

C = np.zeros((M, N), np.float32)
for mo in range(0, M, bm):
    for no in range(0, N, bn):
        Cblk = np.zeros((bm, bn), np.float32)
        As, Bs = load_stage(mo, no, 0)
        for ko in range(0, K, bk):
            nxt = load_stage(mo, no, ko + bk) if ko + bk < K else (None, None)
            mma_block(Cblk, As, Bs)
            As, Bs = nxt
        C[mo:mo + bm, no:no + bn] = Cblk

ref = A.astype(np.float32) @ B.astype(np.float32).T
print("v2_prefetch max_abs_err =", np.max(np.abs(C - ref)))
