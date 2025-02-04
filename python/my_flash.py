import torch
import numpy as np
import random
torch.set_printoptions(sci_mode=False)

"""
This script implements the flash attention example in doc/note.md.
"""
torch.random.manual_seed(123)

bs = 5
nheads = 32
nheads_k = 4
seqlen = 7
hdim = 128

device = 'cpu:0'
# device = 'cuda:0'
dtype = torch. float64
context_seqlen_max = 8192
decoded_seqlen_max = 512

cache_seqlen_context = 8192
cache_seqlen_decoded = torch.randint(low=421, high=421+1, size=(1,), device=device, dtype=torch.int32).repeat(bs)
seqlen_k_cache = cache_seqlen_context + cache_seqlen_decoded

q = torch.randn([bs, seqlen, nheads, hdim], device=device, dtype=dtype)
k = torch.randn([bs, seqlen, nheads_k, hdim], device=device, dtype=dtype)
v = torch.randn([bs, seqlen, nheads_k, hdim], device=device, dtype=dtype)

## single context (bs = 1)
Kcontext_cache = torch.rand([1, context_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)
Vcontext_cache = torch.rand([1, context_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)

## the decoding cache is batched (preallocated to the max seqlen)
Kdecode_cache = torch.rand([bs, decoded_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)
Vdecode_cache = torch.rand([bs, decoded_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)

# append kv to decode_cache
seqlen_knew = seqlen
for xxx in range(0, bs):
    Kdecode_cache[xxx, cache_seqlen_decoded[xxx] : cache_seqlen_decoded [xxx] + seqlen_knew, :, :] = k[xxx, :, :, :]
    Vdecode_cache[xxx, cache_seqlen_decoded[xxx] : cache_seqlen_decoded [xxx] + seqlen_knew, :, :] = v[xxx, :, :, :]
Kcache_tmp = torch.concat((Kcontext_cache [:,:cache_seqlen_context, :,:].repeat([bs, 1, 1, 1]), Kdecode_cache), dim=1)
Vcache_tmp = torch.concat((Vcontext_cache [:,:cache_seqlen_context, :,:].repeat([bs, 1, 1, 1]), Vdecode_cache), dim=1)
Kcache = torch.randn([bs, context_seqlen_max+decoded_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)
Vcache = torch.randn([bs, context_seqlen_max+decoded_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)
Kcache[:, :cache_seqlen_context + decoded_seqlen_max, :, :]= Kcache_tmp
Vcache[:, :cache_seqlen_context + decoded_seqlen_max, :, :]= Vcache_tmp

def standard_attention(q, k, v, Kcache, Vcache, cache_seqlens):
    output = torch.zeros_like(q)
    # Append kv
    bs = q.shape[0]
    seqlen_knew = q.shape[1]
    nheads = q.shape[2]
    nheads_kv=k.shape[2]
    gqa_factor = nheads // nheads_kv

    # for each batch size
    for xxx in range(0, bs):
        Kcache[xxx, cache_seqlens[xxx]: cache_seqlens[xxx] + seqlen_knew, :, :] = k[xxx, :, :, :]
        Vcache[xxx, cache_seqlens[xxx]: cache_seqlens[xxx] + seqlen_knew, :, :] = v[xxx, :, :, :]

        K = Kcache[xxx, :cache_seqlens[xxx] + seqlen_knew, :, :]
        V = Vcache[xxx, :cache_seqlens[xxx] + seqlen_knew, :, :]
        # for each q heads
        for hh in range (0, nheads):
            q1 = q[xxx, :, hh, :]
            K1 = K[:, hh // gqa_factor, :]
            V1 = V[:, hh // gqa_factor, :]
            p1 = torch.matmul(q1, torch.transpose(K1, 0, 1))
            tmp7 = p1 / np.sqrt(float(hdim))
            m = torch.max(tmp7, dim=1, keepdim=True).values
            tmp8 = tmp7 - m
            tmp9= torch.exp(tmp8)
            l = torch.sum(tmp9, dim=1, keepdim=True)
            S = tmp9 / l
            O = torch.matmul(S, V1)
            output [xxx, :, hh, :] = O
    return output

def flash_attention1(q, k, v, Kcache, Vcache, cache_seqlens):
    output = torch. zeros_like(q)
    # Append kv
    bs = q.shape[0]
    seqlen_knew = q.shape[1]
    nheads = q.shape[2]
    nheads_kv=k.shape[2]
    gqa_factor = nheads // nheads_kv
    # for each batch size
    for xxx in range(0, bs):
        Kcache [xxx, cache_seqlens [xxx]: cache_seqlens [xxx] + seqlen_knew, :, :] = k[xxx, :, :, :]
        Vcache [xxx, cache_seqlens [xxx]: cache_seqlens [xxx] + seqlen_knew, :, :] = v[xxx, :, :, :]
        K = Kcache[xxx, :cache_seqlens [xxx] + seqlen_knew, :, :]
        V = Vcache [xxx, :cache_seqlens [xxx] + seqlen_knew, :, :]
        actual_cache_seqlen = cache_seqlens [xxx] + seqlen_knew
        # for each q heads
        for hh in range(0, nheads):
           q1 = q[xxx, :, hh, :]
           K1 = K[:, hh // gqa_factor, :]
           V1 = V[:, hh // gqa_factor, :]

           #Initialize m, 1, and 0
           m = torch.full((seqlen_knew, 1), float('-inf'), device=device)
           l = torch.full((seqlen_knew, 1), float('0.0'), device=device)
           O = torch.full((seqlen_knew, hdim), float('0.0'), device=device)
           is_first_blk = True
           kBlockN = 128 # block size for kv blocks
           # FLASH ATTENTION (github) ITERATES THROUGH THE KV BLOCK IN REVERSED ORDER (backwards)
           # for blk in reversed (range(0, (actual_cache_seqlen + kBlockN - 1) // kBlockN)):
           # THIS RANDOMLY SHUFFLED ITERATION ORDERING PROVES THAT FLASH ATTENTION IS INVARIANT TO THE ORDERING IT ITERATES THROUGH THE KV BLOCKS.
           for blk in random.sample(range(0, (actual_cache_seqlen + kBlockN - 1) // kBlockN), (actual_cache_seqlen + kBlockN - 1) // kBlockN):
               # print("head = ", hh, ", blk = ", blk)
               # we don't need to clear K2 since we will mask out the scores
               K2 = torch.full((kBlockN, hdim), float('nan'), device=device, dtype=dtype)
               # we do need to set V2 to 0.0 as other P x V2 would produce all nans
               V2= torch.full((kBlockN, hdim), float('0.0'), device=device, dtype=dtype)
               if actual_cache_seqlen - blk * kBlockN < kBlockN:
                   K2[:actual_cache_seqlen - blk * kBlockN, :] = K1 [blk * kBlockN: (blk + 1) * kBlockN, :]
                   V2[:actual_cache_seqlen - blk * kBlockN, :] = V1 [blk * kBlockN: (blk + 1) * kBlockN, :]
               else:
                   K2 = K1[blk * kBlockN: (blk + 1) * kBlockN, :]
                   V2 = V1[blk * kBlockN: (blk + 1) * kBlockN, :]

               S = torch.matmul(q1, torch.transpose(K2, 0, 1)) / np.sqrt(float (hdim))

               if actual_cache_seqlen - blk * kBlockN <kBlockN:
                  # Need to apply padding mask
                  S[:, actual_cache_seqlen - blk * kBlockN: ] = float("-Inf")

               m_prev = m
               m = torch.maximum (m_prev, torch.max(S, dim=1, keepdim=True).values)
               tmp0= S - m
               P = torch.exp(tmp0)
               if is_first_blk:
                   l = torch.sum (P, dim=1, keepdim=True)
                   O = torch.matmul (P, V2)
                   is_first_blk = False
               else:
                   l = torch.exp(m_prev - m) * l + torch. sum (P, dim=1, keepdim=True)
                   O = O * torch.exp(m_prev - m) + torch.matmul (P, V2)

           tmp6 = torch.diag(torch.squeeze(torch.reciprocal(l), -1))
           O = torch.matmul(tmp6, O)
           output [xxx, :, hh, :] = O
    return output

def branched_flash_attention (q, k, v, Kcontext_cache, Vcontext_cache, Kdecode_cache, Vdecode_cache, cache_seqlen_context, cache_seqlen_decoded):
    output = torch. zeros_like(q)
    # Append kv
    bs = q.shape[0]
    seqlen_knew = q.shape[1]
    nheads= q.shape[2]
    nheads_kv=k.shape[2]
    gqa_factor = nheads // nheads_kv


    # for each batch size
    for xxx in range(0, bs):
        Kdecode_cache[xxx, cache_seqlen_decoded [xxx]: cache_seqlen_decoded [xxx] + seqlen_knew, :, :] = k[xxx, :, :, :]
        Vdecode_cache[xxx, cache_seqlen_decoded [xxx]: cache_seqlen_decoded [xxx] + seqlen_knew, :, :] = v[xxx, :, :, :]
        K = Kdecode_cache[xxx, :cache_seqlen_decoded [xxx] + seqlen_knew, :, :]
        V = Vdecode_cache[xxx, :cache_seqlen_decoded [xxx] + seqlen_knew, :, :]
        actual_cache_seqlen_decoded = cache_seqlen_decoded [xxx] + seqlen_knew
        # for each q heads
        for hh in range(0, nheads):
            q1 = q[xxx, :, hh, :]
            K1_decoded = K[:, hh // gqa_factor, :]
            V1_decoded = V[:, hh // gqa_factor, :]
            K1_context = Kcontext_cache [0, :, hh // gqa_factor, :]
            V1_context = Vcontext_cache [0, :, hh // gqa_factor, :]
            # Initialize m, l, and O
            m = torch.full((seqlen_knew, 1), float('-inf'), device=device)
            l = torch.full((seqlen_knew, 1), float('0.0'), device=device)
            O = torch. full((seqlen_knew, hdim), float('0.0'), device=device)
            is_first_blk = True
            kBlockN = 128 # block size for kv blocks
            for blk in range(0, (cache_seqlen_context + kBlockN -1) // kBlockN + (actual_cache_seqlen_decoded + kBlockN - 1) // kBlockN):
                # print("head = ", hh, ", blk = ", blk)
                # we don't need to clear K2 since we will mask out the scores
                K2 = torch. full((kBlockN, hdim), float('nan'), device=device, dtype=dtype)
                # we do need to set V2 to 0.0 as other P x V2 would produce all nans
                V2 = torch. full((kBlockN, hdim), float('0.0'), device=device, dtype=dtype)
                if blk < (cache_seqlen_context + kBlockN - 1) // kBlockN:
                    # load kv block from context kv cache.
                    if cache_seqlen_context - blk * kBlockN < kBlockN:
                      K2[:cache_seqlen_context - blk * kBlockN, :] = K1_context[blk * kBlockN : blk * kBlockN + cache_seqlen_context - blk * kBlockN, :]
                      V2[:cache_seqlen_context - blk * kBlockN, :] = V1_context[blk * kBlockN : blk * kBlockN + cache_seqlen_context - blk * kBlockN, :]
                    else:
                      K2 = K1_context[blk * kBlockN : (blk + 1) * kBlockN, :]
                      V2 = V1_context[blk * kBlockN : (blk + 1) * kBlockN, :]
                else:
                      # load kv block from decoded kv cache
                      if actual_cache_seqlen_decoded - blk * kBlockN < kBlockN:
                          K2 [:actual_cache_seqlen_decoded - blk * kBlockN, :] = K1_decoded [blk * kBlockN : blk * kBlockN + actual_cache_seqlen_decoded - blk * kBlockN, :]
                          V2 [:actual_cache_seqlen_decoded - blk * kBlockN, :] = V1_decoded [blk * kBlockN : blk * kBlockN + actual_cache_seqlen_decoded - blk * kBlockN, :]
                      else:
                          K2 = K1_decoded[blk * kBlockN : (blk + 1) * kBlockN, :]
                          V2 = V1_decoded[blk * kBlockN : (blk + 1) * kBlockN, :]
                S = torch.matmul(q1, torch. transpose(K2, 0, 1)) / np. sqrt (float (hdim))
                if (blk >= (cache_seqlen_context + kBlockN - 1) // kBlockN) and (actual_cache_seqlen_decoded - blk * kBlockN < kBlockN):
                    # Need to apply decoded padding mask
                    S[:, actual_cache_seqlen_decoded - blk * kBlockN:] = float ("-Inf")
                elif (blk < (cache_seqlen_context + kBlockN - 1) // kBlockN) and (cache_seqlen_context - blk * kBlockN< kBlockN):
                    # Need to apply context padding mask
                    S[:, cache_seqlen_context - blk * kBlockN:] = float("-Inf")

                m_prev = m
                m = torch.maximum(m_prev, torch.max(S, dim=1, keepdim=True). values)
                tmp0 = S - m
                P = torch.exp(tmp0)
                if is_first_blk:
                   l = torch. sum(P, dim=1, keepdim=True)
                   O = torch.matmul(P, V2)
                   is_first_blk = False
                else:
                   l = torch.exp(m_prev - m) * l + torch.sum(P, dim=1, keepdim=True)
                   O = O * torch.exp(m_prev - m) + torch.matmul (P, V2)

            tmp6 = torch. diag (torch. squeeze (torch. reciprocal (l), -1))
            O = torch.matmul (tmp6, O)
            output [xxx, :, hh, :] = O
    return output

out_ba = branched_flash_attention(q, k, v, Kcontext_cache, Vcontext_cache, Kdecode_cache, Vdecode_cache, cache_seqlen_context, cache_seqlen_decoded)
print ("out_ba = \n", out_ba[3, :1, 15, :])
out_fa = flash_attention1(q, k, v, Kcache, Vcache, seqlen_k_cache)
print ("out_fa = \n", out_fa[3, :1, 15, :])
out_gold = standard_attention(q, k, v, Kcache, Vcache, seqlen_k_cache)
print ("out_gold = \n", out_gold[3, : 1, 15, :])
def check_results(out_a, out_b) :
    assert len(out_a.shape) == 4
    assert len(out_b.shape) == 4
    # compare with flash_attn
    diff = (out_a - out_b). abs ()
    xxx = torch.argwhere(diff>=diff.max())
    print("out_a@max_diff =", out_a[xxx[0][0], xxx[0][1], xxx[0][2], xxx[0][3]])
    print("out_b@max_diff =", out_b[xxx[0][0], xxx[0][1], xxx[0][2], xxx[0][3]])
    assert torch.allclose(out_a, out_b, rtol=1e-08, atol=9.8e-04)
    print ("PASS.")

check_results (out_ba, out_gold)
# check_results (out_fa, out_gold)

