import torch
torch.set_printoptions(sci_mode=False)

"""
This script implements the flash attention example in doc/note.md.
"""

torch.random.manual_seed(123)

bs = 20
nheads = 64
nheads_k = 8
seqlen = 7
hdim = 256

device = 'cpu:0'
dtype = torch.float64
context_seqlen_max = 4096
decoded_seqlen_max = 512

cache_seqlen_context = 110
cache_seqlen_decoded = 90

assert seqlen + cache_seqlen_decoded < 128, \
       "only 2 blocks are supported : one context, one decoded; \
       num of new tokens will overflow the decoded block > 128."

q = torch.randn([bs, seqlen, nheads, hdim], device=device, dtype=dtype)
k = torch.randn([bs, seqlen, nheads_k, hdim], device=device, dtype=dtype)
v = torch.randn([bs, seqlen, nheads_k, hdim], device=device, dtype=dtype)

## single context (bs = 1)
Kcontext_cache = torch.rand([1, context_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)
Vcontext_cache = torch.rand([1, context_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)

## the decoding cache is batched (preallocated to the max seqlen)
Kdecode_cache = torch.rand([bs, decoded_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)
Vdecode_cache = torch.rand([bs, decoded_seqlen_max, nheads_k, hdim], device=device, dtype=dtype)

# take a slice from the 8th batch and the 45th q head
q1 = q[8, :, 45, :].to(torch.float)
# in GQA, 1 k head attends to 8 q heads, so the 5th k head attends to the 45th q head
k1 = Kcontext_cache[0, :128, 5, :].to(torch.float)
k2 = Kdecode_cache[8, :128, 5, :].to(torch.float)
v1 = Vcontext_cache[0, :128, 5, :].to(torch.float)
v2 = Vdecode_cache[8, :128, 5, :].to(torch.float)


# append kv
k2[cache_seqlen_decoded:cache_seqlen_decoded+seqlen, :] = k[8, :, 5, :].to(torch.float)
v2[cache_seqlen_decoded:cache_seqlen_decoded+seqlen, :] = v[8, :, 5, :].to(torch.float)

def mini_flash(q1, k1, k2, v1, v2, cache_seqlen_context):
    # flash attention
    s1 = torch.matmul(q1, torch.transpose(k1, 0, 1)) / 11.3137 # sqrt(128.0) = 11.3137
    s1[:, cache_seqlen_context:] = float("-Inf") 
    m1 = torch.max(s1, dim=1, keepdim=True).values
    tmp0 = s1 - m1
    tmp1 = torch.exp(tmp0)
    l1 = torch.sum(tmp1, dim=1, keepdim=True)

    tmp2 = torch.diag(torch.squeeze(torch.reciprocal(l1), -1))
    P1 = torch.matmul(tmp2, tmp1)
    O1 = torch.matmul(P1, v1)

    s2 = torch.matmul(q1, torch.transpose(k2, 0, 1)) / 11.3137
    s2[:, cache_seqlen_decoded + seqlen:] = float("-Inf") 
    tmp3 = torch.max(s2, dim=1, keepdim=True).values
    m2 = torch.maximum(m1, tmp3)

    tmp4 = s2 - m2
    tmp5 = torch.exp(tmp4)

    l2 = torch.exp(m1 - m2) * l1 + torch.sum(tmp5, dim=1, keepdim=True)
    tmp6 = torch.diag(torch.squeeze(torch.reciprocal(l2), -1))
    P2 = torch.matmul(tmp6, tmp5)
    O2 = torch.matmul(torch.diag(torch.squeeze(l1 / l2, -1)), O1) * torch.exp(m1 - m2) + torch.matmul(P2, v2)
    return O2

out_ = mini_flash(q1, k1, k2, v1, v2, cache_seqlen_context)
print(out_)

# standard attention
Q = q1
# K = torch.concat([k1, k2], dim=0)
K = torch.concat([k1[:cache_seqlen_context, :], k2[:cache_seqlen_decoded + seqlen, :]], dim=0)
P = torch.matmul(Q, torch.transpose(K, 0, 1))
tmp7 = P / 11.3137
m = torch.max(tmp7, dim=1, keepdim=True).values
tmp8 = tmp7 - m
tmp9 = torch.exp(tmp8)
l = torch.sum(tmp9, dim=1, keepdim=True)
S = tmp9 / l

# V = torch.concat([v1, v2], dim=0)
V = torch.concat([v1[:cache_seqlen_context, :], v2[:cache_seqlen_decoded + seqlen, :]], dim=0)
out_gold = torch.matmul(S, V)
print(out_gold)

# compare with flash_attn
# gold = out_og[8, :, 45, :]
# print(gold)

# assert torch.allclose(out_, out_gold, rtol=1e-08, atol=1e-06)
assert torch.allclose(out_, out_gold)
print("PASS.")
