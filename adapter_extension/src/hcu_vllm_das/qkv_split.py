# SPDX-License-Identifier: Apache-2.0
"""Keep native MXFP8 KV writes; avoid the fused Q/KV kernel's scratch traffic.

The verified gfx938 binary leaves the first even rotation in each 16-element
lane group unfused. Its other even rotations and odd rotations use different
FMA operands. Preserve those boundaries; unknown native builds keep Core.
"""
from __future__ import annotations

import hashlib
from importlib.util import find_spec
from pathlib import Path

import torch
from vllm.triton_utils import tl, triton


_VERIFIED_NATIVE_SHA256 = '06ecf1eaa83b9b29be5d3a7ef4eed9b66af5b47c97cfef6ccd48b5011f256568'


def _native_build_matches():
    try:
        spec = find_spec('vllm._C_stable_libtorch')
        return (spec is not None and spec.origin is not None
                and hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
                == _VERIFIED_NATIVE_SHA256)
    except (ImportError, OSError, ValueError):
        return False


# Evaluated at model-class import, before compilation/capture; no file IO on
# the forward path and no process-global native operator replacement.
_NATIVE_BUILD_MATCHES = _native_build_matches()


def supported(q, kv, cache, slots, positions, cos_sin):
    tensors = (q, kv, cache, slots, positions, cos_sin)
    if (not _NATIVE_BUILD_MATCHES
            or not all(isinstance(t, torch.Tensor) for t in tensors)
            or q.ndim != 3 or q.shape[1:] != (64, 512)
            or not 0 < q.shape[0] <= 64
            or kv.shape != (q.shape[0], 512)
            or cache.ndim != 3 or cache.shape[0] <= 0 or cache.shape[1:] != (64, 528)
            or slots.ndim != 1 or slots.shape[0] > q.shape[0]
            or positions.shape != (q.shape[0],)
            or cos_sin.ndim != 2 or cos_sin.shape[0] <= 0 or cos_sin.shape[1] != 64
            or q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16
            or cache.dtype != torch.uint8 or cos_sin.dtype != torch.float32
            or slots.dtype != torch.int64 or positions.dtype != torch.int64
            or not all(t.is_cuda and t.device == q.device and t.is_contiguous() for t in tensors)):
        return False
    return getattr(torch.cuda.get_device_properties(q.device), 'gcnArchName', '').split(':')[0] == 'gfx938'


@triton.jit
def _q_rope_native_512(Q, Positions, CosSin, Out):
    token = tl.program_id(0)
    head = tl.program_id(1) * 4 + tl.arange(0, 4)[:, None]
    col = tl.arange(0, 512)[None, :]
    raw = tl.load(Q + token * 32768 + head * 512 + col).to(tl.float32)
    peer = tl.load(Q + token * 32768 + head * 512 + (col ^ 1), col >= 448, other=0).to(tl.float32)
    position = tl.load(Positions + token)
    index = (col - 448) // 2
    cosine = tl.load(CosSin + position * 64 + index, col >= 448, other=1.)
    sine = tl.load(CosSin + position * 64 + 32 + index, col >= 448, other=0.)
    even = col % 2 == 0
    ordinary = tl.where(even, tl.fma(raw, cosine, -peer * sine),
                        tl.fma(raw, cosine, peer * sine))
    unfused = raw * cosine + tl.where(even, -peer * sine, peer * sine)
    value = tl.where(col % 16 == 0, unfused, ordinary)
    value = tl.where(col >= 448, value, raw)
    tl.store(Out + token * 32768 + head * 512 + col, value)


def split_qkv(q, kv, cache, slots, positions, cos_sin):
    if not supported(q, kv, cache, slots, positions, cos_sin):
        raise ValueError('Q/KV split is outside the verified gfx938 native-build contract')
    out = torch.empty_like(q)
    _q_rope_native_512[(q.shape[0], 16)](
        q, positions, cos_sin, out, num_warps=4, enable_fp_fusion=False)
    torch.ops._C.fused_deepseek_v4_kv_rope_insert(
        kv, cache, slots, positions, cos_sin, 64, None, True)
    return out
