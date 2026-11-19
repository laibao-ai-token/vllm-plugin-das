# SPDX-License-Identifier: Apache-2.0
"""Experimental masked receiver quantization using the actual AG/RS row JIT."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import torch
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.quantization.utils import int8_utils as native

_row_quantize = native._per_token_quant_int8
_public_quantize = native.per_token_quant_int8


@lru_cache(maxsize=1)
def _source_matches():
    return hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == (
        "e046508a0884c7f0a2913f34e5f21b82121e9e40f74dd355ded16a5d0d5801b7")


@triton.jit
def _masked_native_int8(X, Q, S, Counts, CAPACITY: tl.constexpr,
                        ROWS: tl.constexpr):
    start = tl.program_id(0)
    expert = tl.program_id(1)
    row = start
    end = tl.minimum(tl.maximum(tl.load(Counts + expert), 0), CAPACITY)
    while row < end:
        # The native JIT uses program_id(0) for its row. Offset its expert
        # base by the loop's current displacement to preserve that body and
        # the native BLOCK8192/8-warp arithmetic exactly.
        offset = expert * CAPACITY + row - start
        _row_quantize(X + offset * 5120, Q + offset * 5120, S + offset,
                      5120, 5120, 5120, BLOCK=8192)
        row += ROWS


def quantize_receiver(x, counts, expected_m=32):
    """Use native round/minimum-scale rules after BF16 LL transmission."""
    from vllm.model_executor.layers.fused_moe import utils

    if (not _source_matches() or native._per_token_quant_int8 is not _row_quantize
            or native.per_token_quant_int8 is not _public_quantize
            or utils.per_token_quant_int8 is not _public_quantize):
        raise RuntimeError("W4A8 draft receiver native quantization changed")
    if (x.dtype != torch.bfloat16 or x.ndim != 3 or x.shape[0] not in (16, 48)
            or x.shape[1:] != (2048, 5120) or not x.is_cuda or not x.is_contiguous()
            or counts.dtype != torch.int32 or counts.shape != x.shape[:1]
            or counts.device != x.device or not counts.is_contiguous()):
        raise ValueError("W4A8 draft receiver requires the fixed256x8 LL layout")
    props = torch.cuda.get_device_properties(x.device)
    if props.gcnArchName.split(":")[0] != "gfx938" or props.multi_processor_count != 64:
        raise ValueError("W4A8 draft receiver is outside verified device geometry")
    q = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    scales = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
    rows = min(2048, max(1, int(expected_m)))
    _masked_native_int8[(rows, x.shape[0])](
        x, q, scales, counts, 2048, rows, num_warps=8, num_stages=1)
    return q, scales
