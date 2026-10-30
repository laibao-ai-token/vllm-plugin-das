# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Functional FP8 candidate filtering; request-local compressed block IDs.

Reuse the fixed core's bounded Triton candidate kernels. The Torch consumer
selector sorts only the selected K indices when preserving short-row order.
"""
import torch

from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask as _apply_candidate_mask,
    select_candidate_blocks as _select_candidate_blocks,
)


def validate_candidate_schema(candidates, block_size, write, topk_buffer,
                              use_fp4_cache=False, dcp_world_size=1, pcp_world_size=1):
    if candidates is None:
        if write:
            raise ValueError("candidate_write=True requires candidate_blocks")
        return
    if use_fp4_cache or dcp_world_size != 1 or pcp_world_size != 1:
        raise NotImplementedError("HCU candidate filtering requires FP8, DCP=1 and PCP=1")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("candidate_block_size must be a positive integer")
    if candidates.ndim != 2 or candidates.shape[1] <= 0 or candidates.dtype != torch.int32:
        raise ValueError("candidate_blocks must be a nonempty-width 2D int32 buffer")
    if topk_buffer is None or topk_buffer.ndim != 2 or topk_buffer.dtype != torch.int32:
        raise ValueError("candidate filtering requires a 2D int32 topk buffer")
    if candidates.shape[0] != topk_buffer.shape[0] or candidates.device != topk_buffer.device:
        raise ValueError("candidate and topk buffers must have equal rows and device")
    if any(s <= 0 for s in candidates.stride()):
        raise ValueError("candidate buffer strides must be positive")


def _candidate_bounds(logits, starts, ends):
    rows = logits.shape[0]
    if ends.numel() != rows or (starts is not None and starts.numel() != rows):
        raise ValueError("candidate bounds must describe each logits row")
    ends = ends.reshape(-1).to(device=logits.device, dtype=torch.long)
    starts = (torch.zeros_like(ends) if starts is None else
              starts.reshape(-1).to(device=logits.device, dtype=torch.long))
    return starts, ends


def select_candidate_blocks(logits, starts, ends, block_size, out):
    _select_candidate_blocks(logits, starts, ends, out.shape[1], block_size, out)


def apply_candidate_mask(logits, starts, ends, candidates, block_size):
    _apply_candidate_mask(logits, starts, ends, candidates, block_size)


def candidate_topk(logits, topk_tokens, starts, ends):
    """Top-k of masked scores; never resurrect masked positions on short rows."""
    rows, width = logits.shape
    starts, ends = _candidate_bounds(logits, starts, ends)
    out = torch.full((rows, topk_tokens), -1, dtype=torch.int32, device=logits.device)
    if width == 0 or rows == 0:
        return out
    # Native apply_candidate_mask already bounded this matrix. Reuse it
    # directly; all following scratch is proportional to selected K.
    k = min(width, topk_tokens)
    top = logits.topk(k, dim=-1)
    selected = (top.indices - starts[:, None]).to(torch.int32)
    selected.masked_fill_(top.values == float('-inf'), -1)
    ordered = top.indices.masked_fill(top.values == float('-inf'), width).sort(dim=1).values
    ordered_local = (ordered - starts[:, None]).to(torch.int32).masked_fill(ordered == width, -1)
    selected = torch.where(((ends - starts) <= topk_tokens)[:, None], ordered_local, selected)
    out[:, :k].copy_(selected)
    return out
