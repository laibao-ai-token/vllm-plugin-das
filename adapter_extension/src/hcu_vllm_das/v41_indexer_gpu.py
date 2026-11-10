# SPDX-License-Identifier: Apache-2.0
"""Device-only decode scoring for the V41 FP8 reference indexer.

The reference callback can run between replayed graph segments. Its launch
shapes come from tensor/metadata shapes; GPU lengths and page IDs never become
Python values. Row mapping preserves compact queries and padded candidate banks.
"""
from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _decode_rows(lens, seq, requests, ends, banks, error,
                 lens_stride, seq_stride, seq_col_stride,
                 ROWS: tl.constexpr, REQUESTS: tl.constexpr,
                 LENS_TILE: tl.constexpr, SEQ_COLS: tl.constexpr,
                 EFFECTIVE_2D: tl.constexpr, PADDED: tl.constexpr,
                 RATIO: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    ids = tl.arange(0, LENS_TILE)
    counts = tl.load(lens + ids * lens_stride, ids < REQUESTS, other=0)
    prefix = tl.cumsum(counts)
    owned = (ids < REQUESTS) & (row >= prefix - counts) & (row < prefix)
    request = tl.sum(tl.where(owned, ids, 0), 0)
    count = tl.sum(tl.where(owned, counts, 0), 0)
    start = tl.sum(tl.where(owned, prefix - counts, 0), 0)
    within = row - start
    valid = tl.sum(owned.to(tl.int32), 0) == 1
    if EFFECTIVE_2D:
        valid = valid & (within < SEQ_COLS)
        end = tl.load(seq + request * seq_stride + within * seq_col_stride,
                      mask=valid, other=0)
    else:
        # A 1D final length can reconstruct per-query bounds only for ratio 1.
        valid = valid & ((count <= 1) | (RATIO == 1))
        final = tl.load(seq + request * seq_stride, mask=valid, other=0)
        valid = valid & (final >= 0)
        end = tl.maximum(final - count + within + 1, 0)
    if (not valid) | (end < 0) | (end > WIDTH):
        tl.atomic_or(error, 1)
        end = 0
    stride = tl.max(counts, 0)
    bank = request * stride + within if PADDED else row
    tl.store(requests + row, request)
    tl.store(ends + row, end)
    tl.store(banks + row, bank)


@triton.jit
def _paged_scores(q, cache_values, cache_scales, weights, requests, ends,
                  tables, logits, error,
                  q_row_stride, q_head_stride, q_dim_stride,
                  weight_row_stride, weight_head_stride,
                  cache_page_bytes, table_row_stride, table_col_stride,
                  WIDTH: tl.constexpr, TABLE_ROWS: tl.constexpr,
                  TABLE_COLS: tl.constexpr, PAGES: tl.constexpr,
                  PAGE_SIZE: tl.constexpr, HEADS: tl.constexpr,
                  HEAD_TILE: tl.constexpr, KV_TILE: tl.constexpr,
                  SPLITS: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    req = tl.load(requests + row)
    end = tl.load(ends + row)
    valid_row = (req >= 0) & (req < TABLE_ROWS)
    if (end < 0) | (end > WIDTH) | (end > TABLE_COLS * PAGE_SIZE) | ~valid_row:
        tl.atomic_or(error, 2)
    length = tl.minimum(tl.maximum(end, 0), tl.minimum(WIDTH, TABLE_COLS * PAGE_SIZE))
    h = tl.arange(0, HEAD_TILE)
    d = tl.arange(0, 128)
    query = tl.load(q + row * q_row_stride + h[:, None] * q_head_stride
                    + d[None, :] * q_dim_stride, h[:, None] < HEADS, other=0.0).to(tl.float32)
    weight = tl.load(weights + row * weight_row_stride + h * weight_head_stride,
                     h < HEADS, other=0).to(tl.float32)
    k = tl.arange(0, KV_TILE)
    for first in tl.range(split * KV_TILE, length, SPLITS * KV_TILE):
        pos = first + k
        logical_page = pos // PAGE_SIZE
        valid_pos = (pos < length) & valid_row
        page = tl.load(tables + req * table_row_stride + logical_page * table_col_stride,
                       valid_pos, other=-1)
        safe = valid_pos & (page >= 0) & (page < PAGES)
        if tl.sum((valid_pos & ~safe).to(tl.int32), 0) != 0:
            tl.atomic_or(error, 4)
        page = tl.where(safe, page, 0).to(tl.int64)
        slot = pos % PAGE_SIZE
        if PAGE_SIZE == 1:
            in_page = d[:, None] + tl.zeros((1, KV_TILE), tl.int32)
        else:
            # Values are [block/16, D/16, 16 tokens, 16 bytes], then FP32 scales.
            in_page = ((slot[None, :] // 16) * (16 * 128)
                       + (d[:, None] // 16) * 256
                       + (slot[None, :] % 16) * 16 + d[:, None] % 16)
        key = tl.load(cache_values + page[None, :] * cache_page_bytes + in_page,
                      safe[None, :], other=0.0).to(tl.float32)
        scale = tl.load(cache_scales + page * (cache_page_bytes // 4)
                        + PAGE_SIZE * 128 // 4 + slot, safe, other=0)
        dot = tl.dot(query, key, input_precision='ieee')
        score = tl.sum(tl.maximum(dot, 0.0) * weight[:, None], 0) * scale
        tl.store(logits + row * WIDTH + pos,
                 tl.where(safe, score, -float('inf')), pos < length)


def paged_logits(q, cache, weights, ends, tables, requests, width):
    """Score compact FP8 queries without host reads or a full-cache temporary."""
    rows, heads, dim = q.shape
    if dim != 128 or heads <= 0 or q.stride(-1) != 1:
        raise ValueError('V41 GPU scoring requires FP8 [tokens,heads,128]')
    if (tables.ndim != 2 or requests.shape != (rows,) or ends.shape != (rows,)
            or weights.shape != (rows, heads) or width < 0 or width > 2147483647):
        raise ValueError('V41 GPU scoring row/table/weight shapes disagree')
    for tensor in (tables, requests, ends):
        if tensor.device != q.device or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError('V41 GPU scoring metadata requires same-device integer tensors')
    logits = torch.full((rows, width), -float('inf'), dtype=torch.float32, device=q.device)
    if not rows or not width:
        return logits
    error = torch.zeros((), dtype=torch.int32, device=q.device)
    splits = max(1, min(triton.cdiv(width, 32), max(1, 256 // rows)))
    _paged_scores[(rows, splits)](
        q, cache.view(q.dtype), cache.view(torch.float32), weights, requests, ends,
        tables, logits, error, *q.stride(), *weights.stride(), cache.stride(0),
        *tables.stride(), WIDTH=width, TABLE_ROWS=tables.shape[0],
        TABLE_COLS=tables.shape[1], PAGES=cache.shape[0], PAGE_SIZE=cache.shape[1],
        HEADS=heads, HEAD_TILE=max(16, triton.next_power_of_2(heads)), KV_TILE=32,
        SPLITS=splits, num_warps=4, enable_fp_fusion=False)
    # Device-side assertion preserves fail-closed page checks without D2H.
    torch._assert_async(error == 0, 'V41 indexer invalid context length or cache page')
    return logits


def rectangular_logits(q, cache, weights, limits, tables, width):
    batch, next_n, heads, dim = q.shape
    if limits.ndim == 1 and limits.shape == (batch,):
        ends = (limits[:, None] - next_n + torch.arange(next_n, device=q.device)[None, :] + 1).clamp_min(0)
    elif limits.shape == (batch, next_n):
        ends = limits
    else:
        raise ValueError('V41 GPU scoring needs final1D or effective2D lengths')
    requests = torch.arange(batch, device=q.device, dtype=torch.int32).repeat_interleave(next_n)
    return paged_logits(q.reshape(batch * next_n, heads, dim), cache, weights,
                        ends.reshape(-1), tables, requests, width)


def decode_indexer(q, cache, weights, decode, rows, out, ratio, candidates,
                   candidate_block_size, candidate_write, width, select):
    """Map final native decode metadata on device, preserving candidate row ABI."""
    counts = decode.decode_lens
    for tensor in (counts, decode.seq_lens, decode.block_table):
        if tensor.device != q.device or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError('V41 GPU decode metadata requires same-device integer tensors')
    if counts.ndim != 1 or decode.seq_lens.ndim not in (1, 2):
        raise ValueError('V41 GPU decode metadata must contain vector counts and 1D/2D contexts')
    if decode.seq_lens.shape[0] != counts.numel() or decode.block_table.shape[0] != counts.numel():
        raise ValueError('V41 GPU decode request extents disagree')
    if not rows:
        return
    if not counts.numel():
        raise ValueError('V41 GPU decode has query rows without requests')
    torch._assert_async((counts >= 0).all() & (counts.sum() == rows),
                        'V41 decode_lens do not cover actual query rows')
    requests = torch.empty(rows, dtype=torch.int32, device=q.device)
    ends, banks = torch.empty_like(requests), torch.empty_like(requests)
    error = torch.zeros((), dtype=torch.int32, device=q.device)
    seq = decode.seq_lens
    _decode_rows[(rows,)](counts, seq, requests, ends, banks, error,
        counts.stride(0), seq.stride(0), seq.stride(1) if seq.ndim == 2 else 0,
        ROWS=rows, REQUESTS=counts.numel(), LENS_TILE=triton.next_power_of_2(counts.numel()),
        SEQ_COLS=seq.shape[1] if seq.ndim == 2 else 1,
        EFFECTIVE_2D=seq.ndim == 2, PADDED=decode.requires_padding, RATIO=ratio,
        WIDTH=width)
    torch._assert_async(error == 0, 'V41 effective context rows are incomplete')
    logits = paged_logits(q[:rows], cache, weights[:rows], ends,
                          decode.block_table, requests, width)
    compact_candidates = None
    if candidates is not None:
        torch._assert_async(((banks >= 0) & (banks < candidates.shape[0])).all(),
                            'V41 candidate bank lacks padded request-major rows')
        if candidate_write:
            if decode.requires_padding:
                padded_rows = counts.numel() * counts.max()
                torch._assert_async(padded_rows <= candidates.shape[0],
                                    'V41 candidate bank lacks padded request-major rows')
                clear = torch.arange(candidates.shape[0], device=q.device) < padded_rows
                candidates.copy_(torch.where(clear[:, None], -1, candidates))
            compact_candidates = torch.empty((rows, candidates.shape[1]), dtype=torch.int32, device=q.device)
        else:
            compact_candidates = candidates.index_select(0, banks)
    select(logits, torch.zeros_like(ends), ends, out[:rows], compact_candidates,
           candidate_block_size, candidate_write)
    if compact_candidates is not None and candidate_write:
        candidates.index_copy_(0, banks.to(torch.int64), compact_candidates)
