# SPDX-License-Identifier: Apache-2.0
"""Graph-aware V41 FP8 indexer reference, with real 16x16 cache decoding.

This correctness path owns scoring/selection only. Native V41 still produces K,
cache metadata, shared candidate buffers and attention. No hardware identity,
AITER flags, native operator schemas or model constructors are rebound.
"""
from __future__ import annotations

import torch
from vllm.config import get_current_vllm_config as current_config
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import _resolve_layer_name
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask, select_candidate_blocks,
)
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    FP8_DTYPE, cp_gather_indexer_k_quant_cache_triton, fp8_mqa_logits_torch,
    rocm_aiter_sparse_attn_indexer as native_indexer,
)

from .environment import _flag, candidate_blocks_enabled
from .quantization.dsv41_channel_fp8 import _is_v41
from .v41_indexer_graph import graph_capture_allowed, register_splitting_op

logger = init_logger(__name__)
FLAG = 'VLLM_HCU_DSV41_FP8_INDEXER_REFERENCE'
_registered = False


def enabled():
    # HCU V41 FP8 indexer reference path is enabled by default on v0.33:
    # the native FP8 indexer rejects AITER-off HCU before execution, so the
    # reference path is the only usable decode route there. Set the flag to 0
    # to fall back to the native path explicitly.
    return _flag(FLAG, default=True)


def _cache_flat(cache, head_dim):
    if cache.dtype != torch.uint8 or cache.ndim not in (3, 4):
        raise ValueError('FP8 indexer cache must be uint8 [pages,block,128+4]')
    if cache.shape[-1] != head_dim + 4 or (cache.ndim == 4 and cache.shape[2] != 1):
        raise ValueError('FP8 indexer cache byte width/head count mismatch')
    pages, block = cache.shape[:2]
    if head_dim != 128 or block <= 0 or (block > 1 and block % 16):
        raise ValueError('V41 FP8 indexer requires D128 and block1 or16-aligned SHUFFLE')
    try:
        return cache.view(pages, block * (head_dim + 4))
    except RuntimeError as error:
        raise ValueError('FP8 indexer cache has incompatible packed inner strides') from error


def _cache_pages(cache, head_dim, page_ids):
    block = cache.shape[1]
    flat = _cache_flat(cache, head_dim)[page_ids]
    pages = flat.shape[0]
    values = flat[:, :block * head_dim].contiguous().view(FP8_DTYPE)
    if block == 1:
        values = values.reshape(pages, block, head_dim)
    else:
        values = values.reshape(pages, block // 16, head_dim // 16, 16, 16)
        values = values.permute(0, 1, 3, 2, 4).reshape(pages, block, head_dim)
    scales = flat[:, block * head_dim:].contiguous().view(torch.float32).reshape(pages, block)
    return values, scales


def decode_logits(q, cache, weights, context_lens, block_tables, max_model_len):
    if q.ndim != 4 or q.dtype != FP8_DTYPE:
        raise ValueError('V41 reference Q must be FP8 [requests,next_n,heads,128]')
    batch, next_n, heads, dim = q.shape
    flat = _cache_flat(cache, dim)
    if (weights.dtype != torch.float32 or weights.shape != (batch * next_n, heads)
            or weights.device != q.device or cache.device != q.device
            or block_tables.ndim != 2 or block_tables.shape[0] != batch):
        raise ValueError('FP8 indexer weights/table/device contract mismatch')
    if context_lens.ndim == 1 and context_lens.shape == (batch,):
        limits = (context_lens[:, None] - next_n +
                  torch.arange(next_n, device=context_lens.device)[None, :] + 1).clamp_min(0)
    elif context_lens.shape == (batch, next_n):
        limits = context_lens
    else:
        raise ValueError('FP8 indexer requires final1D or effective2D context lengths')
    if q.is_cuda:
        from .v41_indexer_gpu import rectangular_logits
        return rectangular_logits(q, cache, weights, context_lens, block_tables, max_model_len)
    logits = torch.full((batch * next_n, max_model_len), -float('inf'),
                        dtype=torch.float32, device=q.device)
    block = cache.shape[1]
    for req in range(batch):
        length = int(limits[req].max().item())
        if length < 0 or length > max_model_len:
            raise ValueError('FP8 indexer context exceeds checked model length')
        if length == 0:
            continue
        count = (length + block - 1) // block
        if count > block_tables.shape[1]:
            raise ValueError('FP8 indexer block table is too short')
        pages = block_tables[req, :count].long()
        if bool(((pages < 0) | (pages >= flat.shape[0])).any().item()):
            raise ValueError('FP8 indexer block ID is outside cache allocation')
        values, scales = _cache_pages(cache, dim, pages)
        key = values.reshape(-1, dim)[:length].float()
        scale = scales.reshape(-1)[:length]
        scores = torch.einsum('qhd,nd->hqn', q[req].float(), key).relu()
        scores = (scores * weights[req * next_n:(req + 1) * next_n].t()[:, :, None]).sum(0)
        scores = scores * scale[None, :]
        visible = torch.arange(length, device=q.device)[None, :] < limits[req, :, None]
        logits[req * next_n:(req + 1) * next_n, :length] = scores.masked_fill(~visible, -float('inf'))
    return logits


def local_topk(logits, starts, ends, k):
    rows, width = logits.shape
    output = torch.full((rows, k), -1, dtype=torch.int32, device=logits.device)
    if not rows or not width:
        return output
    top = logits.topk(min(k, width), dim=-1)
    selected = (top.indices - starts[:, None]).to(torch.int32)
    selected.masked_fill_(top.values == -float('inf'), -1)
    ordered = top.indices.masked_fill(top.values == -float('inf'), width).sort(-1).values
    ordered = (ordered - starts[:, None]).to(torch.int32).masked_fill(ordered == width, -1)
    output[:, :top.indices.shape[1]] = torch.where((ends - starts <= k)[:, None], ordered, selected)
    return output


def _lightop_decode_candidate_blocks(logits, ends, candidates, block_size):
    """Write DSV4.1 decode candidate blocks through the LightOp producer.

    The public LightOp API allocates and returns both an expanded mask and the
    block IDs. The mask is deliberately discarded: the caller already holds the
    logits it needs, and only the IDs feed the shared candidate bank.

    ``candidates`` here is the caller-owned destination for exactly these rows.
    On the padded decode path ``v41_indexer_gpu.decode_indexer`` hands in a
    compact ``(rows, width)`` scratch tensor and performs the request-major
    ``index_copy_`` itself afterwards, so this helper must never reach past
    ``candidates.shape[0]`` into a wider shared buffer.

    Only the decode call sites use this helper. The LightOp contract takes
    per-row lengths and always indexes packed columns from 0, while the upstream
    helper also accepts the non-zero ``starts`` vector that prefill chunks pass.
    Decode always starts each row at column 0, so the two agree there; prefill
    keeps the Triton chain and never reaches this function.

    The producer returns ascending block IDs whereas the upstream helper keeps
    top-k score order. ``apply_candidate_mask`` reduces the IDs into a per-block
    flag table, so it is invariant to both the order and the -1 placement; the
    set is what carries meaning.

    Returns True only when the LightOp producer ran. Contract mismatches return
    False so the caller keeps the upstream Triton chain; a native execution
    failure is not converted into a silent fallback.
    """
    if not candidate_blocks_enabled():
        return False
    if (
        logits.ndim != 2
        or not logits.is_cuda
        or logits.dtype != torch.float32
        or not logits.is_contiguous()
        or logits.shape[0] == 0
        or logits.shape[1] == 0
        or logits.shape[1] % 8 != 0
        or block_size != 8
        or candidates.ndim != 2
        or candidates.dtype != torch.int32
        or not candidates.is_cuda
        or not candidates.is_contiguous()
        or candidates.device != logits.device
        or candidates.shape[0] != logits.shape[0]
        or ends.ndim != 1
        or ends.dtype not in (torch.int32, torch.int64)
        or ends.device != logits.device
        or ends.shape[0] != logits.shape[0]
        or not ends.is_contiguous()
    ):
        return False
    try:
        from lightop import attention
    except (ImportError, AttributeError):
        return False
    producer = getattr(attention, 'dsv41_select_candidate_blocks', None)
    if not callable(producer):
        return False
    rows = logits.shape[0]
    width = candidates.shape[1]
    lengths = ends.to(torch.int32)
    result = producer(logits, lengths, width, 8)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            'LightOp dsv41_select_candidate_blocks returned an invalid result')
    mask, block_ids = result
    if (
        mask.shape != logits.shape
        or mask.dtype != torch.bool
        or mask.device != logits.device
        or block_ids.shape != (rows, width)
        or block_ids.dtype != torch.int32
        or block_ids.device != logits.device
    ):
        raise RuntimeError(
            'LightOp dsv41_select_candidate_blocks returned an invalid layout')
    candidates.copy_(block_ids)
    return True


def _select_decode(logits, starts, ends, out, candidates, block_size, write):
    """Decode-only selection: every row's packed columns start at 0.

    Split from ``_select`` so the LightOp producer is chosen statically by call
    site instead of by reading ``starts`` back to the host, which would add a
    synchronisation to every decode step and break graph replay.
    """
    if candidates is not None:
        if write:
            if not _lightop_decode_candidate_blocks(logits, ends, candidates, block_size):
                select_candidate_blocks(logits, starts, ends, candidates.shape[1],
                                        block_size, candidates)
        else:
            apply_candidate_mask(logits, starts, ends, candidates, block_size)
    out.copy_(local_topk(logits, starts, ends, out.shape[1]))


def _select(logits, starts, ends, out, candidates, block_size, write):
    if candidates is not None:
        if write:
            select_candidate_blocks(logits, starts, ends, candidates.shape[1], block_size, candidates)
        else:
            apply_candidate_mask(logits, starts, ends, candidates, block_size)
    out.copy_(local_topk(logits, starts, ends, out.shape[1]))


@eager_break_during_capture
def indexer(hidden_states: torch.Tensor, prefix: str, cache: torch.Tensor,
            q: torch.Tensor, weights: torch.Tensor, topk_tokens: int,
            head_dim: int, max_model_len: int, total_seq_lens: int,
            out: torch.Tensor, compress_ratio: int,
            candidates: torch.Tensor | None = None, candidate_block_size: int = 0,
            candidate_write: bool = False) -> torch.Tensor:
    prefix = _resolve_layer_name(prefix)
    context = get_forward_context()
    if not isinstance(context.attn_metadata, dict):
        # Reuse the actual native profiling reservations/fake ABI; this branch
        # never reads cache bytes or invokes the faulty paged decode fallback.
        return native_indexer(hidden_states, prefix, cache, q, None, weights,
                              128, 'ue8m0', topk_tokens, head_dim, max_model_len,
                              total_seq_lens, out, True, compress_ratio,
                              candidates, candidate_block_size, candidate_write)
    meta = context.attn_metadata[prefix]
    if not isinstance(meta, DeepseekV32IndexerMetadata):
        raise TypeError('V41 reference indexer requires native indexer metadata')
    actual = meta.num_decode_tokens + meta.num_prefill_tokens
    if q.ndim != 3 or q.dtype != FP8_DTYPE or q.shape[0] < actual or q.shape[2] != head_dim:
        raise ValueError('V41 reference query layout/token count mismatch')
    if weights.dtype != torch.float32 or weights.shape != q.shape[:2] or weights.device != q.device:
        raise ValueError('V41 indexer requires same-device FP32 per-head weights')
    if out.dtype != torch.int32 or out.ndim != 2 or out.shape[0] < actual or out.shape[1] < topk_tokens or out.device != q.device:
        raise ValueError('V41 indexer output buffer shape/dtype/device mismatch')
    selected_out = out[:, :topk_tokens]
    if cache.device != q.device:
        raise ValueError('V41 indexer cache must be on the query device')
    _cache_flat(cache, head_dim)
    if candidates is not None:
        if (candidates.dtype != torch.int32 or candidates.ndim != 2 or candidates.shape[0] < actual
                or candidates.shape[1] <= 0 or candidates.device != q.device
                or type(candidate_block_size) is not int or candidate_block_size <= 0):
            raise ValueError('V41 candidate schema requires int32 rows and positive block size')
        if meta.decode is not None and meta.decode.requires_padding:
            if not q.is_cuda:
                counts = meta.decode.decode_lens.tolist()
                padded_rows = len(counts) * max(counts, default=0)
                if candidates.shape[0] < padded_rows:
                    raise ValueError('ragged candidate bank lacks padded request-major rows')
            if meta.num_prefill_tokens:
                raise ValueError('padded decode candidates overlap prefill rows; use separate batches')
    elif candidate_write:
        raise ValueError('candidate_write requires a candidate buffer')
    out[:hidden_states.shape[0]].fill_(-1)
    if meta.num_prefill_tokens:
        if meta.prefill is None:
            raise ValueError('missing V41 prefill chunk metadata')
        for chunk in meta.prefill.chunks:
            start, end = chunk.token_start, chunk.token_end
            keys = torch.empty((chunk.total_seq_lens, head_dim), dtype=FP8_DTYPE, device=q.device)
            scales = torch.empty((chunk.total_seq_lens, 4), dtype=torch.uint8, device=q.device)
            cp_gather_indexer_k_quant_cache_triton(cache, keys, scales, chunk.block_table,
                chunk.cu_seq_lens, token_to_seq=chunk.token_to_seq)
            logits = fp8_mqa_logits_torch(q[start:end], (keys, scales.view(torch.float32)),
                                         weights[start:end], chunk.cu_seqlen_ks, chunk.cu_seqlen_ke)
            _select(logits, chunk.cu_seqlen_ks, chunk.cu_seqlen_ke, selected_out[start:end],
                    None if candidates is None else candidates[start:end], candidate_block_size, candidate_write)
    if q.is_cuda:
        if meta.num_decode_tokens:
            if meta.decode is None:
                raise ValueError('missing V41 decode metadata')
            from .v41_indexer_gpu import decode_indexer
            decode_indexer(q, cache, weights, meta.decode, meta.num_decode_tokens,
                           selected_out, compress_ratio, candidates, candidate_block_size,
                           candidate_write, max_model_len, _select_decode)
        return out
    if meta.num_decode_tokens:
        if meta.decode is None:
            raise ValueError('missing V41 decode metadata')
        decode = meta.decode
        lengths = decode.decode_lens.tolist()
        if sum(lengths) != meta.num_decode_tokens or any(n < 0 for n in lengths):
            raise ValueError('V41 decode_lens do not cover actual query rows')
        offset = 0
        candidate_stride = max(lengths, default=0) if decode.requires_padding else None
        if candidates is not None and candidate_write and candidate_stride is not None:
            candidates[:len(lengths) * candidate_stride].fill_(-1)
        # Q and outputs stay compact. Candidate rows retain Core's padded
        # request-major ABI without allocating padded queries or weights.
        for req, count in enumerate(lengths):
            if not count:
                continue
            if decode.seq_lens.ndim == 2:
                limits = decode.seq_lens[req:req + 1, :count]
                if limits.shape[1] != count:
                    raise ValueError('V41 effective context rows are incomplete')
            else:
                if count > 1 and compress_ratio != 1:
                    raise ValueError('ratio2 multi-query decode requires effective per-query context lengths')
                limits = (decode.seq_lens[req:req + 1, None] - count +
                          torch.arange(count, device=q.device)[None, :] + 1).clamp_min(0)
            end = offset + count
            logits = decode_logits(q[offset:end].unsqueeze(0), cache, weights[offset:end],
                                    limits, decode.block_table[req:req + 1], max_model_len)
            starts = torch.zeros(count, dtype=torch.int64, device=q.device)
            bank_start = req * candidate_stride if candidate_stride is not None else offset
            bank_rows = None if candidates is None else candidates[bank_start:bank_start + count]
            _select_decode(logits, starts, limits.reshape(-1), selected_out[offset:end],
                           bank_rows, candidate_block_size, candidate_write)
            offset = end
    return out


def _fake(hidden_states, prefix, cache, q, weights, topk_tokens, head_dim,
          max_model_len, total_seq_lens, out, compress_ratio, candidates=None,
          candidate_block_size=0, candidate_write=False):
    return out


def register_op():
    global _registered
    if _registered:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name='hcu_v41_fp8_indexer_reference', op_func=indexer,
                              mutates_args=['out', 'candidates'], fake_impl=_fake,
                              dispatch_key='CompositeExplicitAutograd',
                              tags=(torch._C.Tag.cudagraph_unsafe,))
    _registered = True


class HcuV41SparseAttnIndexer(SparseAttnIndexer):
    def __init__(self, *args, **kwargs):
        cfg = current_config()
        is_v41 = _is_v41(cfg.model_config.hf_config)
        if is_v41:
            from .v41_indexer_activation import check_health
            check_health()
            if enabled():
                register_splitting_op(cfg)
        self._hcu_owned = enabled() and is_v41
        if self._hcu_owned:
            import inspect
            bound = inspect.signature(SparseAttnIndexer.__init__).bind(self, *args, **kwargs)
            bound.apply_defaults()
            p = cfg.parallel_config
            if cfg.kernel_config.monokernel:
                raise ValueError('V41 reference indexer cannot mix candidate buffers with monokernel producers')
            if not cfg.model_config.enforce_eager and not graph_capture_allowed(cfg):
                raise ValueError(
                    'V41 reference indexer requires eager execution or a '
                    'PIECEWISE breakable/inductor-split graph capture; '
                    'full cudagraph cannot break out the host-metadata op')
            if p.prefill_context_parallel_size != 1 or p.decode_context_parallel_size != 1:
                raise ValueError('V41 reference indexer requires PCP1/DCP1; PCP8 is a separate adapter')
            values = bound.arguments
            if (values['use_fp4_cache'] or not values['skip_k_cache_insert']
                    or values['compress_ratio'] not in (1, 2) or values['head_dim'] != 128):
                raise ValueError('V41 reference indexer requires source-owned FP8 K, ratio1/2, D128')
        super().__init__(*args, **kwargs)
        if self._hcu_owned:
            logger.info('HCU_V41_FP8_INDEXER_REFERENCE: graph/eager PCP1/DCP1 ratio=%s cache=%s',
                        self.compress_ratio, self.k_cache.prefix)

    def forward_hip(self, hidden_states, q_quant, k, weights):
        if not self._hcu_owned:
            return super().forward_hip(hidden_states, q_quant, k, weights)
        if k is not None or not isinstance(q_quant, torch.Tensor):
            raise ValueError('V41 reference indexer consumes source-owned K and one FP8 Q tensor')
        return torch.ops.vllm.hcu_v41_fp8_indexer_reference(
            hidden_states, _resolve_layer_name(self.k_cache.prefix), self.k_cache.kv_cache, q_quant,
            weights, self.topk_tokens, self.head_dim, self.max_model_len,
            self.max_total_seq_len, self.topk_indices_buffer, self.compress_ratio,
            self.candidate_blocks, self.candidate_block_size, self.candidate_write)
