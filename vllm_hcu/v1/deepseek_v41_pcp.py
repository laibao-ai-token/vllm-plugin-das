# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Eager V41 PCP correctness baseline: local Engram, global attention/cache rows.

Input projections and Q are deliberately replicated here. Hidden states are
restored before CR2 compression; candidate/topk and both source caches retain
native global row order. Only the completed attention output is localized.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import textwrap
from contextlib import contextmanager
from contextvars import ContextVar
from numbers import Integral
from types import MethodType

import torch

from vllm_hcu.runtime_compat.deepseek_v41_pcp import is_v41_pcp, require_v41_pcp_contract

_ACTIVE_ROWS = ContextVar("hcu_v41_global_attention_rows", default=None)
_INITIAL_PROFILE = ContextVar("hcu_v41_initial_memory_profile", default=None)
_MARKER = "_hcu_v41_full_cache_pcp"
_AUDITED_SOURCES = {
    "DeepseekV4Attention.forward": "ad4ac3bb35faffc86cf11f5b7537ab1af8400e739c8622b5e91533c8048a22b8",
    "DeepseekV4Attention._prepare_and_attn": "584389b969ab08a3a4ebeb79be3b356c0b920f6348b5cd3bd1a20260cb7febbc",
    "DeepseekV4Indexer.forward": "f81d9bb9a709bd209a0a8ad8c6c19a69d2da30fd6655331069777b4c1724cb44",
    "DeepseekV4Indexer._produce_k": "5b75def248f7b2b42d7ff4ea07f30702c11bb4113f6f1996dc2e53e130ccc96c",
    "DeepseekCompressor.forward": "383d27502064d5693f4e1a357df4c2c5257ded00eeb30263fb28db3288e95f50",
    "DeepseekCompressor.insert_cache": "7343920064feaf7975c583956201c447651957985384b8a568faa56f674d4736",
    "DeepseekV41ModelState.prepare_attn": "2265004d08be62ded754d578b824d4cb0ce487b2929b59f95ac174d16bb11337",
}


def _batch():
    from vllm_hcu.v1.glm53_pcp import _current_pcp_batch
    return _current_pcp_batch()


@contextmanager
def initial_profile_scope(runner, num_tokens, *, skip_attn, is_profile):
    """Permit only the native pre-cache memory-profile path, without metadata."""
    if not is_v41_pcp(runner.vllm_config) or not (skip_attn and is_profile):
        yield
        return
    require_v41_pcp_contract(runner.vllm_config)
    if getattr(runner, "_hcu_v41_attention_installed", False):
        # Installed attention wrappers provide the ordinary active row scope.
        yield
        return
    if (getattr(runner, "pcp_manager", None) is not None
            or getattr(runner, "kv_cache_config", None) is not None
            or _batch() is not None):
        raise RuntimeError("V41 initial profile requires the uninitialized cache lifecycle")
    budget = int(num_tokens)
    if budget <= 0 or budget > int(runner.max_num_tokens):
        raise RuntimeError("V41 initial profile token budget is invalid")
    token = _INITIAL_PROFILE.set((runner, budget))
    try:
        yield
    finally:
        _INITIAL_PROFILE.reset(token)


@contextmanager
def _logical_builder_scope(attn_groups, ubatch_idx):
    """Keep all native PCP builder state consistent for V41 full-cache rows."""
    fields = ("pcp_world_size", "use_pcp", "pcp_rank")
    saved, seen = [], set()
    try:
        for groups in attn_groups:
            for group in groups:
                builder = group.get_metadata_builder(ubatch_idx)
                if id(builder) in seen:
                    continue
                seen.add(id(builder))
                present = tuple(hasattr(builder, name) for name in fields)
                if not any(present):
                    continue
                if not all(present):
                    raise RuntimeError("V41 logical builder PCP state is incomplete")
                previous = tuple(getattr(builder, name) for name in fields)
                saved.append((builder, previous))
                builder.pcp_world_size, builder.use_pcp, builder.pcp_rank = 1, False, 0
        yield
    finally:
        for builder, previous in reversed(saved):
            builder.pcp_world_size, builder.use_pcp, builder.pcp_rank = previous


def _guarded_compressed_build(builder, original):
    @functools.wraps(original)
    def checked(common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
        count = int(common_attn_metadata.num_actual_tokens)
        slots = common_attn_metadata.slot_mapping
        queries = common_attn_metadata.query_start_loc
        output = builder.compressed_slot_mapping_buffer
        requests = int(common_attn_metadata.num_reqs)
        if (count < 0 or not isinstance(slots, torch.Tensor) or slots.ndim != 1
                or slots.dtype != torch.int64 or slots.numel() < count
                or not isinstance(queries, torch.Tensor) or queries.ndim != 1
                or queries.numel() < requests + 1
                or not isinstance(output, torch.Tensor) or output.ndim != 1
                or output.dtype != torch.int64 or output.numel() < count):
            raise RuntimeError("V41 compressed-slot metadata has insufficient rows or capacity")
        return original(common_prefix_len, common_attn_metadata, fast_build, **kwargs)
    return checked


@contextmanager
def _compressed_slot_guard_scope(attn_groups, ubatch_idx):
    """Guard CR2 helper extents only during the dedicated V41 metadata build."""
    sentinel, saved, seen = object(), [], set()
    try:
        for groups in attn_groups:
            for group in groups:
                builder = group.get_metadata_builder(ubatch_idx)
                if id(builder) in seen:
                    continue
                seen.add(id(builder))
                if int(getattr(builder, "compress_ratio", 0)) <= 1:
                    continue
                original = getattr(builder, "build", None)
                if not callable(original) or not hasattr(builder, "compressed_slot_mapping_buffer"):
                    raise RuntimeError("V41 CR2 metadata builder has no compressed-slot contract")
                previous = vars(builder).get("build", sentinel)
                saved.append((builder, previous))
                builder.build = _guarded_compressed_build(builder, original)
        yield
    finally:
        for builder, previous in reversed(saved):
            if previous is sentinel:
                delattr(builder, "build")
            else:
                builder.build = previous


def _snapshot_metadata(metadata, *inputs):
    """Clone builder scratch; retain caller-owned tensor and metadata identities."""
    memo, seen = {}, set()

    def preserve(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            memo[id(value)] = value
        elif isinstance(value, dict):
            for child in value.values():
                preserve(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                preserve(child)
        elif hasattr(value, "__dict__"):
            for child in vars(value).values():
                preserve(child)
    for value in inputs:
        preserve(value)
    # The final input is the caller's model-specific metadata object. Neither
    # native call may replace it, nor may snapshotting duplicate it.
    if inputs and inputs[-1] is not None:
        memo[id(inputs[-1])] = inputs[-1]
    return copy.deepcopy(metadata, memo)


def _local_slots(batch, global_slots):
    width = batch.local_batch.num_tokens_after_padding
    if batch.local_indices.shape != (width,):
        raise RuntimeError("V41 PCP local/global row map is incomplete")
    padding = batch.local_batch.is_padding[:width]
    if padding.shape != (width,) or padding.dtype != torch.bool:
        raise RuntimeError("V41 PCP requires the local fragment padding mask")
    # Unlike the durable-KV ownership map, replicated decode slots remain
    # valid on every peer: the outer view is consumed by local Engram only.
    slots = global_slots.index_select(1, batch.local_indices)
    slots.masked_fill_(padding.unsqueeze(0), -1)
    return slots


def _local_swa_groups(attn_groups, ubatch_idx):
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
    local = [[group for group in groups if isinstance(
        group.get_metadata_builder(ubatch_idx), DeepseekSparseSWAMetadataBuilder)]
        for groups in attn_groups]
    if not any(local):
        raise RuntimeError("V41 PCP requires local SWA metadata for Engram")
    return local


def _ensure_fragment_replay_capacity(attn_groups, num_reqs, ubatch_idx):
    """Only the V41 adapter's local build needs virtual-request zero starts."""
    for groups in attn_groups:
        for group in groups:
            builder = group.get_metadata_builder(ubatch_idx)
            starts = getattr(builder, "no_replay_start", None)
            if starts is not None:
                if (not isinstance(starts, torch.Tensor) or starts.ndim != 1
                        or starts.dtype != torch.int32):
                    raise RuntimeError("V41 PCP SWA requires an int32 default replay vector")
                if starts.numel() < num_reqs:
                    # Eager only; grow this instance once for the actual local
                    # request count. Never resize native physical-request state.
                    builder.no_replay_start = starts.new_zeros(num_reqs)


def _prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                  attn_groups, kv_cache_config, for_capture=False, ubatch_idx=0,
                  model_specific_attn_metadata=None):
    from vllm_hcu.model_executor.layers.attention.pcp import (
        logical_pcp_metadata_scope, replicated_mtp_batch_scope,
    )
    from vllm_hcu.patch.worker.framework_opt.patch_pcp_model_state import _attach_pcp_cache_ownership
    from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer

    original = self._hcu_v41_original_prepare_attn
    batch = _batch()
    if batch is None:
        with replicated_mtp_batch_scope(), logical_pcp_metadata_scope(1), _logical_builder_scope(attn_groups, ubatch_idx), _compressed_slot_guard_scope(attn_groups, ubatch_idx):
            result = original(input_batch, cudagraph_mode, block_tables, slot_mappings,
                              attn_groups, kv_cache_config, for_capture, ubatch_idx,
                              model_specific_attn_metadata)
        _attach_pcp_cache_ownership(result, False)
        return result
    if input_batch is not batch.local_batch or for_capture:
        raise RuntimeError("V41 PCP metadata requires the current eager local batch")
    caller_starts = getattr(model_specific_attn_metadata, "replay_start", None)
    if caller_starts is not None:
        needed = max(input_batch.num_reqs, batch.global_batch.num_reqs)
        if (not isinstance(caller_starts, torch.Tensor) or caller_starts.ndim != 1
                or caller_starts.dtype != torch.int32 or caller_starts.numel() < needed
                or caller_starts.device != input_batch.positions.device):
            raise RuntimeError("V41 PCP caller replay metadata has insufficient fragment capacity")
        torch._assert_async((caller_starts == 0).all(), "V41 PCP bounded replay is unsupported")
    global_tables, global_slots = batch.manager.prepare_global_attn()
    with replicated_mtp_batch_scope(), logical_pcp_metadata_scope(1), _logical_builder_scope(attn_groups, ubatch_idx), _compressed_slot_guard_scope(attn_groups, ubatch_idx):
        full = original(batch.global_batch, cudagraph_mode, global_tables, global_slots,
                        attn_groups, kv_cache_config, for_capture, ubatch_idx,
                        model_specific_attn_metadata)
    _attach_pcp_cache_ownership(full, False)
    full = _snapshot_metadata(full, batch.global_batch, global_tables, global_slots,
                              model_specific_attn_metadata)
    if getattr(self, "_replay", None) is not None and self._replay[0]:
        raise RuntimeError("V41 PCP bounded replay requires a separate fragment adapter")
    local_groups = _local_swa_groups(attn_groups, ubatch_idx)
    _ensure_fragment_replay_capacity(local_groups, input_batch.num_reqs, ubatch_idx)
    local_slots = _local_slots(batch, global_slots)
    with replicated_mtp_batch_scope(), logical_pcp_metadata_scope(1), _logical_builder_scope(local_groups, ubatch_idx), _compressed_slot_guard_scope(local_groups, ubatch_idx):
        local = original(input_batch, cudagraph_mode, block_tables, local_slots,
                         local_groups, kv_cache_config, for_capture, ubatch_idx,
                         model_specific_attn_metadata)
    _attach_pcp_cache_ownership(local, False)
    # Store on the step batch, never on VllmConfig (config.replace clones it).
    batch._hcu_v41_metadata = (local, full,
        build_slot_mappings_by_layer(global_slots, kv_cache_config))
    return local


@contextmanager
def _attention_scope(batch, rows):
    from vllm.forward_context import get_forward_context
    from vllm_hcu.model_executor.layers.attention.pcp import (
        logical_pcp_metadata_scope, replicated_mtp_batch_scope,
    )
    context = get_forward_context()
    saved = (context.attn_metadata, context.slot_mapping, context.is_padding)
    token = None
    try:
        if batch is not None:
            views = getattr(batch, "_hcu_v41_metadata", None)
            if views is None or context.attn_metadata is not views[0]:
                raise RuntimeError("V41 PCP attention has stale or missing metadata views")
            padding = batch.global_batch.is_padding[:rows]
            context.attn_metadata, context.slot_mapping = views[1], views[2]
            context.is_padding = padding
        token = _ACTIVE_ROWS.set((rows, context.attn_metadata))
        with replicated_mtp_batch_scope(), logical_pcp_metadata_scope(1):
            yield
    finally:
        if token is not None:
            _ACTIVE_ROWS.reset(token)
        context.attn_metadata, context.slot_mapping, context.is_padding = saved


def _preflight_attention(layer, hidden_states, positions):
    """Check structural contracts before any native cache/topk writer runs.

    This is not a transaction: backend numerical/device failures may occur
    after writes. Only facts available before the native body are checked.
    """
    from vllm.forward_context import get_forward_context
    context = get_forward_context()
    active = _ACTIVE_ROWS.get()
    rows = hidden_states.shape[0]
    if active is None or active[0] != rows or active[1] is not context.attn_metadata:
        raise RuntimeError("V41 attention writers require the active global row scope")
    if (positions.ndim != 1 or positions.shape[0] != rows
            or positions.dtype != torch.int64 or positions.device != hidden_states.device):
        raise RuntimeError("V41 attention positions do not match global query rows")
    ratio = layer.compress_ratio
    if ratio not in (0, 1, 2):
        raise RuntimeError("V41 attention compression schema is unsupported")
    indexer = layer.indexer
    if (layer.topk_indices_buffer is not getattr(layer, "_hcu_v41_shared_topk_buffer", None)
            or layer.candidate_block_buffer is not getattr(layer, "_hcu_v41_shared_candidate_buffer", None)):
        raise RuntimeError("V41 attention shared buffer bindings changed after installation")
    main_source = None
    if ratio > 0:
        main_prefix = getattr(layer, "_hcu_v41_main_cache_prefix", None)
        main_source = layer._static_forward_context.get(main_prefix)
        if (not isinstance(main_prefix, str) or layer.compressed_cache_prefix != main_prefix
                or main_source is None or getattr(main_source, "prefix", None) != main_prefix
                or not bool(getattr(main_source, "is_kv_source", False))
                or (layer.is_kv_source and main_source is not layer)):
            raise RuntimeError("V41 native main cache source ownership is invalid")
        if layer.compressor is not None and (
                not layer.is_kv_source or layer.compressor.k_cache_prefix != main_prefix
                or layer.compressor._static_forward_context.get(main_prefix) is not main_source):
            raise RuntimeError("V41 native compressor cache source ownership is invalid")
    expected_topk = (indexer.topk_tokens if indexer is not None
                     else getattr(layer, "_hcu_v41_topk_tokens", 0))

    def matrix(value, name, width, required):
        if value is None and not required:
            return
        if (not isinstance(value, torch.Tensor) or value.ndim != 2
                or value.dtype != torch.int32 or value.device != hidden_states.device
                or value.shape[0] < rows or value.shape[1] < width):
            raise RuntimeError("V41 attention shared " + name + " capacity/schema is invalid")
    matrix(layer.topk_indices_buffer, "topk", max(1, int(expected_topk)), ratio > 0)
    matrix(layer.candidate_block_buffer, "candidate",
           max(1, int(layer.candidate_topk_blocks)), False)
    if indexer is not None:
        expected_prefix = getattr(layer, "_hcu_v41_index_cache_prefix", None)
        if (bool(indexer.owns_k) != bool(layer.is_kv_source)
                or not isinstance(expected_prefix, str)
                or indexer.k_cache.prefix != expected_prefix
                or layer._static_forward_context.get(expected_prefix) is not indexer.k_cache):
            raise RuntimeError("V41 indexer native cache source ownership is invalid")
        if indexer.topk_indices_buffer is not layer.topk_indices_buffer:
            raise RuntimeError("V41 indexer does not own the attention shared topk buffer")
        op = indexer.indexer_op
        if (op.k_cache is not indexer.k_cache or not op.skip_k_cache_insert
                or op.topk_indices_buffer is not layer.topk_indices_buffer
                or (op.candidate_blocks is not None
                    and op.candidate_blocks is not layer.candidate_block_buffer)):
            raise RuntimeError("V41 indexer shared candidate/topk ownership is invalid")

    metadata = context.attn_metadata
    if metadata is None:
        # Installed dummy/profile attention retains its native allocation path.
        return
    if not isinstance(metadata, dict):
        raise RuntimeError("V41 attention requires one eager global metadata view")

    def vector(value, size, name, slot=False):
        allowed = (torch.int64,) if slot else (torch.int32, torch.int64)
        if (not isinstance(value, torch.Tensor) or value.ndim != 1
                or value.numel() < size or value.dtype not in allowed
                or value.device != hidden_states.device):
            raise RuntimeError("V41 attention " + name + " metadata rows/schema are invalid")

    def scalar(value, field, name):
        result = getattr(value, field, None)
        if not isinstance(result, Integral) or isinstance(result, bool) or result < 0:
            raise RuntimeError("V41 attention " + name + " metadata count is missing or invalid")
        return int(result)

    def entry(prefix, name, cache):
        if not isinstance(prefix, str) or prefix not in metadata:
            raise RuntimeError("V41 attention is missing global " + name + " metadata")
        value = metadata[prefix]
        vector(getattr(value, "slot_mapping", None), rows, name + " slot", slot=True)
        if int(getattr(value, "pcp_world_size", 1)) != 1:
            raise RuntimeError("V41 attention " + name + " metadata is not logical PCP1")
        actual = getattr(value, "num_actual_tokens", None)
        if actual is None and hasattr(value, "num_decode_tokens") and hasattr(value, "num_prefill_tokens"):
            actual = value.num_decode_tokens + value.num_prefill_tokens
        if actual is not None and int(actual) != rows:
            raise RuntimeError("V41 attention " + name + " metadata token count is invalid")
        if not isinstance(cache, torch.Tensor) or cache.numel() == 0:
            raise RuntimeError("V41 attention " + name + " cache is not materialized")
        return value

    swa = entry(layer.swa_cache_layer.prefix, "SWA", layer.swa_cache_layer.kv_cache)
    requests = scalar(swa, "num_decodes", "SWA") + scalar(swa, "num_prefills", "SWA")
    if scalar(swa, "num_decode_tokens", "SWA") + scalar(swa, "num_prefill_tokens", "SWA") != rows:
        raise RuntimeError("V41 attention SWA metadata token count is invalid")
    vector(getattr(swa, "query_start_loc", None), requests + 1, "SWA query")
    vector(getattr(swa, "seq_lens", None), requests, "SWA sequence")
    if (not isinstance(swa.block_table, torch.Tensor) or swa.block_table.ndim != 2
            or swa.block_table.shape[0] < requests or swa.block_table.device != hidden_states.device):
        raise RuntimeError("V41 attention SWA block-table rows are invalid")
    if ratio > 0:
        prefix = layer._hcu_v41_main_cache_prefix
        main = entry(prefix, "main", getattr(main_source, "kv_cache", None))
        vector(getattr(main, "query_start_loc", None), requests + 1, "main query")
        if scalar(main, "num_reqs", "main") != requests or scalar(main, "num_actual_tokens", "main") != rows:
            raise RuntimeError("V41 attention main request/token rows are invalid")
    if indexer is not None:
        index_metadata = entry(indexer.k_cache.prefix, "index", indexer.k_cache.kv_cache)
        if (scalar(index_metadata, "num_decode_tokens", "index") + scalar(index_metadata, "num_prefill_tokens", "index") != rows
                or not indexer.indexer_op.skip_k_cache_insert):
            raise RuntimeError("V41 attention index source rows/mode are invalid")
    compressor = layer.compressor
    if compressor is not None and compressor.state_cache is not None:
        state = entry(compressor.state_cache.prefix, "compressor state", compressor.state_cache.kv_cache)
        vector(getattr(state, "query_start_loc", None), requests + 1, "compressor query")
        vector(getattr(state, "token_to_req_indices", None), rows, "compressor token map")


def _forward(self, positions, hidden_states, llama_4_scaling=None):
    batch = _batch()
    if batch is None:
        full_hidden, full_positions = hidden_states, positions
    else:
        width = batch.local_batch.num_tokens_after_padding
        if hidden_states.shape[0] != width or positions.shape[0] != width:
            raise RuntimeError("V41 PCP attention input does not match its local fragment rows")
        count = batch.global_batch.num_tokens
        # Decode rows are already replicated; collecting them again corrupts
        # their row order and adds an unnecessary collective.
        full_hidden = (batch.gather(hidden_states) if batch.manager._global_has_prefill
                       else hidden_states[:count])
        if full_hidden.shape[0] != count:
            raise RuntimeError("V41 PCP restore did not produce the complete global batch")
        full_positions = batch.global_batch.positions[:count]
    aux_streams = self.aux_stream_list
    self.aux_stream_list = None
    try:
        with _attention_scope(batch, full_hidden.shape[0]):
            _preflight_attention(self, full_hidden, full_positions)
            # Native ROCm's newer CSA2 path contains an unadapted weight.T GEMM.
            # Call the audited serial base body; virtual methods still dispatch
            # to the native ROCm insert/norm/output implementations.
            output = self._hcu_v41_serial_forward(full_positions, full_hidden, llama_4_scaling)
    finally:
        self.aux_stream_list = aux_streams
    if batch is None:
        return output
    if not isinstance(output, torch.Tensor) or output.shape[0] != batch.global_batch.num_tokens:
        raise RuntimeError("V41 PCP native attention returned an invalid global output")
    local = batch.localize(output)
    local.masked_fill_(batch.local_batch.is_padding[:local.shape[0], None], 0)
    return local


def logical_indexer_pcp_width(config, k_cache, skip_k_cache_insert, compress_ratio):
    parallel = config.parallel_config
    physical = int(parallel.prefill_context_parallel_size)
    if not is_v41_pcp(config):
        return physical
    require_v41_pcp_contract(config)
    if (not skip_k_cache_insert or compress_ratio not in (1, 2)
            or type(k_cache).__module__ != "vllm.models.deepseek_v41.attention"
            or type(k_cache).__name__ != "DeepseekV4IndexerCache"):
        raise ValueError("V41 PCP requires the native source-owned FP8 index cache")
    return 1


def require_global_indexer_rows(indexer, hidden_states, q, k, weights):
    """Constructor admission is insufficient: every execution needs this scope."""
    from vllm.forward_context import get_forward_context
    active = _ACTIVE_ROWS.get()
    context = get_forward_context()
    if active is not None:
        if active[1] is not context.attn_metadata:
            raise RuntimeError("V41 PCP indexer requires the active global attention scope")
        rows = active[0]
    else:
        profile = _INITIAL_PROFILE.get()
        if profile is None:
            raise RuntimeError("V41 PCP indexer requires the active global attention scope")
        runner, rows = profile
        slots = context.slot_mapping
        cache = getattr(indexer.k_cache, "kv_cache", None)
        if (getattr(runner, "_hcu_v41_attention_installed", False)
                or getattr(runner, "pcp_manager", None) is not None
                or getattr(runner, "kv_cache_config", None) is not None
                or context.attn_metadata is not None
                or (slots is not None and (not isinstance(slots, dict) or len(slots)))
                or not isinstance(cache, torch.Tensor) or cache.numel() != 0):
            raise RuntimeError("V41 initial profile requires no metadata and an unbound cache")
        # No metadata or active attention scope is fabricated here. The caller
        # continues through the unchanged opaque op to its native profile
        # allocator, reserving flattened KV and padded Top-K workspaces.
    if (not isinstance(q, torch.Tensor) or q.shape[0] != rows
            or hidden_states.shape[0] != rows or weights.shape[0] != rows
            or k is not None or not indexer.skip_k_cache_insert):
        raise RuntimeError("V41 PCP Q/weights/source-cache rows are not globally aligned")
    if rows > indexer.topk_indices_buffer.shape[0]:
        raise RuntimeError("V41 PCP global queries exceed the shared topk capacity")
    if indexer.candidate_blocks is not None and rows > indexer.candidate_blocks.shape[0]:
        raise RuntimeError("V41 PCP global queries exceed the shared candidate capacity")
    if isinstance(context.attn_metadata, dict):
        metadata = context.attn_metadata[indexer.k_cache.prefix]
        if (int(getattr(metadata, "pcp_world_size", 1)) != 1
                or metadata.num_decode_tokens + metadata.num_prefill_tokens != rows):
            raise RuntimeError("V41 PCP indexer requires complete logical-PCP1 metadata")


@functools.lru_cache(None)
def _replicated_backend(base):
    class V41FullCachePCPBackend(base):
        @classmethod
        def supports_pcp(cls):
            return True
    return V41FullCachePCPBackend


def _install_cache_backend(cache):
    if getattr(cache, "_hcu_v41_original_backend", None) is None:
        cache._hcu_v41_original_backend = cache.get_attn_backend
        backend = _replicated_backend(cache.get_attn_backend())
        cache.get_attn_backend = MethodType(lambda self: backend, cache)


def _require_sources(functions):
    from vllm_hcu.patch.worker.core_fix._common import PatchCompatibilityError
    for name, function in functions.items():
        try:
            actual = hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode()).hexdigest()
        except (OSError, TypeError) as exc:
            raise PatchCompatibilityError("cannot audit V41 PCP source " + name) from exc
        if actual != _AUDITED_SOURCES[name]:
            raise PatchCompatibilityError("unreviewed V41 PCP source " + name + ": " + actual)


def install_v41_pcp(runner):
    """Install before backend discovery; PCP1 and unrelated models are untouched."""
    from vllm_hcu.patch.worker.core_fix._common import (
        PatchCompatibilityError, load_exact_module, require_exact_signature,
    )
    from vllm_hcu.patch.config import get_hcu_config
    require_v41_pcp_contract(runner.vllm_config)
    feature = get_hcu_config(runner.vllm_config)
    if (feature.enable_dsa_cache_layer_split or feature.enable_lightly_cp
            or feature.enable_multi_layers_mtp):
        raise ValueError("V41 PCP does not support layer-split, lightly-CP or multi-layer MTP")
    state_module = load_exact_module("vllm.models.deepseek_v41.nvidia.model_state", None)
    attention = load_exact_module("vllm.models.deepseek_v41.attention", None)
    compressor = load_exact_module("vllm.models.deepseek_v41.compressor", None)
    rocm = load_exact_module("vllm.models.deepseek_v41.amd.rocm", None)
    state = runner.model_state
    if not isinstance(state, state_module.DeepseekV41ModelState):
        raise PatchCompatibilityError("V41 PCP requires the native V41 model state")
    if state.decoder_replay_layers is not None:
        raise ValueError("V41 PCP decoder replay requires a separate sub-batch adapter")
    if hasattr(state, "_hcu_glm53_original_prepare_attn"):
        raise PatchCompatibilityError("V41 PCP cannot stack on the V4 global-only metadata adapter")
    functions = {
        "DeepseekV4Attention.forward": attention.DeepseekV4Attention.forward,
        "DeepseekV4Attention._prepare_and_attn": attention.DeepseekV4Attention._prepare_and_attn,
        "DeepseekV4Indexer.forward": attention.DeepseekV4Indexer.forward,
        "DeepseekV4Indexer._produce_k": attention.DeepseekV4Indexer._produce_k,
        "DeepseekCompressor.forward": compressor.DeepseekCompressor.forward,
        "DeepseekCompressor.insert_cache": compressor.DeepseekCompressor.insert_cache,
        "DeepseekV41ModelState.prepare_attn": state_module.DeepseekV41ModelState.prepare_attn,
    }
    _require_sources(functions)
    require_exact_signature(functions["DeepseekV4Attention.forward"], "V41Attention.forward",
        positional=("self", "positions", "hidden_states", "llama_4_scaling"), defaults={"llama_4_scaling": None})
    require_exact_signature(functions["DeepseekV41ModelState.prepare_attn"], "V41ModelState.prepare_attn",
        positional=("self", "input_batch", "cudagraph_mode", "block_tables", "slot_mappings", "attn_groups", "kv_cache_config", "for_capture", "ubatch_idx", "model_specific_attn_metadata"),
        defaults={"for_capture": False, "ubatch_idx": 0, "model_specific_attn_metadata": None})
    layers = [layer for layer in runner.model.modules()
              if isinstance(layer, rocm.DeepseekV41ROCMAiterMLAAttention)]
    if not layers:
        raise PatchCompatibilityError("V41 PCP found no native ROCm attention layers")
    topk_sizes = {int(layer.indexer.topk_tokens) for layer in layers if layer.indexer is not None}
    if len(topk_sizes) > 1:
        raise PatchCompatibilityError("V41 PCP indexers disagree on the shared topk width")
    topk_size = next(iter(topk_sizes), 0)
    if not hasattr(state, "_hcu_v41_original_prepare_attn"):
        state._hcu_v41_original_prepare_attn = state.prepare_attn
        state.prepare_attn = MethodType(_prepare_attn, state)
    elif getattr(state.prepare_attn, "__func__", None) is not _prepare_attn:
        raise PatchCompatibilityError("stale V41 PCP model-state adapter")
    for layer in layers:
        layer._hcu_v41_topk_tokens = topk_size
        layer._hcu_v41_shared_topk_buffer = layer.topk_indices_buffer
        layer._hcu_v41_shared_candidate_buffer = layer.candidate_block_buffer
        main_prefix = None
        if layer.compress_ratio > 0:
            main_prefix = (layer.prefix if layer.is_kv_source else
                attention._replace_layer_index(layer.prefix, layer.kv_source_layer_id))
            if layer.compressed_cache_prefix != main_prefix:
                raise PatchCompatibilityError("V41 native compressed-cache source prefix is inconsistent")
        layer._hcu_v41_main_cache_prefix = main_prefix
        if layer.indexer is not None:
            expected_prefix = main_prefix + ".indexer.k_cache"
            if (bool(layer.indexer.owns_k) != bool(layer.is_kv_source)
                    or layer.indexer.k_cache.prefix != expected_prefix
                    or layer._static_forward_context.get(expected_prefix) is not layer.indexer.k_cache):
                raise PatchCompatibilityError("V41 native index-cache source ownership is inconsistent")
            layer._hcu_v41_index_cache_prefix = expected_prefix
        caches = [layer, layer.swa_cache_layer]
        if layer.indexer is not None:
            caches.append(layer.indexer.k_cache)
        if layer.compressor is not None and layer.compressor.state_cache is not None:
            caches.append(layer.compressor.state_cache)
        for cache in caches:
            _install_cache_backend(cache)
        if not getattr(layer, _MARKER, False):
            layer._hcu_v41_original_forward = layer.forward
            layer._hcu_v41_serial_forward = MethodType(attention.DeepseekV4Attention.forward, layer)
            layer.forward = MethodType(_forward, layer)
            setattr(layer, _MARKER, True)
        elif getattr(layer.forward, "__func__", None) is not _forward:
            raise PatchCompatibilityError("stale V41 PCP attention adapter")
    runner._hcu_v41_attention_installed = True
    return len(layers)
