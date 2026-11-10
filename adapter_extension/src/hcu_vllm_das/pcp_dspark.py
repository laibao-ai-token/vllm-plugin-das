# SPDX-License-Identifier: Apache-2.0
"""PCP CR2 history for eager V41 DSpark verification and rejection rollback."""
from __future__ import annotations

import hashlib
import inspect

import torch
from vllm.models.deepseek_v41 import attention, compressor
from vllm.v1.kv_cache_interface import CircularBufferSpec

from .v41_pcp import HcuV41PcpAttention
from .models.v41_pcp import _constructor


def validate_dspark_config(cfg):
    spec = cfg.speculative_config
    if getattr(spec, 'method', None) != 'dspark':
        raise ValueError('owned PCP supports DSpark speculative decoding only')
    draft = getattr(spec, 'draft_model_config', None)
    if draft is None or not getattr(draft, 'enforce_eager', False):
        raise ValueError('PCP DSpark requires an eager draft model')
    if cfg.parallel_config.tensor_parallel_size != 1:
        raise ValueError('PCP DSpark requires TP1')
    if getattr(draft.hf_config, 'model_type', None) not in (
        'deepseek_v41', 'deepseek_v41_text',
    ):
        raise ValueError('PCP DSpark requires the V41 draft model')
    from .dspark_agrs import token_capacity
    token_capacity(cfg)


_BUILDER_INIT = compressor.CompressorMetadataBuilder.__init__
_BUILDER_HASH = 'ea5840880e19becfd81c87c37866b76a6e8595215c33ba9e4fd8a01ac65f2722'
if hashlib.sha256(inspect.getsource(_BUILDER_INIT).encode()).hexdigest() != _BUILDER_HASH:
    raise RuntimeError('PCP DSpark compressor metadata constructor changed')


class HcuPcpDSparkCompressorMetadataBuilder(compressor.CompressorMetadataBuilder):
    def __init__(self, *args, **kwargs):
        # The inherited metadata build and ring-slot kernel remain native. Own
        # the buffers and PCP history policy instead of invoking its open-row
        # only constructor, which deliberately rejects speculative decoding.
        super(compressor.CompressorMetadataBuilder, self).__init__(*args, **kwargs)
        validate_dspark_config(self.vllm_config)
        if not isinstance(self.kv_cache_spec, CircularBufferSpec):
            raise ValueError('PCP DSpark compressor requires a circular buffer')
        self.capacity = self.kv_cache_spec.block_size
        required = self.vllm_config.speculative_config.num_speculative_tokens + 2
        if self.capacity < required:
            raise ValueError('PCP DSpark ring cannot retain the verification window')
        self.pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.token_to_req_indices = torch.zeros(
            max_tokens, dtype=torch.int32, device=self.device)
        self.slot_mapping_buffer = torch.empty(
            max_tokens, dtype=torch.int64, device=self.device)

    def _link_pcp_chunks(self, metadata, num_reqs, positions):
        """Replicate the latest full ring window, including rejected draft rows.

        A later verify can restart behind the previous end position. Keeping
        only the newest open pair loses its predecessor after rejection. Each
        rank therefore supplies a bounded tail from every local chunk. Only
        the globally newest window per ring block is written, with duplicate
        positions assigned one writer.
        """
        num_tokens = metadata.slot_mapping.numel()
        num_rows = min(num_tokens, 2 * self.vllm_config.scheduler_config.max_num_seqs)
        num_reqs = min(num_reqs, num_rows)
        query_start = metadata.query_start_loc.long()
        starts = query_start[:num_reqs].clamp(max=num_tokens - 1)
        ends = query_start[1:num_reqs + 1]
        nonempty = ends > query_start[:num_reqs]
        offsets = torch.arange(self.capacity, device=self.device)
        tail = ends[:, None] - self.capacity + offsets
        valid = nonempty[:, None] & (tail >= query_start[:num_reqs, None])
        tail = tail.clamp(min=0, max=num_tokens - 1)

        gather_tokens = torch.zeros(
            (num_rows, self.capacity), dtype=torch.long, device=self.device)
        gather_tokens[:num_reqs] = tail
        local_keys = torch.full_like(gather_tokens, -1)
        slots = metadata.slot_mapping[tail]
        keys = ((slots // self.capacity) << 32) | positions[tail]
        local_keys[:num_reqs] = torch.where(valid & (slots >= 0), keys, -1)
        gathered = compressor.get_pcp_group().all_gather(local_keys.flatten(), dim=0)
        sorted_keys, order = gathered.sort()
        blocks = sorted_keys >> 32
        sorted_positions = sorted_keys & 0xFFFFFFFF

        last_in_block = torch.ones_like(sorted_keys, dtype=torch.bool)
        last_in_block[:-1] = blocks[:-1] != blocks[1:]
        end_keys = torch.where(last_in_block, sorted_keys, torch.iinfo(torch.int64).max)
        newest_keys = end_keys.flip(0).cummin(0).values.flip(0)
        newest_positions = newest_keys & 0xFFFFFFFF
        unique = torch.ones_like(sorted_keys, dtype=torch.bool)
        unique[:-1] = sorted_keys[:-1] != sorted_keys[1:]
        keep = ((sorted_keys >= 0) & unique
                & (sorted_positions > newest_positions - self.capacity))
        ring_write_slots = torch.empty_like(sorted_keys)
        ring_write_slots[order] = torch.where(
            keep, blocks * self.capacity + sorted_positions % self.capacity, -1)

        first_positions = positions[starts]
        first_slots = metadata.slot_mapping[starts]
        wanted = ((first_slots // self.capacity) << 32) | (first_positions - 1)
        found = torch.searchsorted(sorted_keys, wanted).clamp(max=order.numel() - 1)
        has_prev = (nonempty & (first_slots >= 0) & (first_positions % 2 == 1)
                    & (sorted_keys[found] == wanted))
        metadata.gather_tokens = gather_tokens.flatten()
        metadata.ring_write_slots = ring_write_slots
        metadata.prev_row_indices = torch.where(has_prev, order[found], -1)


class HcuPcpDSparkCompressorBackend(compressor.CompressorBackend):
    @staticmethod
    def get_name():
        return 'HcuPcpDSparkCompressorBackend'

    @staticmethod
    def get_builder_cls():
        return HcuPcpDSparkCompressorMetadataBuilder


class HcuPcpDSparkStateCache(compressor.CompressorStateCache):
    def get_attn_backend(self):
        return HcuPcpDSparkCompressorBackend


class HcuPcpDSparkCompressor(compressor.DeepseekCompressor):
    __init__ = _constructor(
        compressor.DeepseekCompressor.__init__,
        '25653552a3c582786c1fe575416ee55547d09f1910396be808e609e5ae515be3',
        {'CompressorStateCache': HcuPcpDSparkStateCache})


class _PcpDSparkAttentionBase(attention.DeepseekV4Attention):
    __init__ = _constructor(
        attention.DeepseekV4Attention.__init__,
        '3763cdcf7175cc9938fca5b7346d58a316fd7e9252f0adfea68ace3ddc2f6e4d',
        {'DeepseekCompressor': HcuPcpDSparkCompressor})


class HcuV41PcpDSparkAttention(HcuV41PcpAttention, _PcpDSparkAttentionBase):
    """Keep native AMD execution with the owned CR2 metadata backend."""
