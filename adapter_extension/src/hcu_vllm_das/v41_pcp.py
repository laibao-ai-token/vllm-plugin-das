# SPDX-License-Identifier: Apache-2.0
"""Eager HCU V41 PCP attention using native replicated-cache preparation.

Only owned classes advertise PCP. Common V41 produces rank-major cache payloads
and local queries; AMD owns attention and inverse RoPE. No Core symbols change.
"""
from __future__ import annotations

import inspect
import torch
from vllm.forward_context import get_forward_context
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.models.deepseek_v41.attention import DeepseekV4Attention, DeepseekV4Indexer
from vllm.models.deepseek_v41.amd.rocm import (
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterSparseSWABackend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from . import v41_indexer as reference
from .quantization.dsv41_channel_fp8 import _is_v41


def validate_config(cfg) -> bool:
    """Validate the native PCP structure before allocating model parameters.

    Width is unconstrained. PP/DP/DCP=1 follows the native V41 PCP contract;
    DSpark uses the owned CR2 history metadata for rejection rollback.
    Engram uses the owned private registered-host TPxPCP table adapter.
    """
    p = cfg.parallel_config
    if p.prefill_context_parallel_size == 1:
        return False
    errors = []
    if not _is_v41(cfg.model_config.hf_config):
        errors.append('V41 model family required')
    if p.prefill_context_parallel_size < 2:
        errors.append('positive PCP width required')
    if not cfg.model_config.enforce_eager:
        errors.append('eager execution required by host-driven reference scoring')
    for name in ('pipeline_parallel_size', 'data_parallel_size', 'decode_context_parallel_size'):
        if getattr(p, name) != 1:
            errors.append(f'{name}=1 required by native V41 PCP metadata')
    if cfg.speculative_config is not None:
        from .pcp_dspark import validate_dspark_config
        try:
            validate_dspark_config(cfg)
        except ValueError as error:
            errors.append(str(error))
    if cfg.lora_config is not None or p.enable_eplb:
        errors.append('LoRA/EPLB are not supported by this PCP model adapter')
    if p.enable_dbo:
        errors.append('DBO would change per-step metadata/collective ordering')
    if not cfg.use_v2_model_runner:
        errors.append('model runner V2 PCP metadata required')
    if cfg.kernel_config.monokernel:
        errors.append('monokernel bypasses common PCP preparation')
    if cfg.kv_transfer_config is not None:
        errors.append('KV transfer requires a separate replicated-cache contract')
    # SWA bounded replay: native decoder replay pads each rank's replay batch to
    # the global kept rows and rebuilds PCP metadata for it; this attention reads
    # only that metadata. Prefix-cache SWA replay stays off on ROCm natively.
    if getattr(cfg.model_config.hf_config, 'engram_layer_ids', None):
        from .v41_engram import validate_engram_config
        if not hasattr(cfg, 'engram_config'):
            errors.append('HCU PCP Engram requires resolved Engram config')
        else:
            try:
                validate_engram_config(cfg)
            except ValueError as error:
                errors.append(str(error))
    if (cfg.attention_config.resolve_indexer_kv_dtype('fp8') != 'fp8'
            or cfg.attention_config.indexer_sparse_logits):
        errors.append('source-owned FP8 indexer with dense logits required')
    if cfg.attention_config.backend not in (
        None, AttentionBackendEnum.ROCM_FLASHMLA_SPARSE_DSV4,
        AttentionBackendEnum.ROCM_FLASHMLA_SPARSE_DSV41,
    ):
        errors.append('ROCM_FLASHMLA_SPARSE_DSV41 backend required')
    if errors:
        raise ValueError('HCU V41 PCP attention: ' + '; '.join(errors))
    return True


class HcuV41PcpBackend(DeepseekV41ROCMAiterMLASparseBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return True


class HcuV41PcpSwaBackend(DeepseekV41ROCMAiterSparseSWABackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return True


class HcuV41PcpSparseIndexer(SparseAttnIndexer):
    """Local scoring over the full cache replicated by common V41 sources.

    This is an owned layer selected by an owned model, independent of the
    optional PCP1 OOT scorer. Never falsify the native parallel configuration.
    """
    def __init__(self, *args, **kwargs):
        validate_config(get_current_vllm_config())
        bound = inspect.signature(SparseAttnIndexer.__init__).bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if (not values['skip_k_cache_insert'] or values['use_fp4_cache']
                or values['head_dim'] != 128 or values['compress_ratio'] not in (1, 2)):
            raise ValueError('PCP indexer requires source-owned FP8 K, D128, CR1/2')
        super().__init__(*args, **kwargs)

    def forward_hip(self, hidden_states, q_quant, k, weights):
        if k is not None or not isinstance(q_quant, torch.Tensor):
            raise ValueError('PCP scoring expects one FP8 Q and source-owned K')
        return reference.indexer(
            hidden_states, self.k_cache.prefix, self.k_cache.kv_cache, q_quant,
            weights, self.topk_tokens, self.head_dim, self.max_model_len,
            self.max_total_seq_len, self.topk_indices_buffer, self.compress_ratio,
            self.candidate_blocks, self.candidate_block_size, self.candidate_write,
        )

    # CustomOp disables hardware dispatch when the user excludes its op name.
    # The owned model still needs the same scorer in that configuration.
    forward_native = forward_hip


class HcuV41PcpIndexer(DeepseekV4Indexer):
    attn_cls = HcuV41PcpSparseIndexer


class HcuV41PcpAttention(DeepseekV41ROCMAiterMLAAttention):
    backend_cls = HcuV41PcpBackend
    swa_backend_cls = HcuV41PcpSwaBackend

    def __init__(self, *args, **kwargs):
        cfg = args[0] if args else kwargs['vllm_config']
        if not validate_config(cfg):
            raise ValueError('owned PCP attention requires PCP greater than one')
        super().__init__(*args, **kwargs)
        # Common preparation serializes PCP gathers. AMD's fast compressor
        # chain publishes local latent against global slots and cannot run here.
        self.aux_stream_list = None

    def _indexer_cls(self, k_cache):
        if k_cache is not None and k_cache.rocm_mxfp4:
            raise ValueError('PCP adapter requires the FP8 SHUFFLE index K layout')
        return HcuV41PcpIndexer

    forward = DeepseekV4Attention.forward
    _alloc_attn_out = DeepseekV4Attention._alloc_attn_out

    def forward_mqa(self, q, kv, positions, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError('HCU PCP attention requires its BF16 output allocation')
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return super().forward_mqa(q, kv, positions, output)
        if not isinstance(metadata, dict) or self.swa_cache_layer.prefix not in metadata:
            raise RuntimeError('HCU PCP attention requires native local SWA metadata')
        swa = metadata[self.swa_cache_layer.prefix]
        count = swa.num_decode_tokens + swa.num_prefill_tokens
        if count < 0 or count > q.shape[0] or q.shape[0] != output.shape[0]:
            raise ValueError('PCP local query/output extent mismatch')
        # Collectives already ran during preparation, including on empty ranks.
        # AMD inverse RoPE sees only initialized, actual local query rows.
        output[count:].zero_()
        if count:
            return super().forward_mqa(q[:count], kv[:count], positions[:count], output[:count])

    def _forward_decode(self, q, *args, **kwargs):
        # A native PCP placeholder may retain a request with zero actual tokens.
        if q.shape[0] == 0:
            return 0
        return super()._forward_decode(q, *args, **kwargs)
