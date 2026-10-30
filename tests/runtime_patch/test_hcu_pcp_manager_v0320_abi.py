# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable CPU ABI checks using pinned native AST definitions, no vLLM imports.

Snapshots below are checked against fixed native source by the handoff probe.
Framework configuration, process groups and block tables are controlled CPU
boundaries. InputBatch/InputBuffers, descriptor, graph enum, ParallelConfig
capability property, native factory and native make_dummy retain their bodies.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import enum
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import NamedTuple
from uuid import uuid4

import numpy as np
import pytest
import torch

NATIVE_ENUM_SOURCE = 'class CUDAGraphMode(enum.Enum):\n    """Constants for the cudagraph mode in CompilationConfig.\n    Meanwhile, the subset enum `NONE`, `PIECEWISE` and `FULL` are also\n    treated as concrete runtime mode for cudagraph runtime dispatching.\n    """\n\n    NONE = 0\n    PIECEWISE = 1\n    FULL = 2\n    FULL_DECODE_ONLY = (FULL, NONE)\n    FULL_AND_PIECEWISE = (FULL, PIECEWISE)\n\n    def decode_mode(self) -> "CUDAGraphMode":\n        return CUDAGraphMode(self.value[0]) if self.separate_routine() else self\n\n    def mixed_mode(self) -> "CUDAGraphMode":\n        return CUDAGraphMode(self.value[1]) if self.separate_routine() else self\n\n    def has_mode(self, mode: "CUDAGraphMode") -> bool:\n        assert not mode.separate_routine()\n        if self.separate_routine():\n            return mode.value in self.value\n        return self == mode\n\n    def requires_piecewise_compilation(self) -> bool:\n        return self.has_mode(CUDAGraphMode.PIECEWISE)\n\n    def without_piecewise(self) -> "CUDAGraphMode":\n        if self == CUDAGraphMode.PIECEWISE:\n            return CUDAGraphMode.NONE\n        if self == CUDAGraphMode.FULL_AND_PIECEWISE:\n            return CUDAGraphMode.FULL_DECODE_ONLY\n        return self\n\n    def max_cudagraph_mode(self) -> "CUDAGraphMode":\n        return CUDAGraphMode(max(self.value)) if self.separate_routine() else self\n\n    def has_full_cudagraphs(self) -> bool:\n        return self.max_cudagraph_mode() == CUDAGraphMode.FULL\n\n    def has_piecewise_cudagraphs(self) -> bool:\n        return self.requires_piecewise_compilation()\n\n    def separate_routine(self) -> bool:\n        return isinstance(self.value, tuple)\n\n    @classmethod\n    def valid_runtime_modes(cls) -> frozenset["CUDAGraphMode"]:\n        return frozenset({cls.NONE, cls.PIECEWISE, cls.FULL})\n\n    def is_valid_runtime_mode(self) -> bool:\n        return self in CUDAGraphMode.valid_runtime_modes()\n\n    def __str__(self) -> str:\n        return self.name\n\n    def __bool__(self) -> bool:\n        return self != CUDAGraphMode.NONE\n'

NATIVE_INPUT_SOURCE = 'class InputBuffers:\n    def __init__(self, max_num_reqs: int, max_num_tokens: int, device: torch.device):\n        self.max_num_reqs = max_num_reqs\n        self.max_num_tokens = max_num_tokens\n        self.device = device\n\n        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)\n        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)\n        self.is_padding = torch.zeros(max_num_tokens, dtype=torch.bool, device=device)\n        self.query_start_loc = torch.zeros(\n            max_num_reqs + 1, dtype=torch.int32, device=device\n        )\n        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)\n        # DCP: per-request local seq_lens buffer\n        self.dcp_local_seq_lens = torch.zeros(\n            max_num_reqs, dtype=torch.int32, device=device\n        )\n\n\n@dataclass\nclass InputBatch:\n    # batch_idx -> req_id\n    req_ids: list[str]\n    num_reqs: int\n    num_reqs_after_padding: int\n\n    # batch_idx -> req_state_idx\n    idx_mapping: torch.Tensor\n    idx_mapping_np: np.ndarray\n    # Identical to idx_mapping except for spec decoding.\n    expanded_idx_mapping: torch.Tensor\n    # [total_num_logits] position within request for each logit\n    expanded_local_pos: torch.Tensor\n\n    # [num_reqs]\n    # batch_idx -> num_scheduled_tokens, (upper bound when using adaptive verification)\n    num_scheduled_tokens: np.ndarray\n    # number of tokens in the batch,\n    #  may be < sum(num_scheduled_tokens) when using adaptive verification\n    num_tokens: int\n    num_tokens_after_padding: int\n    # Sum of draft tokens scheduled across requests.\n    num_draft_tokens: int\n    # [num_reqs] number of draft tokens scheduled for each request, if any.\n    num_draft_tokens_per_req: np.ndarray | None\n\n    # [num_reqs + 1]\n    query_start_loc: torch.Tensor\n    query_start_loc_np: np.ndarray\n    # [num_reqs]\n    seq_lens: torch.Tensor\n    # [num_reqs] CPU upper bound on seq_lens (see CommonAttentionMetadata).\n    seq_lens_cpu_upper_bound: torch.Tensor\n    # [num_reqs]\n    dcp_local_seq_lens: torch.Tensor | None\n    # [num_reqs]\n    num_computed_tokens_np: np.ndarray\n    # [num_reqs]\n    prefill_len_np: np.ndarray\n    # [num_reqs]\n    num_computed_prefill_tokens_np: np.ndarray\n    # [num_reqs] CPU bool array == (num_computed_prefill_tokens_np < prefill_len_np).\n    is_prefilling_np: np.ndarray\n    # == np.any(is_prefilling_np)\n    has_prefill: bool\n    # No prefills, or only prefill rows that run as decodes.\n    decode_graph_eligible: bool\n\n    # [num_tokens_after_padding]\n    input_ids: torch.Tensor\n    # [num_tokens_after_padding]\n    positions: torch.Tensor\n    # [num_tokens_after_padding]\n    is_padding: torch.Tensor\n\n    # [total_num_logits]\n    logits_indices: torch.Tensor\n    # [num_reqs + 1]\n    cu_num_logits: torch.Tensor\n    cu_num_logits_np: np.ndarray\n\n    # Whether any requests in batch use structured output.\n    has_structured_output_reqs: bool\n\n    # [num_reqs] per-request prompt length, only populated for R-SWA.\n    prompt_lens: torch.Tensor | None\n\n    # [num_reqs] only populated for non-dummy pipeline-parallel batches.\n    max_seq_len_np: np.ndarray | None = None\n\n    # Longest query the batch may contain. Set when a cudagraph descriptor promises\n    # a query length this batch\'s own split does not reach, so attention metadata\n    # stays valid for every replay the graph serves.\n    max_query_len: int | None = None\n\n    # Arms the KV-sharing fast prefill path for this step. Absent for dummy\n    # (cudagraph capture) batches, which run the KV-sharing layers in full.\n    fast_prefill: "FastPrefillBatchMetadata | None" = None\n\n    # [num_reqs] set only under PCP+DCP (see CommonAttentionMetadata).\n    dcp_local_seq_lens_cpu_upper_bound: torch.Tensor | None = None\n\n    # [num_reqs] prefilling rows that schedule one new prompt token (excluding\n    # drafts) over existing context and so compute exactly like decodes.\n    # None if there are no prefills.\n    prefill_runs_as_decode_np: np.ndarray | None = None\n\n    @classmethod\n    def make_dummy(\n        cls,\n        num_reqs: int,\n        num_tokens: int,\n        input_buffers: InputBuffers,\n        max_query_len: int | None = None,\n        is_padding: bool = True,\n    ) -> "InputBatch":\n        assert 0 < num_reqs <= num_tokens\n        device = input_buffers.device\n\n        req_ids = [f"req_{i}_{random_uuid()}" for i in range(num_reqs)]\n        idx_mapping_np = np.arange(num_reqs, dtype=np.intp)\n        idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device=device)\n        expanded_idx_mapping = idx_mapping\n        expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=device)\n\n        # Distribute the remainder evenly so that no dummy request exceeds\n        # ceil(num_tokens / num_reqs) <= max_model_len tokens. Varlen graphs\n        # accept any split with non-empty slots, so this shape works for them\n        # too; attention metadata is built from the promised max_query_len.\n        base_tokens = num_tokens // num_reqs\n        num_extra = num_tokens % num_reqs\n        assert max_query_len is None or base_tokens + (num_extra > 0) <= max_query_len\n        num_scheduled_tokens = np.full(num_reqs, base_tokens, dtype=np.int32)\n        if num_extra > 0:\n            num_scheduled_tokens[-num_extra:] += 1\n        assert int(num_scheduled_tokens.sum()) == num_tokens\n\n        # seq_len equals to query_len\n        input_buffers.seq_lens[: num_reqs - num_extra] = base_tokens\n        input_buffers.seq_lens[num_reqs - num_extra : num_reqs] = base_tokens + 1\n        # Pad for full CUDA graph mode.\n        input_buffers.seq_lens[num_reqs:] = 0\n        seq_lens = input_buffers.seq_lens[:num_reqs]\n\n        query_start_loc_np = np.empty(num_reqs + 1, dtype=np.int32)\n        query_start_loc_np[0] = 0\n        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])\n        input_buffers.query_start_loc[:1] = 0\n        torch.cumsum(\n            seq_lens, dim=0, out=input_buffers.query_start_loc[1 : num_reqs + 1]\n        )\n        # Pad for full CUDA graph mode.\n        input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens\n        query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]\n\n        input_ids = input_buffers.input_ids[:num_tokens].zero_()\n        positions = input_buffers.positions[:num_tokens].zero_()\n\n        input_buffers.is_padding[:num_tokens].fill_(is_padding)\n        is_padding = input_buffers.is_padding[:num_tokens]\n\n        logits_indices = query_start_loc[1:] - 1\n        cu_num_logits = torch.arange(num_reqs + 1, device=device, dtype=torch.int32)\n        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)\n        # Copy so set_dummy_context can add context in place without touching\n        # num_scheduled_tokens.\n        seq_lens_cpu_upper_bound = torch.from_numpy(num_scheduled_tokens.copy())\n        return cls(\n            req_ids=req_ids,\n            num_reqs=num_reqs,\n            num_reqs_after_padding=num_reqs,\n            idx_mapping=idx_mapping,\n            idx_mapping_np=idx_mapping_np,\n            expanded_idx_mapping=expanded_idx_mapping,\n            expanded_local_pos=expanded_local_pos,\n            num_scheduled_tokens=num_scheduled_tokens,\n            num_tokens=num_tokens,\n            num_tokens_after_padding=num_tokens,\n            num_draft_tokens=0,\n            num_draft_tokens_per_req=None,\n            query_start_loc=query_start_loc,\n            query_start_loc_np=query_start_loc_np,\n            seq_lens=seq_lens,\n            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,\n            dcp_local_seq_lens=None,\n            num_computed_tokens_np=np.zeros(num_reqs, dtype=np.int32),\n            prefill_len_np=np.zeros(num_reqs, dtype=np.int32),\n            num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),\n            is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),\n            has_prefill=False,\n            max_seq_len_np=None,\n            decode_graph_eligible=True,\n            input_ids=input_ids,\n            positions=positions,\n            is_padding=is_padding,\n            logits_indices=logits_indices,\n            cu_num_logits=cu_num_logits,\n            cu_num_logits_np=cu_num_logits_np,\n            has_structured_output_reqs=False,\n            prompt_lens=None,\n            max_query_len=max_query_len,\n        )\n'

NATIVE_DESCRIPTOR_SOURCE = '@dataclass(frozen=True)\nclass BatchExecutionDescriptor:\n    """Describes the shape of the batch and CG mode to run; this is used to make shape\n    matches between the capture and runtime."""\n\n    cg_mode: CUDAGraphMode\n    num_tokens: int\n    num_reqs: int | None  # None means no request padding is needed (PIECEWISE graphs)\n    uniform_token_count: int | None = None\n    # Upper bound on per-request query length. Varlen decode graphs leave\n    # uniform_token_count unset, so this is what keeps a prefill batch out of one:\n    # the runner passes None for any batch with a prefill.\n    max_query_len: int | None = None\n    num_active_loras: int = 0\n    # Number of microbatches the batch is split into (DBO). 1 means no splitting.\n    num_ubatches: int = 1\n'

NATIVE_PARALLEL_SOURCE = '@dataclass\nclass ParallelConfig:\n    prefill_context_parallel_size: int = 8\n    decode_context_parallel_size: int = 1\n    cp_kv_cache_interleave_size: int = 1\n    tensor_parallel_size: int = 1\n    pipeline_parallel_size: int = 1\n    data_parallel_size: int = 1\n    enable_expert_parallel: bool = True\n\n    @property\n    def pcp_shard_decode_requests(self) -> bool:\n        """Whether PCP can shard decode requests across its ranks.\n\n        PCP-only execution replicates the KV cache, so each decode request can\n        have a single PCP owner. DCP shards the KV cache and therefore requires\n        every decode request to run on every participating DCP rank.\n        """\n        return (\n            self.prefill_context_parallel_size > 1\n            and self.decode_context_parallel_size == 1\n        )\n'

NATIVE_FACTORY_SOURCE = 'def maybe_build_pcp_manager(\n    vllm_config: VllmConfig,\n    device: torch.device,\n    supports_mm_inputs: bool,\n    block_tables: BlockTables,\n    cls: type[PCPManager] = PCPManager,\n) -> PCPManager | None:\n    parallel_config = vllm_config.parallel_config\n    pcp_size = parallel_config.prefill_context_parallel_size\n    if pcp_size <= 1:\n        return None\n\n    cls.validate_config(vllm_config, supports_mm_inputs)\n\n    pcp_rank = get_pcp_group().rank_in_group\n    dcp_size = parallel_config.decode_context_parallel_size\n    dcp_rank = get_dcp_group().rank_in_group if dcp_size > 1 else 0\n\n    return cls(\n        pcp_world_size=pcp_size,\n        pcp_rank=pcp_rank,\n        device=device,\n        shard_decode_requests=parallel_config.pcp_shard_decode_requests,\n        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,\n        max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,\n        block_tables=block_tables,\n        dcp_world_size=dcp_size,\n        dcp_rank=dcp_rank,\n        cp_interleave=parallel_config.cp_kv_cache_interleave_size,\n    )\n'

HCU_SCOPE_SOURCE = 'def _require_hcu_pcp_attribute(owner: object, name: str, owner_name: str) -> Any:\n    try:\n        return getattr(owner, name)\n    except AttributeError as exc:\n        raise PatchCompatibilityError(\n            f"HCU PCP requires {owner_name}.{name} in vLLM 0.25.1"\n        ) from exc\n\n\ndef _hy4_pcp_topology(parallel_config: object) -> tuple[int, int, int]:\n    """Identify the validated eight-card Hy4 PCP layouts."""\n    return (\n        int(\n            _require_hcu_pcp_attribute(\n                parallel_config, "tensor_parallel_size", "ParallelConfig"\n            )\n        ),\n        int(\n            _require_hcu_pcp_attribute(\n                parallel_config, "prefill_context_parallel_size", "ParallelConfig"\n            )\n        ),\n        int(\n            _require_hcu_pcp_attribute(\n                parallel_config, "pipeline_parallel_size", "ParallelConfig"\n            )\n        ),\n    )\n\n\ndef _require_hy4_pcp8_contract(vllm_config: object) -> None:\n    """Admit only the hardware-gated Hy4 PCP8 MTP3 configuration."""\n\n    speculative_config = _require_hcu_pcp_attribute(\n        vllm_config, "speculative_config", "VllmConfig"\n    )\n    if (\n        speculative_config is None\n        or _require_hcu_pcp_attribute(\n            speculative_config, "method", "SpeculativeConfig"\n        )\n        != "mtp"\n        or int(\n            _require_hcu_pcp_attribute(\n                speculative_config,\n                "num_speculative_tokens",\n                "SpeculativeConfig",\n            )\n        )\n        != 3\n    ):\n        raise ValueError("HY V4 PCP8 requires checkpoint-native MTP3.")\n\n    parallel_config = _require_hcu_pcp_attribute(\n        vllm_config, "parallel_config", "VllmConfig"\n    )\n    kernel_config = _require_hcu_pcp_attribute(\n        vllm_config, "kernel_config", "VllmConfig"\n    )\n    cache_config = _require_hcu_pcp_attribute(\n        vllm_config, "cache_config", "VllmConfig"\n    )\n    if bool(getattr(parallel_config, "enable_eplb", False)):\n        raise ValueError("HY V4 PCP8 does not support EPLB.")\n    if (\n        _require_hcu_pcp_attribute(\n            parallel_config, "all2all_backend", "ParallelConfig"\n        )\n        != "deepep_high_throughput"\n        or _require_hcu_pcp_attribute(\n            kernel_config, "moe_backend", "KernelConfig"\n        )\n        != "deep_gemm"\n        or _require_hcu_pcp_attribute(\n            cache_config, "cache_dtype", "CacheConfig"\n        )\n        not in ("fp8_e4m3", "fp8_ds_mla")\n    ):\n        raise ValueError(\n            "HY V4 PCP8 requires DeepEP HT, DeepGEMM and FP8 E4M3 KV."\n        )\n    from vllm.v1.attention.backends.registry import AttentionBackendEnum\n\n    attention_config = _require_hcu_pcp_attribute(\n        vllm_config, "attention_config", "VllmConfig"\n    )\n    if (\n        _require_hcu_pcp_attribute(\n            attention_config, "backend", "AttentionConfig"\n        )\n        != AttentionBackendEnum.FLASHMLA_SPARSE\n    ):\n        raise ValueError(\n            "HY V4 PCP8 requires the FLASHMLA_SPARSE attention backend."\n        )\n    from vllm_hcu.platforms import envs as hcu_envs\n\n    gate_shard_enabled = hcu_envs.linear_gate_pcp_shard_enabled(\n        tp_size=int(parallel_config.tensor_parallel_size),\n        pcp_size=int(parallel_config.prefill_context_parallel_size),\n        pp_size=int(parallel_config.pipeline_parallel_size),\n    )\n    if not gate_shard_enabled:\n        if not hcu_envs.custom_ops_enabled():\n            raise ValueError(\n                "HY V4 PCP8 requires effective linear_gate PCP sharding, "\n                "but VLLM_HCU_USE_CUSTOM_OPS=0 disables it."\n            )\n        raise ValueError(\n            "HY V4 PCP8 requires effective linear_gate PCP sharding; set "\n            "VLLM_HCU_ENABLE_LINEAR_GATE_PCP_SHARD=1 to avoid replicated "\n            "linear_gate weights exhausting KV-cache memory."\n        )\n\n\ndef is_glm53_pcp(vllm_config: object) -> bool:\n    model = getattr(vllm_config, "model_config", None)\n    parallel = getattr(vllm_config, "parallel_config", None)\n    return (\n        getattr(model, "architectures", None)\n        in (["Glm5NextForConditionalGeneration"], ["Glm5NextForCausalLM"])\n        and getattr(parallel, "prefill_context_parallel_size", 1) > 1\n    )\n\n\ndef _is_glm53_mtp_pcp(vllm_config: object) -> bool:\n    model = getattr(vllm_config, "model_config", None)\n    parallel = getattr(vllm_config, "parallel_config", None)\n    return (\n        getattr(model, "architectures", None) == ["Glm5NextMTPModel"]\n        and getattr(parallel, "prefill_context_parallel_size", 1) > 1\n    )\n\n\ndef _require_glm53_pcp_contract(vllm_config: object) -> None:\n    model = vllm_config.model_config\n    parallel = vllm_config.parallel_config\n    topology = (\n        parallel.tensor_parallel_size,\n        parallel.prefill_context_parallel_size,\n        parallel.pipeline_parallel_size,\n    )\n    if topology != (1, 8, 1):\n        raise ValueError("GLM-5.3 PCP baseline requires TP1/PCP8/PP1.")\n    mm = getattr(model, "multimodal_config", None)\n    if model.is_multimodal_model and not getattr(mm, "language_model_only", False):\n        raise ValueError("GLM-5.3 PCP baseline requires --language-model-only.")\n    if not model.use_mla or not model.is_hybrid:\n        raise ValueError("GLM-5.3 PCP requires the MLA/KDA hybrid model.")\n    # Built-in MTP uses the restored global batch in the HCU V2 runner.\n    # The shared PCP contract below validates method=\'mtp\' and 1..3 tokens.\n    if getattr(parallel, "enable_eplb", False):\n        raise ValueError("GLM-5.3 PCP baseline does not support EPLB.")\n\n\ndef _require_glm53_mtp_pcp_contract(vllm_config: object) -> None:\n    model = vllm_config.model_config\n    parallel = vllm_config.parallel_config\n    topology = (\n        parallel.tensor_parallel_size,\n        parallel.prefill_context_parallel_size,\n        parallel.pipeline_parallel_size,\n    )\n    if topology != (1, 8, 1):\n        raise ValueError("GLM-5.3 MTP PCP requires TP1/PCP8/PP1.")\n    if not model.use_mla:\n        raise ValueError("GLM-5.3 MTP PCP requires the built-in MLA draft layer.")\n    if getattr(parallel, "enable_eplb", False):\n        raise ValueError("GLM-5.3 MTP PCP does not support EPLB.")\n\n\ndef _require_mrv2_pcp_contract(vllm_config: object) -> None:\n    """Reject every Model Runner V2 PCP configuration outside HCU support."""\n\n    if not _require_hcu_pcp_attribute(\n        vllm_config, "use_v2_model_runner", "VllmConfig"\n    ):\n        raise ValueError("HCU PCP requires Model Runner V2.")\n\n    model_config = _require_hcu_pcp_attribute(\n        vllm_config, "model_config", "VllmConfig"\n    )\n    parallel_config = _require_hcu_pcp_attribute(\n        vllm_config, "parallel_config", "VllmConfig"\n    )\n    cache_config = _require_hcu_pcp_attribute(\n        vllm_config, "cache_config", "VllmConfig"\n    )\n\n    architectures = _require_hcu_pcp_attribute(\n        model_config, "architectures", "ModelConfig"\n    )\n    use_mla = bool(\n        _require_hcu_pcp_attribute(model_config, "use_mla", "ModelConfig")\n    )\n    from vllm_hcu.v1.dsa_cache_layer_split import is_hcu_dsa_mla_model\n\n    is_dsa_mla = is_hcu_dsa_mla_model(model_config)\n    is_glm52 = architectures == ["GlmMoeDsaForCausalLM"]\n    is_hy4 = architectures == ["HYV4ForCausalLM"]\n    is_dsv4 = architectures in (\n        ["DeepseekV4ForCausalLM"], ["DeepSeekV4MTPModel"]\n    )\n    is_glm53 = is_glm53_pcp(vllm_config)\n    is_glm53_mtp = _is_glm53_mtp_pcp(vllm_config)\n    if is_glm53:\n        _require_glm53_pcp_contract(vllm_config)\n    if is_glm53_mtp:\n        _require_glm53_mtp_pcp_contract(vllm_config)\n    if use_mla and not (\n        is_dsa_mla or is_glm52 or is_hy4 or is_glm53 or is_glm53_mtp or is_dsv4\n    ):\n        raise ValueError(\n            "HCU MLA PCP only supports a registered DSA MLA family, "\n            "GlmMoeDsaForCausalLM, HYV4ForCausalLM, DeepseekV4ForCausalLM, "\n            "DeepSeekV4MTPModel, or the GLM-5.3 PCP contract."\n        )\n    if is_dsv4:\n        if not use_mla:\n            raise ValueError("DeepSeek V4 PCP requires sparse MLA.")\n        if getattr(parallel_config, "cp_kv_cache_interleave_size", 1) != 1:\n            raise ValueError("DeepSeek V4 PCP requires Full KV without interleaving.")\n    if is_glm52 and not use_mla:\n        raise ValueError("GLM-5.2 PCP requires MLA or sparse MLA.")\n    if is_hy4 and not use_mla:\n        raise ValueError("HY V4 PCP requires sparse MLA.")\n    if not use_mla:\n        from vllm.v1.attention.backends.registry import AttentionBackendEnum\n\n        attention_config = _require_hcu_pcp_attribute(\n            vllm_config, "attention_config", "VllmConfig"\n        )\n        attention_backend = _require_hcu_pcp_attribute(\n            attention_config, "backend", "AttentionConfig"\n        )\n        if attention_backend != AttentionBackendEnum.FLASH_ATTN:\n            raise ValueError(\n                "FlashAttention GQA PCP only supports FLASH_ATTN attention backend."\n            )\n    if not use_mla and _require_hcu_pcp_attribute(\n        model_config, "is_hybrid", "ModelConfig"\n    ):\n        raise ValueError(\n            "FlashAttention PCP does not support hybrid KV cache groups."\n        )\n    hy4_topology = _hy4_pcp_topology(parallel_config) if is_hy4 else None\n    if is_hy4 and hy4_topology not in {(4, 2, 1), (1, 4, 2), (1, 8, 1)}:\n        raise ValueError(f"HY V4 PCP topology is not validated: {hy4_topology}")\n    if is_hy4 and hy4_topology == (1, 8, 1):\n        _require_hy4_pcp8_contract(vllm_config)\n    if _require_hcu_pcp_attribute(\n        parallel_config, "pipeline_parallel_size", "ParallelConfig"\n    ) != 1 and hy4_topology != (1, 4, 2):\n        raise ValueError("HCU PCP does not support pipeline parallelism.")\n    dcp_size = int(\n        _require_hcu_pcp_attribute(\n            parallel_config, "decode_context_parallel_size", "ParallelConfig"\n        )\n    )\n    if use_mla and dcp_size != 1:\n        raise ValueError("GLM-5.2 PCP does not support decode context parallelism.")\n    if not use_mla and dcp_size != 1:\n        raise ValueError(\n            "FlashAttention PCP does not support decode context parallelism."\n        )\n    if _require_hcu_pcp_attribute(\n        parallel_config, "data_parallel_size", "ParallelConfig"\n    ) != 1:\n        raise ValueError("HCU PCP does not support data parallelism.")\n    if use_mla and not _require_hcu_pcp_attribute(\n        parallel_config, "enable_expert_parallel", "ParallelConfig"\n    ):\n        raise ValueError("GLM-5.2 PCP requires expert parallelism.")\n    if use_mla and not _require_hcu_pcp_attribute(\n        model_config, "enforce_eager", "ModelConfig"\n    ):\n        raise ValueError("GLM-5.2 PCP requires eager execution without graphs.")\n    speculative_config = _require_hcu_pcp_attribute(\n        vllm_config, "speculative_config", "VllmConfig"\n    )\n    if speculative_config is not None:\n        if not use_mla:\n            raise ValueError(\n                "FlashAttention PCP does not support speculative decoding or MTP."\n            )\n        method = _require_hcu_pcp_attribute(\n            speculative_config, "method", "SpeculativeConfig"\n        )\n        if method != "mtp":\n            raise ValueError("GLM-5.2 PCP only supports built-in MTP.")\n        num_speculative_tokens = _require_hcu_pcp_attribute(\n            speculative_config,\n            "num_speculative_tokens",\n            "SpeculativeConfig",\n        )\n        if num_speculative_tokens not in (1, 2, 3):\n            raise ValueError(\n                "GLM-5.2 PCP+MTP requires one to three speculative tokens."\n            )\n        if is_hy4 and hy4_topology == (1, 4, 2) and num_speculative_tokens != 2:\n            raise ValueError("HY V4 PP2 PCP+MTP requires exactly two draft tokens.")\n    if _require_hcu_pcp_attribute(vllm_config, "lora_config", "VllmConfig") is not None:\n        raise ValueError("HCU PCP does not support LoRA.")\n    if _require_hcu_pcp_attribute(\n        model_config, "is_multimodal_model", "ModelConfig"\n    ) and not is_glm53:\n        raise ValueError("HCU PCP does not support multimodal models.")\n    if _require_hcu_pcp_attribute(\n        cache_config, "kv_offloading_size", "CacheConfig"\n    ) is not None:\n        raise ValueError("HCU PCP does not support KV offload.")\n\n    kv_transfer_config = _require_hcu_pcp_attribute(\n        vllm_config, "kv_transfer_config", "VllmConfig"\n    )\n    kv_connector = None\n    if kv_transfer_config is not None:\n        kv_connector = _require_hcu_pcp_attribute(\n            kv_transfer_config, "kv_connector", "KVTransferConfig"\n        )\n    if kv_transfer_config is not None and (\n        is_hy4 or is_dsa_mla or kv_connector is not None\n    ):\n        if not (\n            is_dsa_mla\n            and kv_connector == "MooncakeConnector"\n            and getattr(kv_transfer_config, "kv_role", None) == "kv_producer"\n            and getattr(kv_transfer_config, "kv_connector_module_path", None)\n            is None\n        ):\n            raise ValueError(\n                "HCU PCP P/D disaggregation requires a supported DSA MLA "\n                "producer with the registered MooncakeConnector, "\n                "kv_role=\'kv_producer\', and no custom connector module."\n            )\n\n    feature_config = get_hcu_config(vllm_config)\n    if feature_config.enable_lightly_cp:\n        raise ValueError("HCU PCP does not support lightly-CP.")\n    if feature_config.enable_multi_layers_mtp:\n        if use_mla:\n            raise ValueError("GLM-5.2 PCP does not support HCU multi-layer MTP.")\n        raise ValueError("FlashAttention PCP does not support multi-layer MTP.")\n\n\ndef _validate_hcu_pcp_scope(vllm_config: object) -> bool:\n    """Return whether this configuration is in the HCU PCP support scope."""\n\n    parallel_config = _require_hcu_pcp_attribute(\n        vllm_config, "parallel_config", "VllmConfig"\n    )\n    pcp_size = _require_hcu_pcp_attribute(\n        parallel_config, "prefill_context_parallel_size", "ParallelConfig"\n    )\n    if pcp_size <= 1:\n        return False\n\n    _require_mrv2_pcp_contract(vllm_config)\n    return True\n'

ROOT = Path(os.environ.get("HCU_PCP_ABI_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
SOURCE_PATH = ROOT / "vllm_hcu/v1/pcp_manager.py"
STAGED_FIELDS = ("input_ids", "positions", "is_padding", "query_start_loc", "seq_lens")


class PatchCompatibilityError(RuntimeError):
    pass


class Group:
    def __init__(self, rank=0, size=8):
        self.rank_in_group, self.world_size = rank, size


def execute_nodes(source, scope, names=None, strip_imports=False):
    tree = ast.parse(source)
    nodes = tree.body if names is None else [n for n in tree.body if getattr(n, "name", None) in names]
    if strip_imports:
        class Boundaries(ast.NodeTransformer):
            def visit_Import(self, node):
                return ast.copy_location(ast.Pass(), node)
            def visit_ImportFrom(self, node):
                return ast.copy_location(ast.Pass(), node)
        nodes = [Boundaries().visit(copy.deepcopy(n)) for n in nodes]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<pinned-source-ast>", "exec"), scope)


def load_protocol():
    scope = dict(torch=torch, np=np, dataclass=dataclasses.dataclass,
                 replace=dataclasses.replace, enum=enum, NamedTuple=NamedTuple,
                 random_uuid=lambda: uuid4().hex)
    execute_nodes(NATIVE_ENUM_SOURCE, scope)
    execute_nodes(NATIVE_INPUT_SOURCE, scope)
    execute_nodes(NATIVE_DESCRIPTOR_SOURCE, scope)
    execute_nodes(NATIVE_PARALLEL_SOURCE, scope)
    return SimpleNamespace(**scope)


P = load_protocol()


@dataclasses.dataclass
class SchedulerConfig:
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 128


@dataclasses.dataclass
class ModelConfig:
    architectures: list = dataclasses.field(default_factory=lambda: ["DeepseekV4ForCausalLM"])
    use_mla: bool = True
    enforce_eager: bool = True
    is_multimodal_model: bool = False


@dataclasses.dataclass
class Config:
    parallel_config: object = dataclasses.field(default_factory=P.ParallelConfig)
    scheduler_config: object = dataclasses.field(default_factory=SchedulerConfig)
    model_config: object = dataclasses.field(default_factory=ModelConfig)
    use_v2_model_runner: bool = True
    speculative_config: object = None
    lora_config: object = None
    kv_transfer_config: object = None
    cache_config: object = dataclasses.field(default_factory=lambda: SimpleNamespace(kv_offloading_size=None))


class BlockTables:
    """Real tensors at the block-table boundary; no GPU kernels or collectives."""
    def __init__(self):
        table = torch.arange(64, dtype=torch.int32).reshape(16, 4)
        self.block_tables = [SimpleNamespace(gpu=table)]
        self.input_block_tables = [torch.full((12, 4), -77, dtype=torch.int32)]
        self.num_kv_cache_groups = 1

    def compute_slot_mappings(self, idx_mapping, query_start_loc, positions, num_tokens):
        result = torch.empty((1, num_tokens), dtype=torch.int64)
        for row in range(idx_mapping.numel()):
            start, stop = int(query_start_loc[row]), int(query_start_loc[row + 1])
            result[0, start:stop] = idx_mapping[row] * 100 + positions[start:stop]
        return result


def load_manager(rank=0, config=None, path=None):
    config = config or Config()
    pcp = Group(rank, config.parallel_config.prefill_context_parallel_size)
    dcp = Group(0, config.parallel_config.decode_context_parallel_size)
    scope = dict(vars(P), PatchCompatibilityError=PatchCompatibilityError,
                 get_pcp_group=lambda: pcp, get_dcp_group=lambda: dcp,
                 PAD_SLOT_ID=-1, get_dcp_local_seq_lens=lambda *args: None,
                 get_hcu_config=lambda c: SimpleNamespace(enable_lightly_cp=False, enable_multi_layers_mtp=False),
                 is_hcu_dsa_mla_model=lambda c: False)
    execute_nodes(HCU_SCOPE_SOURCE, scope, strip_imports=True)
    source = Path(path or SOURCE_PATH).read_text()
    tree = ast.parse(source)
    # Keep the actual sampling sentinel assignment in source order: its value
    # is captured as the class method's default while the class is defined.
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             or (isinstance(n, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id == "_AUX_HIDDEN_UNSET"
                 for target in n.targets))]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<pinned-source-ast>", "exec"), scope)
    # Every factory call supplies the custom cls; resolve only its unused
    # default-class expression here without importing the native package.
    scope["PCPManager"] = scope["HcuPCPManager"]
    execute_nodes(NATIVE_FACTORY_SOURCE, scope)
    cls = scope["make_hcu_pcp_manager_cls"](config)
    tables = BlockTables()
    def build():
        return scope["maybe_build_pcp_manager"](config, torch.device("cpu"), False, tables, cls=cls)
    return SimpleNamespace(scope=scope, cls=cls, config=config, tables=tables, build=build)


def manager(rank=0):
    return load_manager(rank).build()


def batch(mixed=True):
    lengths = np.asarray([1, 17, 1] if mixed else [1, 1, 1], dtype=np.int32)
    starts = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    size = int(starts[-1])
    batch = P.InputBatch.make_dummy(3, size, P.InputBuffers(4, 128, "cpu"), is_padding=False)
    computed = np.asarray([50, 100, 70], dtype=np.int32)
    # CPU cursors intentionally exceed the rejection-corrected device cursor.
    positions = torch.cat([torch.arange(int(c - 3), int(c - 3 + n), dtype=torch.int64)
                           for c, n in zip(computed, lengths)])
    prefill = np.asarray([False, mixed, False], dtype=np.bool_)
    return dataclasses.replace(batch, req_ids=["d0", "p1" if mixed else "d1", "d2"],
        num_scheduled_tokens=lengths, query_start_loc=torch.from_numpy(starts.copy()),
        query_start_loc_np=starts, input_ids=torch.arange(1000, 1000 + size, dtype=torch.int32),
        positions=positions, seq_lens=torch.tensor([int(positions[int(stop)-1])+1 for stop in starts[1:]], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.from_numpy(computed + lengths),
        num_computed_tokens_np=computed, prefill_len_np=np.where(prefill, computed + lengths, 0),
        num_computed_prefill_tokens_np=np.where(prefill, computed, 0),
        is_prefilling_np=prefill, has_prefill=bool(mixed), decode_graph_eligible=not mixed,
        logits_indices=torch.from_numpy((starts[1:]-1).astype(np.int64)))


def desc(mode=None, tokens=127, reqs=11):
    return P.BatchExecutionDescriptor(P.CUDAGraphMode.NONE if mode is None else mode, tokens, reqs,
                                       max_query_len=111, num_active_loras=7, num_ubatches=1)


def snapshots(obj):
    return {name: getattr(obj, name).clone() for name in STAGED_FIELDS}


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_actual_factory_capability_and_optional_request_state(rank):
    h = load_manager(rank)
    m = h.build()
    assert m.pcp_size == m.pcp_world_size == 8 and m.pcp_rank == rank
    assert h.config.parallel_config.pcp_shard_decode_requests is True
    assert m.native_shard_decode_requests is True
    assert m.requested_shard_decode_requests is True
    assert m.shard_decode_requests is False and m.decode_strategy == "replicated"
    assert m._req_states is None and m._block_tables is h.tables
    assert m.input_buffers.max_num_reqs == 3 * h.config.scheduler_config.max_num_seqs


@pytest.mark.parametrize("pcp,dcp,want", [(8, 1, True), (8, 2, False), (8, 8, False), (1, 1, False)])
def test_actual_parallel_property(pcp, dcp, want):
    p = P.ParallelConfig(prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp)
    assert p.pcp_shard_decode_requests is want


@pytest.mark.parametrize("bad", [False, 1, 0, "True", np.bool_(True)])
def test_constructor_rejects_wrong_capability(bad):
    h = load_manager()
    with pytest.raises(PatchCompatibilityError, match="boolean capability"):
        h.cls(8, 0, torch.device("cpu"), max_num_reqs=4, max_num_tokens=128,
              block_tables=h.tables, shard_decode_requests=bad)


@pytest.mark.parametrize("changes", [{"max_num_reqs": 5}, {"max_num_tokens": 129},
    {"pcp_world_size": 4}, {"dcp_world_size": 2}, {"cp_interleave": 2}, {"pcp_rank": 1}, {"dcp_rank": 1}])
def test_constructor_keeps_config_and_rank_validation(changes):
    h = load_manager()
    args = dict(pcp_world_size=8, pcp_rank=0, device=torch.device("cpu"),
                max_num_reqs=4, max_num_tokens=128, block_tables=h.tables,
                shard_decode_requests=True)
    args.update(changes)
    with pytest.raises(PatchCompatibilityError, match="match"):
        h.cls(**args)


def test_legacy_direct_constructor_and_padding_aliases():
    h = load_manager()
    state = object()
    m = h.cls(8, 0, torch.device("cpu"), state, 4, 128, h.tables)
    assert m._req_states is state and m.requested_shard_decode_requests is None
    raw = h.scope["HcuPCPManager"](h.config, "cpu", state, h.tables, pcp_group=Group())
    assert raw._req_states is state
    b = batch()
    assert m.partition_batch(b, 20).num_tokens_after_padding == 20
    assert m.partition_batch(b, padded_num_tokens=21).num_tokens_after_padding == 21


@pytest.mark.parametrize("rank", [0, 1, 7])
@pytest.mark.parametrize("mixed", [False, True])
def test_real_descriptor_partition_replication_positions_and_ownership(rank, mixed):
    m, b = manager(rank), batch(mixed)
    original = snapshots(b)
    d = desc()
    local = m.partition_batch(b, d)
    assert isinstance(b, P.InputBatch) and isinstance(d, P.BatchExecutionDescriptor)
    assert isinstance(local, P.InputBatch) and m._global_batch is b
    assert local.num_tokens_after_padding == m.get_num_tokens_for_dispatch(b.num_scheduled_tokens, b.is_prefilling_np)
    assert local.num_tokens_after_padding != d.num_tokens
    for segment in m._local_segments:
        if segment.num_actual_tokens:
            start = segment.local_slice.start
            stop = start + segment.num_actual_tokens
            assert torch.equal(local.positions[start:stop], b.positions[segment.global_slice])
            assert local.seq_lens[m._local_segments.index(segment)].item() == b.positions[segment.global_slice.stop - 1].item() + 1
    for name in ("d0", "d2"):
        assert local.req_ids.count(name) == 1
    assert m.shard_decode_requests is False
    m._install_dsa_sparse_broadcast_block_ids = lambda: None
    _, slots = m.prepare_attn(local)
    assert getattr(local, "_vllm_hcu_pcp_replicated_token_mask").sum().item() == (2 if mixed else 3)
    if mixed:
        # Exactly one global writer per token, despite all-rank decode replicas.
        assert m._gathered_kv_write_mask.sum().item() == b.num_tokens
        mask = m._gathered_kv_write_mask.reshape(8, -1)
        gather = m._padded_gather_idx.reshape(8, -1)
        for decode_idx in (0, b.num_tokens - 1):
            assert ((gather == decode_idx) & mask).sum().item() == 1
            assert ((gather[0] == decode_idx) & mask[0]).sum().item() == 1
    else:
        assert slots.shape == (1, b.num_tokens)
        assert torch.equal(slots[0], b.idx_mapping.to(torch.int64) * 100 + b.positions)
    for name, value in original.items():
        assert torch.equal(getattr(b, name), value)
        assert getattr(local, name).data_ptr() != getattr(b, name).data_ptr()


@pytest.mark.parametrize("mode", [P.CUDAGraphMode.PIECEWISE, P.CUDAGraphMode.FULL,
    P.CUDAGraphMode.FULL_DECODE_ONLY, P.CUDAGraphMode.FULL_AND_PIECEWISE])
def test_non_none_descriptors_fail_before_layout_mutation(mode):
    m = manager()
    b = batch()
    m.partition_batch(b, desc())
    saved = (m._global_batch, m._hidden_restore_idx, m._padded_gather_idx, m._local_segments)
    values = snapshots(m.input_buffers)
    with pytest.raises(PatchCompatibilityError, match="unsupported.*NONE"):
        m.partition_batch(batch(False), desc(mode))
    assert all(x is y for x, y in zip(saved, (m._global_batch, m._hidden_restore_idx, m._padded_gather_idx, m._local_segments)))
    assert all(torch.equal(v, getattr(m.input_buffers, k)) for k, v in values.items())


@pytest.mark.parametrize("rank", [0, 1, 7])
@pytest.mark.parametrize("padding", [False, True])
def test_native_make_dummy_stages_five_fields_stable_addresses(rank, padding):
    m, global_batch = manager(rank), batch()
    m.partition_batch(global_batch, desc())
    saved = (m._global_batch, m._hidden_restore_idx, m._padded_gather_idx, m._local_segments,
             m._global_attn_ready, m._cached_global_block_tables, m._padded_num_tokens)
    for nreq, ntok in [(3, 19), (1, 1), (3, 19)]:
        b = P.InputBatch.make_dummy(nreq, ntok, P.InputBuffers(4, 128, "cpu"),
                                  max_query_len=19, is_padding=padding)
        b.input_ids.copy_(torch.arange(ntok, dtype=torch.int32) + 5)
        b.positions.copy_(torch.arange(ntok, dtype=torch.int64) + 20)
        original = snapshots(b)
        staged = m.prepare_inputs_to_capture(b)
        assert isinstance(staged, P.InputBatch) and staged is not b
        for field in dataclasses.fields(P.InputBatch):
            if field.name not in STAGED_FIELDS:
                assert getattr(staged, field.name) is getattr(b, field.name)
        for name, value in original.items():
            assert torch.equal(getattr(staged, name), value)
            assert torch.equal(getattr(b, name), value)
            assert getattr(staged, name).data_ptr() == getattr(m.input_buffers, name).data_ptr()
            assert getattr(staged, name).data_ptr() != getattr(b, name).data_ptr()
        assert all(x is y for x, y in zip(saved, (m._global_batch, m._hidden_restore_idx, m._padded_gather_idx, m._local_segments,
                   m._global_attn_ready, m._cached_global_block_tables, m._padded_num_tokens)))


@pytest.mark.parametrize("bad", ["token_capacity", "request_capacity", "negative_tokens", "negative_requests",
    "input_ids_shape", "positions_dtype", "is_padding_shape", "query_start_loc_shape", "seq_lens_shape"])
def test_malformed_capture_checks_every_field_before_copy(bad):
    m = manager()
    for name in STAGED_FIELDS:
        getattr(m.input_buffers, name).fill_(True if name == "is_padding" else 91)
    before = snapshots(m.input_buffers)
    b = P.InputBatch.make_dummy(3, 19, P.InputBuffers(4, 128, "cpu"))
    if bad == "token_capacity": b = dataclasses.replace(b, num_tokens_after_padding=129)
    elif bad == "request_capacity": b = dataclasses.replace(b, num_reqs_after_padding=13)
    elif bad == "negative_tokens": b = dataclasses.replace(b, num_tokens_after_padding=-1)
    elif bad == "negative_requests": b = dataclasses.replace(b, num_reqs_after_padding=-1)
    elif bad == "positions_dtype": b = dataclasses.replace(b, positions=b.positions.to(torch.int32))
    else:
        name = bad.removesuffix("_shape")
        b = dataclasses.replace(b, **{name: getattr(b, name)[:-1]})
    with pytest.raises(PatchCompatibilityError, match="dummy"):
        m.prepare_inputs_to_capture(b)
    assert all(torch.equal(v, getattr(m.input_buffers, k)) for k, v in before.items())


@pytest.mark.parametrize("field,value", [("decode_context_parallel_size", 2), ("data_parallel_size", 2), ("pipeline_parallel_size", 2)])
def test_existing_scope_gate_unchanged(field, value):
    c = Config()
    setattr(c.parallel_config, field, value)
    with pytest.raises(ValueError):
        load_manager(config=c).build()


def test_no_framework_import_or_accelerator_initialization():
    assert not any(name == "vllm" or name.startswith("vllm.") or name == "vllm_hcu" or name.startswith("vllm_hcu.") for name in sys.modules)
    assert not torch.cuda.is_initialized()
