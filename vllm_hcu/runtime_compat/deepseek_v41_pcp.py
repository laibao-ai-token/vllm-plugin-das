# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Narrow configuration contract for the V41 replicated-cache PCP baseline."""
from vllm_hcu.deepseek_v4_runtime import config_field, model_architectures


def is_v41_pcp(config):
    parallel = config_field(config, "parallel_config")
    return (model_architectures(config) == ("DeepseekV41ForCausalLM",)
            and int(config_field(parallel, "prefill_context_parallel_size", 1)) > 1)


def require_v41_pcp_contract(config):
    """The target-only dependency slice does not admit speculative execution."""
    if not is_v41_pcp(config):
        raise ValueError("V41 PCP requires the native V41 target with PCP>1")
    parallel = config_field(config, "parallel_config")
    model = config_field(config, "model_config")
    cache = config_field(config, "cache_config")
    attention = config_field(config, "attention_config")
    topology = {"tensor_parallel_size": 1, "prefill_context_parallel_size": 8,
                "data_parallel_size": 1, "pipeline_parallel_size": 1,
                "decode_context_parallel_size": 1, "cp_kv_cache_interleave_size": 1}
    if any(config_field(parallel, name) != value for name, value in topology.items()):
        raise ValueError("V41 PCP requires TP1/PCP8/DP1/PP1/DCP1 without interleaving")
    if not config_field(config, "use_v2_model_runner", False):
        raise ValueError("V41 PCP requires Model Runner V2")
    if not config_field(model, "use_mla", False):
        raise ValueError("V41 PCP requires native sparse MLA")
    if not config_field(model, "enforce_eager", False):
        raise ValueError("V41 PCP requires eager execution")
    if not config_field(parallel, "enable_expert_parallel", False):
        raise ValueError("V41 PCP requires expert parallelism")
    if config_field(parallel, "all2all_backend") != "deepep_high_throughput":
        raise ValueError("V41 PCP requires the explicit DeepEP HT backend")
    if config_field(parallel, "enable_eplb", False):
        raise ValueError("V41 PCP does not support EPLB")
    if config_field(model, "is_multimodal_model", False) or config_field(config, "lora_config") is not None:
        raise ValueError("V41 PCP does not support multimodal inputs or LoRA")
    if config_field(config, "kv_transfer_config") is not None or config_field(cache, "kv_offloading_size") is not None:
        raise ValueError("V41 PCP requires standalone caches without KV offload")
    if config_field(attention, "indexer_kv_dtype", "auto") not in ("auto", "fp8"):
        raise ValueError("V41 PCP supports the FP8 indexer only")
    if config_field(attention, "indexer_sparse_logits", False):
        raise ValueError("V41 PCP does not support sparse-logits indexers")
    if config_field(config, "speculative_config") is not None:
        raise ValueError("V41 PCP speculative execution requires the separate draft/group integration")
