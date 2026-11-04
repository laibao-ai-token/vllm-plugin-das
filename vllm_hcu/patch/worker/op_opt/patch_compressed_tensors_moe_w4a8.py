# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route CT W4A8 with explicit prefill/draft gates and decode diagnostics."""
from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe"
)
PATCH_ID = "worker.op_opt.compressed_tensors.moe_w4a8_deepgemm"
TARGETS = (f"{TARGET_MODULE}.CompressedTensorsMoEMethod.get_moe_method",)
_MARKER = "_vllm_hcu_ct_w4a8_deepgemm"


def _validate_topology(config, moe, is_draft_model):
    """Validate CT W4A8 invariants, not a fixed list of parallel deployments.

    The W4A8 expert path holds no data-parallel state: weights, packing and
    DeepEP/DeepGEMM experts are EP-scoped. DP width is therefore checked
    against the actual EP group and MoE capacity instead of being hard-coded.
    TP/PP/DCP remain real layout constraints of the packed HIPC kernel.
    """
    from vllm_hcu.dspark_runtime import native_decode_dspark
    native_decode = native_decode_dspark(config)
    parallel = config.parallel_config
    scope = "draft" if is_draft_model else "prefill"
    if not is_draft_model and parallel.prefill_context_parallel_size == 1:
        scope = "decode"
    reasons = []
    if parallel.tensor_parallel_size != 1:
        reasons.append(f"tp={parallel.tensor_parallel_size}")
    if parallel.pipeline_parallel_size != 1:
        reasons.append(f"pp={parallel.pipeline_parallel_size}")
    if parallel.decode_context_parallel_size != 1:
        reasons.append(f"dcp={parallel.decode_context_parallel_size}")
    backend = moe.moe_parallel_config.all2all_backend
    actual_parallel = moe.moe_parallel_config
    if scope == "decode":
        dp = parallel.data_parallel_size
        if type(dp) is not int or dp < 1:
            reasons.append(f"dp={dp}")
        if not actual_parallel.use_ep or backend != "deepep_low_latency":
            reasons.append("decode_requires_fixed_ll_ep")
        budget = config.scheduler_config.max_num_batched_tokens
        declared = getattr(moe, "max_num_tokens", None)
        if (type(budget) is not int or budget <= 0
                or type(declared) is not int or declared != budget):
            reasons.append("moe_capacity_must_match_scheduler")
        for name, expected in (("tp_size", 1), ("pcp_size", 1),
                               ("dp_size", dp), ("ep_size", dp)):
            if getattr(actual_parallel, name, None) != expected:
                reasons.append(f"moe_{name}_mismatch")
        from vllm_hcu.dspark_runtime import validate_graph_capacity
        validate_graph_capacity(config, moe)
        if native_decode:
            from vllm_hcu.dspark_runtime import token_capacity
            token_capacity(config)
    else:
        expected_pcp = 1 if is_draft_model else 8
        if parallel.prefill_context_parallel_size != expected_pcp:
            reasons.append(f"pcp={parallel.prefill_context_parallel_size}")
        if parallel.data_parallel_size != 1:
            if is_draft_model and native_decode and parallel.data_parallel_size > 1:
                from vllm_hcu.dspark_runtime import (
                    get_dspark_all2all_manager,
                    get_dspark_token_capacity,
                    validate_graph_capacity,
                )
                manager = get_dspark_all2all_manager(moe)
                capacity = get_dspark_token_capacity()
                dp = parallel.data_parallel_size
                if (manager is None or capacity is None
                        or getattr(manager, "world_size", None) != dp
                        or getattr(manager, "dp_world_size", None) != dp
                        or getattr(actual_parallel, "ep_size", None) != dp
                        or getattr(actual_parallel, "dp_size", None) != dp
                        or getattr(moe, "max_num_tokens", None) != capacity.capacity_per_rank):
                    reasons.append("draft_group_capacity_contract=unverified: draft manager/scope mismatch")
                validate_graph_capacity(config, moe)
            else:
                reasons.append(f"dp={parallel.data_parallel_size}")
                if is_draft_model:
                    reasons.append("draft_group_capacity_contract=unverified: draft manager/scope missing")
        if backend not in ("deepep_high_throughput", "allgather_reducescatter"):
            reasons.append("prefill_or_draft_requires_contiguous_backend")
    if parallel.enable_eplb:
        reasons.append("eplb=True")
    # PCP's virtual batch cannot be captured; PCP1 decode may use graphs.
    if scope == "prefill" and not config.model_config.enforce_eager:
        reasons.append("enforce_eager=False; graph_contract=unverified")
    if is_draft_model:
        spec = getattr(config, "speculative_config", None)
        draft_model = getattr(spec, "draft_model_config", None)
        if draft_model is not None and not draft_model.enforce_eager and not native_decode:
            reasons.append("draft_graph_contract=unverified")
    if config.lora_config is not None:
        reasons.append("lora!=None")
    if reasons:
        raise ValueError(
            "HCU CT W4A8 topology contract rejected; "
            f"scope={scope}; mismatched: {', '.join(reasons)}"
        )


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(target, "CompressedTensorsMoEMethod", TARGETS[0])
    original = require_callable(cls, "get_moe_method", TARGETS[0])
    if getattr(cls, _MARKER, False):
        if not getattr(original, _MARKER, False):
            raise PatchCompatibilityError("stale HCU CT W4A8 routing marker; restart worker")
        return False
    require_exact_signature(
        original, TARGETS[0], positional=("quant_config", "layer", "layer_name")
    )

    @functools.wraps(original)
    def get_moe_method(quant_config, layer, layer_name):
        # Every other quantization/backend retains upstream selection.
        if not target.current_platform.is_rocm() or layer.moe_config.moe_backend != "deep_gemm":
            return original(quant_config, layer, layer_name)
        quant_config._add_fused_moe_to_target_scheme_map()
        schemes = [
            quant_config.get_scheme_dict(layer, f"{layer_name}.0.{projection}")
            for projection in (
                layer.ckpt_gate_proj_name,
                layer.ckpt_up_proj_name,
                layer.ckpt_down_proj_name,
            )
        ]
        scheme = schemes[0]
        if scheme is None:
            return original(quant_config, layer, layer_name)
        weight = scheme.get("weights")
        activation = scheme.get("input_activations")
        if not quant_config._is_dynamic_token_w4a8_int(weight, activation):
            return original(quant_config, layer, layer_name)
        if not all(other == scheme for other in schemes):
            raise ValueError("All HCU W4A8 MoE projections require the same quantization scheme")
        if scheme.get("format") != "pack-quantized":
            raise ValueError("HCU W4A8 requires the pack-quantized checkpoint format")
        from vllm.config import get_current_vllm_config
        from vllm_hcu.deepseek_v4_runtime import model_architectures
        config = get_current_vllm_config()
        if "DeepseekV41ForCausalLM" not in model_architectures(config):
            raise ValueError("HCU CT W4A8 DeepGEMM bring-up currently supports DeepSeek-V4.1 only")
        # Native DFlash/DSpark draft construction sets PCP1. It does not by
        # itself establish draft DP/EP ownership or make graph execution safe.
        from vllm.compilation import backends as _compile_backends
        is_draft_model = _compile_backends.model_tag != "backbone"
        _validate_topology(config, layer.moe_config, is_draft_model)
        from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8_deepgemm import (
            HCUCompressedTensorsW4A8Int8MoEMethod,
        )
        return HCUCompressedTensorsW4A8Int8MoEMethod(weight, activation, layer.moe_config)

    setattr(get_moe_method, _MARKER, True)
    cls._vllm_hcu_original_get_moe_method = staticmethod(original)
    cls.get_moe_method = staticmethod(get_moe_method)
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
