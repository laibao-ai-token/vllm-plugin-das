# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Packed compressed-tensors INT4/INT8 MoE for fixed DeepEP HT on HCU.

CT stores unsigned (q + 8), low-nibble-first INT4 and ordinary channel scales.
HIPC expects signed high-nibble-first bytes and scales in its INT8 (q * 16)
domain. Checkpoint weights remain packed throughout loading.
"""
from __future__ import annotations

import functools

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs


def convert_ct_int4_to_hipc_bytes_(packed: torch.Tensor) -> torch.Tensor:
    """Convert native CT INT32 packing in place, returning an INT8 byte view."""
    if packed.dtype != torch.int32 or packed.ndim != 3 or not packed.is_contiguous():
        raise ValueError("CT W4A8 requires contiguous rank-3 packed INT32 weights")
    data = packed.view(torch.uint8)
    # Numerical INT32 -> INT8 conversion would lose six of eight INT4 values.
    high = data >> 4
    data.bitwise_left_shift_(4).bitwise_or_(high).bitwise_xor_(0x88)
    return data.view(torch.int8)


@functools.lru_cache(maxsize=None)
def _hipc_implicit_scale_factor() -> float:
    """The HIPC W4A8 kernel implies 16x on gfx936 only.

    Ported from the hy4-native W4A8 line, where the compensation is gated on
    the device architecture instead of being applied unconditionally. Keeping
    the guard here means a different HCU arch fails loudly rather than silently
    scaling weights by 16.

    Cached: the factor is constant for the life of the process, and this is
    consulted once per weight-scale tensor on every MoE layer.
    """

    try:
        properties = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        )
        arch = str(properties.gcnArchName).split(":", 1)[0]
    except (AttributeError, AssertionError, RuntimeError):
        return 1.0
    return 16.0 if arch == "gfx936" else 1.0


def _hipc_channel_scale(scale: torch.Tensor) -> torch.Tensor:
    # HCU kernels read channel scales vectorially. Match the reference guard
    # allocation so the final vector cannot cross a 2 MiB device mapping.
    guard = (2 * 1024 * 1024) // 4
    storage = torch.empty(scale.numel() + guard, dtype=torch.float32, device=scale.device)
    view = storage[:scale.numel()].view(scale.shape)
    factor = _hipc_implicit_scale_factor()
    view.copy_(scale)
    if factor != 1.0:
        view.div_(factor)
    return view


def validate_hcu_w4a8(weight_quant, input_quant, moe) -> None:
    """Fail closed outside the initial channel-W4A8, fixed-HT contract."""
    if not (
        weight_quant.num_bits == 4
        and weight_quant.type == "int"
        and weight_quant.strategy == "channel"
        and weight_quant.symmetric
        and not weight_quant.dynamic
        and weight_quant.group_size in (None, -1)
        and input_quant is not None
        and input_quant.num_bits == 8
        and input_quant.type == "int"
        and input_quant.strategy == "token"
        and input_quant.dynamic
        and input_quant.symmetric
    ):
        raise ValueError("HCU DeepGEMM W4A8 requires symmetric channel INT4 and dynamic token INT8")
    parallel = moe.moe_parallel_config
    # Target P uses DeepEP HT; target D uses DeepEP LL.  The expert path
    # receives the backend-produced token metadata and performs its own
    # DeepGEMM permutation or masked batched GEMM, so dispatch/combine backend
    # selection is separate from the packed W4A8 expert GEMM contract. DSpark
    # draft remains on AG-RS.
    if not parallel.use_ep or parallel.all2all_backend not in (
        "deepep_high_throughput",
        "deepep_low_latency",
        "allgather_reducescatter",
    ):
        raise ValueError(
            "HCU CT W4A8 requires EP with all2all_backend "
            f"'deepep_high_throughput' (P), 'deepep_low_latency' (D), or "
            f"'allgather_reducescatter' (draft), got {parallel.all2all_backend!r}"
        )
    if moe.moe_backend != "deep_gemm":
        raise ValueError("HCU CT W4A8 requires moe_backend='deep_gemm'")
    if getattr(moe.activation, "value", moe.activation) != "silu" or not moe.is_act_and_mul:
        raise ValueError("HCU CT W4A8 currently supports gated SiLU only")
    if moe.has_bias:
        raise ValueError("HCU CT W4A8 does not support expert bias")
    if moe.in_dtype != torch.bfloat16:
        raise ValueError("HCU CT W4A8 requires BF16 model activations")
    if (
        getattr(moe, "swiglu_alpha", None) not in (None, 1.0)
        or getattr(moe, "swiglu_beta", None) not in (None, 0.0)
    ):
        raise ValueError("HCU CT W4A8 supports standard SwiGLU alpha=1/beta=0 only")


class HCUCompressedTensorsW4A8Int8MoEMethod(CompressedTensorsMoEMethod):
    """Channel W4A8 with native packed checkpoint loading and HIPC experts."""

    def __init__(self, weight_quant, input_quant, moe):
        super().__init__(moe)
        validate_hcu_w4a8(weight_quant, input_quant, moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self._layout_state: str | None = None

    @property
    def is_monolithic(self) -> bool:
        return False

    def create_weights(
        self, layer, num_experts, hidden_size, intermediate_size_per_partition,
        params_dtype, **extra_weight_attrs,
    ):
        del params_dtype
        intermediate = intermediate_size_per_partition
        # The contiguous HIPC kernel stores four 128-column N tiles without
        # an N-tail mask. Both GEMM outputs must therefore be multiples of 512.
        if hidden_size % 512 or intermediate % 256:
            raise ValueError("HCU CT W4A8 requires hidden divisible by 512 and intermediate by 256")
        attrs = {**extra_weight_attrs, "is_transposed": False, "quant_method": "channel"}
        original_loader = attrs.get("weight_loader")
        if callable(original_loader):
            @functools.wraps(original_loader)
            def load_once(*args, **kwargs):
                if self._layout_state is not None:
                    raise RuntimeError("HCU CT W4A8 hot reload is not supported; restart worker")
                return original_loader(*args, **kwargs)
            attrs["weight_loader"] = load_once
        for name, shape, dtype in (
            ("w13_weight_packed", (num_experts, 2 * intermediate, hidden_size // 8), torch.int32),
            ("w2_weight_packed", (num_experts, hidden_size, intermediate // 8), torch.int32),
            ("w13_weight_scale", (num_experts, 2 * intermediate, 1), torch.float32),
            ("w2_weight_scale", (num_experts, hidden_size, 1), torch.float32),
            ("w13_weight_shape", (num_experts, 2), torch.int64),
            ("w2_weight_shape", (num_experts, 2), torch.int64),
        ):
            parameter = Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            layer.register_parameter(name, parameter)
            set_weight_attrs(parameter, attrs)

    def get_fused_moe_quant_config(self, layer):
        if self.moe_quant_config is not None:
            return self.moe_quant_config
        return FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=_hipc_channel_scale(layer.w13_weight_scale),
            w2_scale=_hipc_channel_scale(layer.w2_weight_scale),
            per_act_token_quant=True,
            per_out_ch_quant=False,
            block_shape=None,
            weight_dtype="int4",
            gemm1_clamp_limit=getattr(self.moe, "swiglu_limit", None),
        )

    @staticmethod
    def _signature(layer):
        return tuple(
            (id(value), value.data_ptr(), value._version, tuple(value.shape))
            for value in (layer.w13_weight_packed, layer.w2_weight_packed,
                          layer.w13_weight_scale, layer.w2_weight_scale)
        )

    def process_weights_after_loading(self, layer):
        if self._layout_state == "ready":
            if self._signature(layer) == self._packed_signature:
                return
            raise RuntimeError("HCU CT W4A8 weight reload is not supported; restart worker")
        if self._layout_state is not None:
            raise RuntimeError("HCU CT W4A8 packing previously failed; restart the worker")
        from vllm.model_executor.layers.fused_moe import modular_kernel as mk
        from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
        from vllm_hcu.model_executor.layers.quantization.slimquant_w4a8_deepgemm_runtime import (
            DeepEPDeepGemmW4A8BatchedExperts,
            DeepEPDeepGemmW4A8ContiguousExperts,
            pack_w4a8_moe_hipc_weight,
        )

        # Convert one projection at a time; retain ONE resident weight owner.
        # General SlimQuant post-load clones both, doubling resident weights.
        self._layout_state = "packing"
        for name in ("w13_weight_packed", "w2_weight_packed"):
            raw = getattr(layer, name)
            with torch.no_grad():
                converted = convert_ct_int4_to_hipc_bytes_(raw)
                packed = pack_w4a8_moe_hipc_weight(converted)
            if (
                packed.dtype != torch.int8 or tuple(packed.shape) != tuple(converted.shape)
                or packed.data_ptr() != converted.data_ptr()
            ):
                raise RuntimeError("HIPC W4A8 packer must retain packed weight storage and shape")
            replace_parameter(layer, name, packed)
        # LightOp expands signed INT4 nibbles into the high half of INT8, so the
        # kernel wants the checkpoint scales divided by 16. The checkpoint
        # scale Parameters stay untouched; ``get_fused_moe_quant_config``
        # derives the divided copies.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        prepare = maybe_make_prepare_finalize(
            moe=self.moe, quant_config=self.moe_quant_config,
            routing_tables=layer._expert_routing_tables(),
            allow_new_interface=True, use_monolithic=False,
        )
        if prepare is None:
            # DeepEP HT (target) or naive AG-RS (DSpark draft) prepare/finalize.
            raise RuntimeError(
                "HCU CT W4A8 did not construct a prepare/finalize for "
                f"all2all_backend={self.moe.moe_parallel_config.all2all_backend!r}"
            )
        if prepare.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
            max_tokens = prepare.max_num_tokens_per_rank()
            if max_tokens is None:
                raise RuntimeError(
                    "HCU CT W4A8 low-latency prepare/finalize did not expose "
                    "max_num_tokens_per_rank"
                )
            experts = DeepEPDeepGemmW4A8BatchedExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                max_num_tokens=max_tokens,
                num_dispatchers=prepare.num_dispatchers(),
            )
        else:
            experts = DeepEPDeepGemmW4A8ContiguousExperts(
                self.moe, self.moe_quant_config
            )
            # Contiguous experts consume the packed checkpoint layout directly.
            experts._deepgemm_w13 = layer.w13_weight_packed
            experts._deepgemm_w2 = layer.w2_weight_packed
        if isinstance(experts, DeepEPDeepGemmW4A8BatchedExperts):
            # The masked HIPC kernel consumes the N32-shuffled view of the
            # packed weights. Build it from the already-packed checkpoint
            # parameters: this layer registers only the ``*_packed`` names, so
            # the SlimQuant post-load hook (which re-packs ``w13_weight``)
            # cannot be reused here.
            experts.adopt_packed_weights(layer)
        self.moe_kernel = mk.FusedMoEKernel(prepare, experts)
        self._packed_signature = self._signature(layer)
        self._layout_state = "ready"

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        if self._layout_state != "ready" or self.moe_kernel is None:
            raise RuntimeError("HCU CT W4A8 weights must be packed before forward")
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight_packed, w2=layer.w2_weight_packed,
            topk_weights=topk_weights, topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
