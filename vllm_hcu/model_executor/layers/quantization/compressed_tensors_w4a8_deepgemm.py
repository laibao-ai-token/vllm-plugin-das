# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Packed compressed-tensors INT4/INT8 MoE for fixed DeepEP HT/LL on HCU.

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


def _hipc_channel_scale(scale: torch.Tensor) -> torch.Tensor:
    # HCU kernels read channel scales vectorially. Match the reference guard
    # allocation so the final vector cannot cross a 2 MiB device mapping.
    guard = (2 * 1024 * 1024) // 4
    storage = torch.empty(scale.numel() + guard, dtype=torch.float32, device=scale.device)
    view = storage[:scale.numel()].view(scale.shape)
    view.copy_(scale).div_(16)
    return view


def _draft_all2all_manager(moe):
    # The producer is optional until its separately reviewed slice is applied.
    # Only absence of this exact module means target/default ownership.
    try:
        from vllm_hcu.dspark_runtime import get_dspark_all2all_manager
    except ModuleNotFoundError as exc:
        if exc.name != "vllm_hcu.dspark_runtime":
            raise
        if getattr(moe, "_hcu_dspark_all2all_manager", None) is not None:
            raise RuntimeError("DSpark manager is bound but its producer API is unavailable") from exc
        return None
    manager = get_dspark_all2all_manager(moe)
    if manager is None:
        return None
    parallel = moe.moe_parallel_config
    if parallel.all2all_backend != "allgather_reducescatter":
        raise RuntimeError("DSpark draft manager cannot own fixed target DeepEP experts")
    if (
        manager.world_size != parallel.ep_size
        or manager.dp_world_size != parallel.dp_size
    ):
        raise RuntimeError("DSpark manager does not match the loaded MoE DP/EP membership")
    return manager


def _draft_token_capacity(moe, manager, stored=None):
    if manager is None:
        if stored is not None:
            raise RuntimeError("DSpark capacity lost its bound draft manager")
        return None
    from vllm_hcu.dspark_runtime import get_dspark_token_capacity

    active = get_dspark_token_capacity()
    capacity = active if active is not None else stored
    if capacity is None:
        raise RuntimeError("DSpark manager has no active or consumer-bound token capacity")
    if stored is not None and active is not None and active != stored:
        raise RuntimeError("DSpark token capacity changed after MoE construction")
    fields = (
        capacity.speculative_tokens, capacity.max_requests,
        capacity.query_tokens_per_request, capacity.query_tokens_per_rank,
        capacity.verify_tokens_per_rank, capacity.capacity_per_rank,
    )
    if any(type(value) is not int or value <= 0 for value in fields):
        raise RuntimeError("DSpark producer returned an invalid token capacity contract")
    declared = getattr(moe, "max_num_tokens", None)
    if (
        type(declared) is not int or declared != capacity.capacity_per_rank
        or max(capacity.query_tokens_per_rank, capacity.verify_tokens_per_rank) > declared
    ):
        raise RuntimeError("DSpark producer token bounds do not match the loaded MoE capacity")
    return capacity


def validate_hcu_w4a8(weight_quant, input_quant, moe) -> None:
    """Fail closed outside channel-W4A8 with a fixed HT/LL or explicit AG-RS path."""
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
    # HIPC experts support fixed HT/LL and an explicitly configured
    # AG-RS prepare/finalize path. The paired community core does not itself
    # switch DSpark to AG-RS; draft EP/groups/backend integration is separate.
    # Accepting the expert method here does not validate that full deployment.
    if not parallel.use_ep or parallel.all2all_backend not in (
        "deepep_high_throughput",
        "deepep_low_latency",
        "allgather_reducescatter",
    ):
        raise ValueError(
            "HCU CT W4A8 requires EP with all2all_backend "
            f"'deepep_high_throughput', 'deepep_low_latency' or 'allgather_reducescatter' "
            f"(draft), got {parallel.all2all_backend!r}"
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
        self._ll_token_capacity: int | None = None
        self._dspark_token_capacity = None

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
                manager = _draft_all2all_manager(self.moe)
                if manager is not self._construction_manager:
                    raise RuntimeError("HCU CT W4A8 manager ownership changed after kernel construction")
                _draft_token_capacity(self.moe, manager, self._dspark_token_capacity)
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
            view_w4a8_moe_hipc_weight_n32_layout,
        )

        manager = _draft_all2all_manager(self.moe)
        draft_capacity = _draft_token_capacity(self.moe, manager)
        fixed_ll = self.moe.moe_parallel_config.all2all_backend == "deepep_low_latency"
        if fixed_ll:
            declared_capacity = getattr(self.moe, "max_num_tokens", None)
            expected_dispatchers = getattr(self.moe.moe_parallel_config, "ep_size", None)
            if (
                type(declared_capacity) is not int or declared_capacity <= 0
                or type(expected_dispatchers) is not int or expected_dispatchers <= 0
            ):
                raise RuntimeError("HCU CT W4A8 LL requires declared MoE capacity and EP size")

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
        # LightOp expands signed nibbles into the high half of INT8. Keep
        # checkpoint scale Parameters unchanged and use derived /16 scales.
        self.moe_quant_config = None
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        prepare = maybe_make_prepare_finalize(
            moe=self.moe, quant_config=self.moe_quant_config,
            routing_tables=layer._expert_routing_tables(),
            allow_new_interface=True, use_monolithic=False,
            all2all_manager=manager,
            # These experts consume quantized inputs, with no deferred dtype override.
            input_dtype=None,
        )
        if prepare is None:
            # Fixed DeepEP HT/LL or naive AG-RS (DSpark draft) prepare/finalize.
            raise RuntimeError(
                "HCU CT W4A8 did not construct a prepare/finalize for "
                f"all2all_backend={self.moe.moe_parallel_config.all2all_backend!r}"
            )
        if fixed_ll:
            from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
                DeepEPLLPrepareAndFinalize,
            )

            # Fixed LL must pair batched dispatch with masked N32 experts. Never
            # accept an auto/HT delegate or silently dispatch unquantized data.
            if (
                not isinstance(prepare, DeepEPLLPrepareAndFinalize)
                or not getattr(prepare, "use_int8_dispatch", False)
                or getattr(prepare, "use_fp8_dispatch", False)
                or getattr(prepare, "use_ue8m0_dispatch", False)
            ):
                raise RuntimeError("HCU CT W4A8 requires fixed DeepEP LL INT8 prepare/finalize")
            capacity = prepare.max_num_tokens_per_rank()
            dispatchers = prepare.num_dispatchers()
            if (
                type(capacity) is not int or capacity <= 0
                or type(dispatchers) is not int or dispatchers <= 0
            ):
                raise RuntimeError("HCU CT W4A8 LL requires positive token capacity and dispatchers")
            if capacity != declared_capacity or dispatchers != expected_dispatchers:
                raise RuntimeError(
                    "HCU CT W4A8 LL prepare/finalize capacity or dispatchers "
                    "does not match the loaded MoE config"
                )
            experts = DeepEPDeepGemmW4A8BatchedExperts(
                self.moe, self.moe_quant_config, capacity, dispatchers,
            )
            # The checkpoint scale dimensions describe the loaded matrices.
            experts._hcu_logical_n = layer.w13_weight_scale.size(1)
            experts._hcu_logical_k = layer.w2_weight_scale.size(1)
            owners = (layer.w13_weight_packed, layer.w2_weight_packed)
            views = tuple(view_w4a8_moe_hipc_weight_n32_layout(w) for w in owners)
            if any(
                view.dtype != torch.int8
                or tuple(view.shape) != (
                    owner.size(0), owner.size(2) // 32, owner.size(1) // 32, 4, 32, 8,
                )
                or view.data_ptr() != owner.data_ptr()
                or view.untyped_storage().data_ptr() != owner.untyped_storage().data_ptr()
                for view, owner in zip(views, owners)
            ):
                raise RuntimeError("HCU CT W4A8 LL N32 views must alias packed weight storage")
            experts._deepgemm_w13, experts._deepgemm_w2 = views
            self._ll_token_capacity = capacity
        else:
            experts = DeepEPDeepGemmW4A8ContiguousExperts(self.moe, self.moe_quant_config)
            experts._deepgemm_w13 = layer.w13_weight_packed
            experts._deepgemm_w2 = layer.w2_weight_packed
        self.moe_kernel = mk.FusedMoEKernel(prepare, experts)
        # None is an ownership choice too; a ready kernel cannot adopt a scope.
        self._construction_manager = manager
        if manager is not None:
            self._dspark_token_capacity = draft_capacity
            self.moe._hcu_dspark_all2all_manager = manager
        self._packed_signature = self._signature(layer)
        self._layout_state = "ready"

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        if self._layout_state != "ready" or self.moe_kernel is None:
            raise RuntimeError("HCU CT W4A8 weights must be packed before forward")
        if self._ll_token_capacity is not None and x.size(0) > self._ll_token_capacity:
            raise ValueError("HCU CT W4A8 LL forward exceeds checked token capacity per rank")
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
