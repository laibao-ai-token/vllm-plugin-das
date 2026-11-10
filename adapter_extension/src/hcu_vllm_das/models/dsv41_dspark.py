# SPDX-License-Identifier: Apache-2.0
"""Load-scoped CT scale aliases without copying the native V41 draft loader.

The public model registry selects this subclass. Inherited construction,
forward, mapping and loading stay owned by Core; no constructor parameter table
or native weight loop is mirrored. Only custom Channel-FP8 loading exposes a
thread-local lookup view and normalizes its synthesized return names. No alias
Parameters are registered and no instance method is replaced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import os
from threading import RLock, get_ident

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
)
from vllm.models.deepseek_v41.amd.dspark import (
    DSparkDeepseekV4ForCausalLM as NativeDSpark,
)

from hcu_vllm_das.dsv41_activation import check_health
from hcu_vllm_das.quantization.dsv41_channel_fp8 import HcuDSV41ChannelFP8Config

_STATE = "_hcu_dsv41_scale_alias_load"
_gate = RLock()


def _after_fork_child():
    global _gate
    _gate = RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


@dataclass
class _AliasLoad:
    pid: int
    thread: int
    aliases: dict[str, str] = field(default_factory=dict)


def _alias(name):
    if name.endswith((".weight_scale", "_weight_scale")):
        return name + "_inv"
    if name.endswith((".weight_scale_inv", "_weight_scale_inv")):
        return name.removesuffix("_inv")
    return None


class HcuDSparkDeepseekV41ForCausalLM(NativeDSpark):
    """Keep native draft construction/loading; own CT DP communication policy."""

    def __init__(self, *, vllm_config, prefix=""):
        from vllm.config import set_current_vllm_config
        from hcu_vllm_das.dspark_agrs import _owned_quant_kind, draft_config

        # CT W4A8 and Channel-FP8 drafts both take a private AG-RS construction
        # config so they do not allocate a second DeepEP/RocSHMEM LL buffer on
        # top of the target's already-resident weights.
        if (
            _owned_quant_kind(vllm_config) is not None
            and vllm_config.parallel_config.data_parallel_size > 1
        ):
            local = draft_config(vllm_config)
            # Native draft decoder construction reads current_config as well as
            # its explicit argument. Keep both references on the same private
            # copy and retain native compiler live tables for layer discovery.
            with set_current_vllm_config(local):
                super().__init__(vllm_config=local, prefix=prefix)
        else:
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    def _is_ct_scale(self, name, prefix):
        relative = name[len(prefix) + 1:] if prefix else name
        parent, _, leaf = relative.rpartition(".")
        owner = self.get_submodule(parent) if parent else self
        method = getattr(owner, "quant_method", None)
        leaf = leaf.removesuffix("_inv")
        if leaf == "weight_scale" and isinstance(method, CompressedTensorsLinearMethod):
            return isinstance(owner.scheme, CompressedTensorsW8A8Fp8)
        return leaf in ("w13_weight_scale", "w2_weight_scale") and isinstance(
            method, CompressedTensorsW8A8Fp8MoEMethod
        )

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        native = super().named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        )
        state = vars(self).get(_STATE)
        # Outside the owning load/thread, retain ordinary canonical enumeration.
        if state is None or state.pid != os.getpid() or state.thread != get_ident():
            yield from native
            return
        names = set()
        candidates = []
        for name, parameter in native:
            names.add(name)
            yield name, parameter
            alias = _alias(name)
            if alias is not None and self._is_ct_scale(name, prefix):
                candidates.append((alias, name, parameter))
        # Discover all real names first. A real direct/inverse pair always wins.
        for alias, name, parameter in candidates:
            if alias not in names:
                state.aliases[alias] = name
                yield alias, parameter

    def load_weights(self, weights):
        if not isinstance(self.quant_config, HcuDSV41ChannelFP8Config):
            return super().load_weights(weights)
        check_health()
        for function, expected in (
            (NativeDSpark.load_weights, ("self", "weights")),
            (NativeDSpark.named_parameters, ("self", "prefix", "recurse", "remove_duplicate")),
        ):
            if tuple(inspect.signature(function).parameters) != expected:
                raise RuntimeError("DSV4.1 native draft loading signature changed")
        if "named_parameters" in vars(self):
            raise RuntimeError("DSV4.1 draft has a foreign named_parameters instance override")
        with _gate:
            if _STATE in vars(self):
                raise RuntimeError("concurrent or reentrant DSV4.1 draft loading is unsupported")
            state = _AliasLoad(os.getpid(), get_ident())
            setattr(self, _STATE, state)
        try:
            loaded = super().load_weights(weights)
            if not isinstance(loaded, set) or not state.aliases:
                return loaded
            return {state.aliases.get(name, name) for name in loaded}
        finally:
            with _gate:
                delattr(self, _STATE)
