# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Restore HCU ordinary/delayed Pre routing from kernel-opt-mhc-pr223.

Pre 5120 is supported by the historical GPU evidence; Post 5120 is not. Thus a
5120 delayed seam uses native Post followed by AITER Pre, while the combined
Post+Pre op takes its complete native path. No execution error is intercepted.
"""
from __future__ import annotations

import functools
from types import ModuleType

import torch

from vllm_hcu.model_executor.layers import mhc_runtime as runtime

from . import patch_mhc_backend as post_backend
from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.mhc"
PATCH_ID = "worker.core_fix.mhc_pre_backend"
_MARKER = "_vllm_hcu_mhc_pre_backend_applied"
_WRAPPER = "_vllm_hcu_mhc_pre_hip_wrapper"
_COMMON = (
    "residual", "fn", "hc_scale", "hc_base", "rms_eps", "hc_pre_eps",
    "hc_sinkhorn_eps", "hc_post_mult_value", "sinkhorn_repeat",
)
_BINDINGS = (
    ("MHCPreOp", "_vllm_hcu_original_pre_forward_hip"),
    ("MHCPreDelayedOp", "_vllm_hcu_original_delayed_pre_forward_hip"),
    ("MHCFusedPostPreOp", "_vllm_hcu_original_fused_post_pre_forward_hip"),
)


def _prepare(module, name, parameters, defaults):
    cls = require_class(module, name, f"{TARGET_MODULE}.{name}")
    original = require_callable(cls, "forward_hip", f"{name}.forward_hip")
    native = require_callable(cls, "forward_native", f"{name}.forward_native")
    for function, method in ((original, "forward_hip"), (native, "forward_native")):
        require_exact_signature(function, f"{name}.{method}",
                                positional=parameters, defaults=defaults)
    return cls, original


def apply_to_module(module: ModuleType) -> bool:
    module = load_exact_module(TARGET_MODULE, module)
    if getattr(module, _MARKER, False):
        for cls_name, backup in _BINDINGS:
            cls = require_class(module, cls_name, f"{TARGET_MODULE}.{cls_name}")
            if not hasattr(module, backup) or not getattr(cls.forward_hip, _WRAPPER, False):
                raise PatchCompatibilityError("stale HCU mHC Pre patch marker")
        return False
    if not getattr(module, "_vllm_hcu_mhc_backend_switch_applied", False):
        raise PatchCompatibilityError("HCU mHC backend must be applied before Pre routing")
    for _, backup in _BINDINGS:
        if hasattr(module, backup):
            raise PatchCompatibilityError(f"stale HCU mHC original {backup}")
    pre_cls, original_pre = _prepare(
        module, "MHCPreOp", ("self", *_COMMON, "n_splits", "norm_weight", "norm_eps"),
        {"n_splits": 1, "norm_weight": None, "norm_eps": 0.0},
    )
    delayed_cls, original_delayed = _prepare(
        module, "MHCPreDelayedOp",
        ("self", *_COMMON, "pre_mix", "x", "norm_weight", "norm_eps",
         "sublayer_out", "post_layer_mix", "comb_res_mix"),
        {"pre_mix": None, "x": None, "norm_weight": None, "norm_eps": 1e-6,
         "sublayer_out": None, "post_layer_mix": None, "comb_res_mix": None},
    )
    fused_cls, original_fused = _prepare(
        module, "MHCFusedPostPreOp",
        ("self", "x", "residual", "post_layer_mix", "comb_res_mix", *_COMMON[1:],
         "n_splits", "tile_n", "norm_weight", "norm_eps"),
        {"n_splits": 1, "tile_n": 1, "norm_weight": None, "norm_eps": 0.0},
    )
    apply_norm = require_callable(module, "_apply_mhc_norm", "mHC decoder RMSNorm")
    require_callable(delayed_cls, "_maybe_post", "MHCPreDelayedOp._maybe_post")
    capabilities = runtime.runtime()
    runtime.register_delayed_op()

    @functools.wraps(original_pre)
    def pre_forward(
        self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1,
        norm_weight=None, norm_eps=0.0,
    ):
        args = (residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
        if (runtime.pre_enabled() and capabilities.pre is not None
                and runtime.pre_supported(residual, fn, hc_scale, hc_base)):
            post, comb, layer_input = torch.ops.vllm.mhc_pre_aiter(
                *args, n_splits, None, norm_eps,
            )
        else:
            post, comb, layer_input = self.forward_native(*args, n_splits)
        # HCU Pre never fuses the decoder Norm. Apply it exactly once here.
        return post, comb, apply_norm(layer_input, norm_weight, norm_eps)

    @functools.wraps(original_delayed)
    def delayed_forward(
        self, residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, pre_mix=None,
        x=None, norm_weight=None, norm_eps=1e-6, sublayer_out=None,
        post_layer_mix=None, comb_res_mix=None,
    ):
        args = (residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
        tail = (pre_mix, x, norm_weight, norm_eps, sublayer_out,
                post_layer_mix, comb_res_mix)
        if not (runtime.pre_enabled() and capabilities.delayed_available
                and runtime.pre_supported(residual, fn, hc_scale, hc_base, pre_mix, x)):
            # This native method already applies its Post and Norm. Returning
            # directly also prevents re-entering the official wider AITER gate.
            return self.forward_native(*args, *tail)
        residual = self._maybe_post(residual, sublayer_out, post_layer_mix, comb_res_mix)
        post, comb, layer_input, next_pre = torch.ops.vllm.hcu_mhc_pre_delayed(
            residual, *args[1:], pre_mix, x,
        )
        return (residual, post, comb,
                apply_norm(layer_input, norm_weight, norm_eps), next_pre)

    @functools.wraps(original_fused)
    def fused_forward(
        self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None, norm_eps=0.0,
    ):
        args = (x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
                hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult_value, sinkhorn_repeat)
        if (runtime.pre_enabled() and post_backend._aiter_mhc_post_enabled()
                and capabilities.pre is not None and capabilities.post is not None
                and runtime.pre_supported(residual, fn, hc_scale, hc_base)
                and post_backend._aiter_mhc_post_supported(
                    x, residual, post_layer_mix, comb_res_mix)):
            # The HCU cold replacement implements this official custom op as
            # separate HIP Post + Pre, not an opt-in fused AITER kernel.
            updated, post, comb, layer_input = torch.ops.vllm.mhc_fused_post_pre_aiter(
                *args, n_splits, tile_n, None, norm_eps,
            )
        else:
            updated, post, comb, layer_input = self.forward_native(*args)
        return updated, post, comb, apply_norm(layer_input, norm_weight, norm_eps)

    for function in (pre_forward, delayed_forward, fused_forward):
        setattr(function, _WRAPPER, True)
    module._vllm_hcu_original_pre_forward_hip = original_pre
    module._vllm_hcu_original_delayed_pre_forward_hip = original_delayed
    module._vllm_hcu_original_fused_post_pre_forward_hip = original_fused
    pre_cls.forward_hip = pre_forward
    delayed_cls.forward_hip = delayed_forward
    fused_cls.forward_hip = fused_forward
    setattr(module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
