# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Restore the historical HCU HIP mHC backend without the TileLang branch."""
from __future__ import annotations

import functools
import os
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.layers.mhc"
PATCH_ID = "worker.core_fix.mhc_backend_switch"
_MARKER = "_vllm_hcu_mhc_backend_switch_applied"
_WRAPPER = "_vllm_hcu_mhc_post_hip_wrapper"


def _aiter_mhc_runtime_complete(mhc) -> bool:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm_hcu.model_executor.layers import mhc_runtime

    ops = mhc_runtime.runtime()
    return (
        mhc_runtime.pre_enabled()
        and ops.delayed_available
        and ops.post is not None
        and all(callable(getattr(rocm_aiter_ops, name, None)) for name in (
            "mhc_pre_delayed", "mhc_fused_post_pre_delayed_prefers_unfused",
        ))
    )


def _aiter_mhc_post_runtime_available() -> bool:
    from vllm_hcu.model_executor.layers import mhc_runtime

    return mhc_runtime.runtime().post is not None


def _aiter_mhc_post_supported(x, residual, post_layer_mix, comb_res_mix) -> bool:
    from vllm_hcu.model_executor.layers import mhc_runtime

    return mhc_runtime.post_supported(x, residual, post_layer_mix, comb_res_mix)


def _aiter_mhc_post_enabled() -> bool:
    from vllm_hcu.model_executor.layers import mhc_runtime

    return mhc_runtime.post_enabled()


def _call_aiter_mhc_post(x, residual, post_layer_mix, comb_res_mix):
    import torch

    return torch.ops.vllm.mhc_post_aiter(x, residual, post_layer_mix, comb_res_mix)


def _prepare_mhc_post_patch(mhc):
    post_op = require_class(mhc, "MHCPostOp", f"{TARGET_MODULE}.MHCPostOp")
    original = require_callable(post_op, "forward_hip", "MHCPostOp.forward_hip")
    native = require_callable(post_op, "forward_native", "MHCPostOp.forward_native")
    for function, name in ((original, "forward_hip"), (native, "forward_native")):
        require_exact_signature(
            function, f"{TARGET_MODULE}.MHCPostOp.{name}",
            positional=("self", "x", "residual", "post_layer_mix", "comb_res_mix"),
        )
    available = _aiter_mhc_post_runtime_available()

    @functools.wraps(original)
    def forward_hip(self, x, residual, post_layer_mix, comb_res_mix):
        if (available and _aiter_mhc_post_enabled()
                and _aiter_mhc_post_supported(x, residual, post_layer_mix, comb_res_mix)):
            # An execution failure propagates; it must never trigger a retry on
            # another backend after the operator may already have written data.
            return _call_aiter_mhc_post(x, residual, post_layer_mix, comb_res_mix)
        return self.forward_native(x, residual, post_layer_mix, comb_res_mix)

    setattr(forward_hip, _WRAPPER, True)
    return post_op, original, forward_hip, available


def apply_to_module(module: ModuleType) -> bool:
    mhc = load_exact_module(TARGET_MODULE, module)
    if getattr(mhc, _MARKER, False):
        post_op = require_class(mhc, "MHCPostOp", f"{TARGET_MODULE}.MHCPostOp")
        if (not hasattr(mhc, "_vllm_hcu_original_has_aiter_mhc")
                or not hasattr(mhc, "_vllm_hcu_original_has_tilelang_mhc")
                or not hasattr(mhc, "_vllm_hcu_original_post_forward_hip")
                or not getattr(post_op.forward_hip, _WRAPPER, False)
                or mhc.HAS_TILELANG_MHC is not False):
            raise PatchCompatibilityError(f"required HCU patch marker for {TARGET_MODULE} is stale")
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm_hcu.model_executor.layers import mhc_runtime

        mhc_runtime.install_aiter_bridge(rocm_aiter_ops)
        return False
    for name in ("_vllm_hcu_original_has_aiter_mhc", "_vllm_hcu_original_has_tilelang_mhc",
                 "_vllm_hcu_original_post_forward_hip"):
        if hasattr(mhc, name):
            raise PatchCompatibilityError(f"stale HCU mHC original {name}")
    has_aiter = getattr(mhc, "HAS_AITER_MHC", None)
    has_tilelang = getattr(mhc, "HAS_TILELANG_MHC", None)
    for name, value in (("HAS_AITER_MHC", has_aiter), ("HAS_TILELANG_MHC", has_tilelang)):
        if not isinstance(value, bool):
            raise PatchCompatibilityError(f"required HCU patch target {TARGET_MODULE}.{name} is missing")

    from vllm import _aiter_ops
    from vllm_hcu.model_executor.layers import mhc_runtime

    ops = require_class(_aiter_ops, "rocm_aiter_ops", "vllm._aiter_ops.rocm_aiter_ops")
    post_op, original_post, post_forward, post_available = _prepare_mhc_post_patch(mhc)
    # Bridge the two methods missing from the cold HCU replacement before
    # computing completeness. Their source file is deliberately left untouched.
    before = _aiter_mhc_runtime_complete(mhc)
    mhc_runtime.install_aiter_bridge(ops)
    after = _aiter_mhc_runtime_complete(mhc)
    mhc._vllm_hcu_mhc_complete_before_bridge = before
    mhc._vllm_hcu_mhc_complete_after_bridge = after
    mhc._vllm_hcu_original_has_aiter_mhc = has_aiter
    mhc._vllm_hcu_original_has_tilelang_mhc = has_tilelang
    mhc.HAS_AITER_MHC = bool(has_aiter and after)
    # Importability did not imply a working TileLang kernel in attempt-10.
    # All five official HIP consumers must see False, regardless of discovery.
    mhc.HAS_TILELANG_MHC = False
    mhc._vllm_hcu_original_post_forward_hip = original_post
    post_op.forward_hip = post_forward
    if post_available and _aiter_mhc_post_enabled():
        os.environ.setdefault("AITER_MHC_POST_KERNEL", "v3")
    setattr(mhc, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
