# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU AITER HIP mHC, re-hosted from kernel-opt-mhc-pr223.

The delayed Pre keeps the previous-sublayer mix contract, the first-layer narrow
projection, split-K=1, and the separate decoder RMSNorm. Historical Pre evidence
includes hidden=5120 (mhc_full_integration.md:59); Post v3 evidence does not, so
Post and the combined Post+Pre route explicitly fall back for that width.
TileLang and fused AITER kernels are outside this port. Optional capability
absence selects native before dispatch; execution failures are never swallowed.
"""
from __future__ import annotations

import functools
import importlib
import logging
from dataclasses import dataclass
from typing import Callable

import torch

from vllm_hcu.platforms import envs as henvs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MhcRuntime:
    pre: Callable | None = None
    post: Callable | None = None
    project: Callable | None = None
    big_fuse: Callable | None = None

    @property
    def delayed_available(self) -> bool:
        return self.project is not None and self.big_fuse is not None


def _operator(module, name):
    value = getattr(module, name, None)
    return value if callable(value) else None


@functools.lru_cache(maxsize=1)
def get_runtime() -> MhcRuntime:
    try:
        native = importlib.import_module("aiter.ops.mhc")
    except (ImportError, AttributeError) as exc:
        logger.warning("AITER HIP mHC unavailable; using native mHC: %s", exc)
        native = None
    return MhcRuntime(
        pre=_operator(native, "mhc_pre"),
        post=_operator(native, "mhc_post"),
        project=_operator(native, "mhc_pre_gemm_sqrsum"),
        big_fuse=_operator(native, "mhc_pre_big_fuse"),
    )


def runtime() -> MhcRuntime:
    # Read both registered switches. No AITER TileLang module is imported, even
    # when its package is installed and importable in DTK2610.
    if henvs.VLLM_HCU_USE_AITER_TILELANG_MHC or henvs.VLLM_HCU_USE_AITER_MHC_FUSED:
        raise ValueError(
            "HCU mHC currently supports AITER HIP only; set "
            "VLLM_HCU_USE_AITER_TILELANG_MHC=0 and VLLM_HCU_USE_AITER_MHC_FUSED=0"
        )
    return get_runtime()


def pre_enabled() -> bool:
    return henvs.optional_custom_op_enabled(
        henvs.VLLM_HCU_USE_AITER_MHC and henvs.VLLM_HCU_USE_AITER_MHC_PRE
    )


def post_enabled() -> bool:
    return henvs.optional_custom_op_enabled(
        henvs.VLLM_HCU_USE_AITER_MHC and henvs.VLLM_HCU_USE_AITER_MHC_POST
    )


def pre_supported(residual, fn, hc_scale, hc_base, pre_mix=None, x=None) -> bool:
    if residual.ndim != 3 or residual.shape[1] != 4:
        return False
    tokens, hc, hidden = residual.shape
    width = hc * hidden if x is None else hidden
    tensors = [residual, fn, hc_scale, hc_base]
    if pre_mix is not None:
        if pre_mix.shape != (tokens, hc) or pre_mix.dtype != torch.float32:
            return False
        tensors.append(pre_mix)
    if x is not None:
        if x.shape != (tokens, hidden) or x.dtype != torch.bfloat16:
            return False
        tensors.append(x)
    return (
        hidden >= 1024
        and hidden % 1024 == 0
        and residual.dtype == torch.bfloat16
        and fn.shape == (hc * (2 + hc), width)
        and hc_scale.shape == (3,)
        and hc_base.shape == (hc * (2 + hc),)
        and fn.dtype == hc_scale.dtype == hc_base.dtype == torch.float32
        and all(t.is_cuda and t.device == residual.device
                and t.is_contiguous() for t in tensors)
    )


def post_supported(x, residual, post_layer_mix, comb_res_mix) -> bool:
    if residual.ndim < 3:
        return False
    hidden = residual.shape[-1]
    outer = residual.shape[:-2]
    tensors = (x, residual, post_layer_mix, comb_res_mix)
    return (
        residual.numel() > 0
        and residual.shape[-2] == 4
        and hidden >= 1024
        and hidden % 1024 == 0
        # Post 5120 has no validation in mhc_post_v3_validation.md:19.
        and hidden != 5120
        and x.shape == (*outer, hidden)
        and post_layer_mix.shape in ((*outer, 4), (*outer, 4, 1))
        and comb_res_mix.shape == (*outer, 4, 4)
        and residual.dtype in (torch.bfloat16, torch.float16)
        and x.dtype == residual.dtype
        and post_layer_mix.dtype == comb_res_mix.dtype == torch.float32
        and all(t.is_cuda and t.device == residual.device
                and t.is_contiguous() for t in tensors)
    )


def _empty_pre(residual):
    tokens, hc, hidden = residual.shape
    options = {"device": residual.device, "dtype": torch.float32}
    return (
        torch.empty(tokens, hc, 1, **options),
        torch.empty(tokens, hc, hc, **options),
        torch.empty(tokens, hidden, device=residual.device, dtype=residual.dtype),
        torch.empty(tokens, hc, **options),
    )


def _staged_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                n_splits=1, x=None):
    ops = runtime()
    if not ops.delayed_available:
        raise RuntimeError("AITER staged mHC Pre is unavailable")
    tokens, hc, hidden = residual.shape
    if tokens == 0:
        post, comb, layer_input, _ = _empty_pre(residual)
        empty_gemm = torch.empty(1, 0, hc * (2 + hc), device=residual.device)
        empty_sqr = torch.empty(1, 0, device=residual.device)
        return post, comb, layer_input, empty_gemm, empty_sqr
    projected = residual
    if x is not None:
        projected = x.unsqueeze(1).expand(-1, hc, -1).contiguous()
        expanded_fn = fn.new_zeros(fn.shape[0], hc * hidden)
        expanded_fn[:, :hidden].copy_(fn)
        fn = expanded_fn
    tile_k = 64
    if n_splits < 1 or (hc * hidden) % (n_splits * tile_k):
        raise ValueError("mHC split-K must divide the projection width")
    if hc * hidden // n_splits < 2 * tile_k:
        raise ValueError("mHC split-K is too large for the projection width")
    padded = torch.empty(n_splits, tokens, 32, dtype=torch.float32,
                         device=residual.device)
    gemm = padded[:, :, :hc * (2 + hc)]
    sqrsum = torch.empty(n_splits, tokens, dtype=torch.float32,
                         device=residual.device)
    post, comb, layer_input, _ = _empty_pre(residual)
    ops.project(gemm, sqrsum, projected, fn, tile_k, False)
    ops.big_fuse(post, comb, layer_input, gemm, sqrsum, hc_scale, hc_base,
                 residual, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                 hc_post_mult_value, sinkhorn_repeat)
    return post, comb, layer_input, gemm, sqrsum


def delayed_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if residual.shape[0] == 0:
        return _empty_pre(residual)
    post, comb, _, gemm, sqrsum = _staged_pre(
        residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
        hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, x=x,
    )
    hc, hidden = residual.shape[1:]
    next_pre = torch.ops.vllm.mhc_pre_mix_triton(
        gemm, sqrsum, hc_scale, hc_base, hc, hc * hidden, rms_eps, hc_pre_eps,
    )
    layer_input = (residual[:, 0].clone() if pre_mix is None
                   else torch.ops.vllm.hc_collapse_triton(residual, pre_mix))
    return post, comb, layer_input, next_pre


def delayed_pre_fake(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                     hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                     pre_mix=None, x=None):
    return _empty_pre(residual)


_REGISTERED = False


def register_delayed_op() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="hcu_mhc_pre_delayed", op_func=delayed_pre,
        mutates_args=[], fake_impl=delayed_pre_fake,
    )
    _REGISTERED = True


def post(x, residual, post_layer_mix, comb_res_mix):
    ops = runtime()
    if residual.numel() == 0:
        return torch.empty_like(residual)
    if not (post_enabled() and ops.post is not None
            and post_supported(x, residual, post_layer_mix, comb_res_mix)):
        from vllm.model_executor.kernels.mhc import mhc_post_torch

        return mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
    out = torch.empty_like(residual)
    ops.post(out, x, residual, post_layer_mix.reshape(-1, 4, 1), comb_res_mix)
    return out


def _bridge_pre_delayed(
    residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
    hc_post_mult_value, sinkhorn_repeat, pre_mix=None, sublayer_out=None,
    post_layer_mix=None, comb_res_mix=None, residual_out=None,
):
    """0.29 rocm_aiter_ops.mhc_pre_delayed, without editing its dirty source."""
    hc, hidden = residual.shape[-2:]
    outer = residual.shape[:-2]
    stream = residual.reshape(-1, hc, hidden)
    if sublayer_out is not None:
        assert residual_out is not None
        assert post_layer_mix is not None and comb_res_mix is not None
        updated = post(sublayer_out.reshape(-1, hidden), stream,
                       post_layer_mix.reshape(-1, hc, 1),
                       comb_res_mix.reshape(-1, hc, hc))
        residual_out.copy_(updated.reshape_as(residual_out))
        stream = residual_out.reshape(-1, hc, hidden)
    result = delayed_pre(
        stream, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat,
        None if pre_mix is None else pre_mix.reshape(-1, hc),
    )
    next_post, comb, layer_input, next_pre = result
    return (next_post.reshape(*outer, hc, 1), comb.reshape(*outer, hc, hc),
            layer_input.reshape(*outer, hidden), next_pre.reshape(*outer, hc))


def _bridge_prefers_unfused(num_tokens: int) -> bool:
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        threshold = {"gfx950": 1024, "gfx942": 128, "gfx1250": 1024}.get(
            get_gfx_runtime(), 1024
        )
    except Exception:
        threshold = 1024
    return num_tokens >= threshold


def install_aiter_bridge(ops: type) -> bool:
    methods = {
        "mhc_pre_delayed": _bridge_pre_delayed,
        "mhc_fused_post_pre_delayed_prefers_unfused": _bridge_prefers_unfused,
    }
    marker = "_vllm_hcu_mhc_hip_bridge_applied"
    originals = "_vllm_hcu_original_mhc_hip_bridge"
    if getattr(ops, marker, False):
        if not hasattr(ops, originals) or any(
            getattr(ops, name, None) is not function
            for name, function in methods.items()
        ):
            raise RuntimeError("stale HCU mHC HIP bridge marker")
        return False
    if hasattr(ops, originals):
        raise RuntimeError("stale HCU mHC HIP bridge originals")
    saved = {name: (hasattr(ops, name), getattr(ops, name, None))
             for name in methods}
    setattr(ops, originals, saved)
    for name, function in methods.items():
        setattr(ops, name, staticmethod(function))
    setattr(ops, marker, True)
    return True
