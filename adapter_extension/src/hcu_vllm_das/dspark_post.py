# SPDX-License-Identifier: Apache-2.0
"""Admit the verified draft Post by default; retain an explicit opt-out."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import inspect
import logging
from types import FunctionType

KEY = "hcu_dsv41_dspark_final_post"


def requested(config):
    extra = getattr(config, "additional_config", None)
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise ValueError("DSpark additional_config must be a dict")
    explicit = KEY in extra
    value = extra.get(KEY, True)
    if value is False:
        return False
    if value is not True:
        raise ValueError("W4A8 draft final Post must be an explicit boolean")
    if not explicit:
        # Default admission must not resolve private native bodies for other
        # models/topologies. An explicit request retains the strict error path.
        if getattr(config, "quant_config", None) is None:
            return False
        try:
            return _validate(config)
        except (AttributeError, ValueError):
            return False
    return _validate(config)


def _validate(config):
    from .dspark_agrs import _owned_quant_kind, is_owned_draft, draft_config
    from .dspark_graph import native_decode_dspark
    owned = is_owned_draft(config)
    if _owned_quant_kind(config) != "ct_w4a8" or not (owned or native_decode_dspark(config)):
        raise ValueError("Draft final Post requires the native W4A8 TP1/DP8/EP8 target")
    if owned:
        draft_config(config)  # Recheck the graph loader's fixed-capacity lease.
    mc = config.model_config
    hf = getattr(mc, "hf_text_config", None)
    if hf is None:
        hf = getattr(mc.hf_config, "text_config", mc.hf_config)
    if (getattr(hf, "hidden_size", None), getattr(hf, "hc_mult", None),
            config.speculative_config.num_speculative_tokens) != (5120, 4, 5):
        raise ValueError("Draft final Post requires the verified H5120/HC4/DSpark5 contract")
    return True


def _clone(owner, name, expected, substitutions):
    original = getattr(owner, name)
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != expected:
        raise RuntimeError("Owned DSpark native source changed: " + name)
    function = inspect.unwrap(original)
    if any(key not in function.__globals__ for key in substitutions):
        raise RuntimeError("Owned DSpark native namespace anchor missing: " + name)
    namespace = dict(function.__globals__)
    namespace.update(substitutions)
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__
    clone.__qualname__ = function.__qualname__
    clone.__module__ = function.__module__
    clone.__doc__ = function.__doc__
    return clone, (owner, name, original)


@lru_cache(maxsize=1)
def implementation():
    from hcu_adapter.kernels import mhc_post_5120_triton as post
    from hcu_adapter.kill_switch import op_disabled
    from vllm.models.deepseek_v41.amd import dspark as native

    original_post = native.mhc_post_torch
    if hashlib.sha256(inspect.getsource(original_post).encode()).hexdigest() != (
            "24f96f096ebd6c6ea2a5b902402f90321c8d4489cb0ab6a351823d6eb86c9b7a"):
        raise RuntimeError("Owned DSpark native source changed: mhc_post_torch")

    def final_post(x, residual, post_mix, comb):
        if not op_disabled('vllm_mhc') and post.supported(x, residual, post_mix, comb):
            return post.post_5120(x, residual, post_mix, comb)
        return original_post(x, residual, post_mix, comb)

    forward, forward_anchor = _clone(native.DSparkDeepseekV4Model, "forward",
        "4fae143677e054b267a5861db30277df79b51608cca8c5b4540220e63746fd01",
        {"mhc_post_torch": final_post})

    class DraftModel(native.DSparkDeepseekV4Model):
        pass

    DraftModel.forward = forward
    outer, outer_anchor = _clone(native.DSparkDeepseekV4ForCausalLM, "__init__",
        "8f63625682361086e3df60598866076e0aefb657a200c03fbc7725b07ddb299f",
        {"DSparkDeepseekV4Model": DraftModel})
    return DraftModel, outer, (forward_anchor, outer_anchor,
                               (native, "mhc_post_torch", original_post))


def initialize_owned(model, *, vllm_config, prefix):
    _, constructor, anchors = implementation()
    if any(getattr(owner, name) is not original for owner, name, original in anchors):
        raise RuntimeError("Owned DSpark native implementation identity changed")
    constructor(model, vllm_config=vllm_config, prefix=prefix)
    logging.getLogger("vllm." + __name__).info(
        "HCU W4A8 private draft final Post constructor admitted: H=5120 HC=4")
