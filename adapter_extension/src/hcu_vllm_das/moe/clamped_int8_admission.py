# SPDX-License-Identifier: Apache-2.0
"""Admission for the byte-verified DSV4.1 W4A8 routed activation."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import logging
from pathlib import Path

import torch


@lru_cache(maxsize=2)
def _log_admission(masked):
    # vLLM's INFO handler lives on its non-propagating namespace. Retain
    # the Plugin identity below it so serving admission is observable.
    logging.getLogger("vllm." + __name__).info(
        "HCU W4A8 live-row INT8 activation admitted: layout=%s H5120 N2304 clamp10",
        "masked" if masked else "contiguous")


@lru_cache(maxsize=1)
def _verified_provider():
    try:
        module = importlib.import_module("lightop.fuse_silu_mul_quant")
        source = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        binary = hashlib.sha256(Path(module.op.__file__).read_bytes()).hexdigest()
    except (ImportError, OSError, AttributeError):
        return None
    if (source != "4aed1c44a5558077ca8d0a01454754883db69eb6b6806c5079e06ddc266ba800"
            or binary != "79138210e4e6fb9329c077f5de998861ef85f5c41d421dab3543f2f6bb3ea102"):
        return None
    return module


def admitted(ops, weights, w1_scale, w2_scale, limit, *, masked):
    """Unknown providers, shapes and devices retain their existing operator."""
    experts = int(w1_scale.shape[0])
    if (limit != 10 or experts not in ((16, 48) if masked else (16,))
            or tuple(w1_scale.shape) != (experts, 4608, 1)
            or tuple(w2_scale.shape) != (experts, 5120, 1)
            or weights[0].device.type != "cuda"):
        return False
    provider = _verified_provider()
    if (provider is None or ops.clamp_quant is not provider.fuse_silu_mul_clamp_quant
            or ops.clamp_quant_ep is not provider.fuse_silu_mul_clamp_quant_ep):
        return False
    props = torch.cuda.get_device_properties(weights[0].device)
    supported = (props.gcnArchName.split(":")[0] == "gfx938"
                 and props.multi_processor_count == 64)
    if supported:
        _log_admission(masked)
    return supported
