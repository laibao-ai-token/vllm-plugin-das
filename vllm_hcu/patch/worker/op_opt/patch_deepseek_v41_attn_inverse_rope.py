# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Let the V4.1 attention epilogue own inverse-RoPE (vLLM 0.32 core).

The 0.32 community core fuses inverse-RoPE into the decode reduce kernel and
expects _o_proj(inverse_rope=False).  Its prefill path never rotates the
rows it writes, so every prefill row at position > 0 reaches wo_a without
the inverse rotation.  Position 0 is an identity rotation, which is why
first-token probes look aligned while longer generations drift.

This adapter disables the fused decode epilogue only for the bf16 output path:
forward_mqa then runs its existing standalone rocm_inverse_rope_rows_
pass over the whole output buffer (decode and prefill rows).  The MXFP8 output
path keeps the fused epilogue because it quantizes the rotated rows in the same
pass.  The standalone kernel was validated bit-identical to the reference
inverse GPT-J RoPE formula used by the 0.29 branch.
"""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v41.amd.rocm"
TARGET_SYMBOL = f"{TARGET_MODULE}.rocm_sparse_attn_decode"
PATCH_ID = "worker.op_opt.deepseek_v41.attn_inverse_rope"
_WRAPPER = "_vllm_hcu_dsv41_attn_inverse_rope_wrapper"
_MARKER = "_vllm_hcu_dsv41_attn_inverse_rope_applied"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        existing = getattr(target, "rocm_sparse_attn_decode", None)
        if existing is not None and getattr(existing, _WRAPPER, False):
            return False
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGET_SYMBOL} was reloaded without its wrapper"
        )
    original = require_callable(target, "rocm_sparse_attn_decode", TARGET_SYMBOL)
    require_exact_signature(
        original,
        TARGET_SYMBOL,
        positional=(
            "q",
            "kv_cache",
            "swa_k_cache",
            "swa_only",
            "topk_indices",
            "topk_lens",
            "swa_indices",
            "swa_lens",
            "swa_ragged_indices",
            "swa_ragged_indptr",
            "topk_ragged_indices",
            "topk_ragged_indptr",
            "attn_sink",
            "scale",
            "head_dim",
            "nope_head_dim",
            "rope_head_dim",
            "output",
            "extra_cache_nan_free",
            "adaptive_splits",
            "inv_rope_positions",
            "inv_rope_cos_sin_cache",
            "output_mxfp8",
        ),
        defaults={
            "extra_cache_nan_free": False,
            "adaptive_splits": False,
            "inv_rope_positions": None,
            "inv_rope_cos_sin_cache": None,
            "output_mxfp8": None,
        },
    )
    signature = inspect.signature(original)

    @functools.wraps(original)
    def decode(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if (
            bound.arguments.get("output_mxfp8") is None
            and bound.arguments.get("inv_rope_positions") is not None
        ):
            bound.arguments["inv_rope_positions"] = None
            bound.arguments["inv_rope_cos_sin_cache"] = None
        return original(*bound.args, **bound.kwargs)

    setattr(decode, _WRAPPER, True)
    setattr(target, "rocm_sparse_attn_decode", decode)
    setattr(target, _MARKER, True)
    return True
