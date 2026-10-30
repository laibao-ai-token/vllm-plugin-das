# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Bind DSpark's actual loader to the explicit eager draft ownership policy."""
from __future__ import annotations

import functools
import sys
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.utils"
PATCH_ID = "worker.core_fix.dspark.draft_scope"
CALLER_MODULE = "vllm.v1.worker.gpu.spec_decode.dspark.speculator"
TARGETS = (f"{TARGET_MODULE}.load_dspark_model", f"{CALLER_MODULE}.load_dspark_model")
_MARKER = "_vllm_hcu_dspark_draft_scope_applied"
_WRAPPER = "_vllm_hcu_dspark_draft_scope_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    original = require_callable(target, "load_dspark_model", TARGETS[0])
    caller = sys.modules.get(CALLER_MODULE)
    alias = getattr(caller, "load_dspark_model", None)
    if getattr(target, _MARKER, False):
        if not getattr(original, _WRAPPER, False):
            raise PatchCompatibilityError(f"required HCU patch marker for {TARGETS[0]} is stale")
        previous = target._vllm_hcu_original_load_dspark_model
        if alias is not None and alias is not original and alias is not previous:
            raise PatchCompatibilityError(f"required HCU patch caller alias {TARGETS[1]} is incompatible")
        if alias is previous:
            caller.load_dspark_model = original
        return False
    if alias is not None and alias is not original:
        raise PatchCompatibilityError(f"required HCU patch caller alias {TARGETS[1]} is incompatible")
    require_exact_signature(original, TARGETS[0],
                            positional=("target_model", "vllm_config"))

    @functools.wraps(original)
    def hcu_load_dspark_model(target_model, vllm_config):
        from vllm_hcu.dspark_runtime import load_with_draft_scope
        return load_with_draft_scope(original, target_model, vllm_config)

    setattr(hcu_load_dspark_model, _WRAPPER, True)
    target._vllm_hcu_original_load_dspark_model = original
    target.load_dspark_model = hcu_load_dspark_model
    if alias is original:
        caller.load_dspark_model = hcu_load_dspark_model
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
