# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Select the draft config before native DFlash builds its forward context."""
from __future__ import annotations

import functools
from types import ModuleType

from ._common import (PatchCompatibilityError, load_exact_module, require_callable,
                      require_class, require_exact_signature)

TARGET_MODULE = "vllm.v1.worker.gpu.spec_decode.dflash.speculator"
PATCH_ID = "worker.core_fix.dspark.draft_forward_context"
TARGETS = (f"{TARGET_MODULE}.DFlashSpeculator._run_model",)
_MARKER = "_vllm_hcu_dspark_draft_context_applied"
_WRAPPER = "_vllm_hcu_dspark_draft_context_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    cls = require_class(target, "DFlashSpeculator", f"{TARGET_MODULE}.DFlashSpeculator")
    original = require_callable(cls, "_run_model", TARGETS[0])
    if getattr(cls, _MARKER, False):
        if not getattr(original, _WRAPPER, False):
            raise PatchCompatibilityError(f"required HCU patch marker for {TARGETS[0]} is stale")
        return False
    mode_none = target.CUDAGraphMode.NONE
    require_exact_signature(
        original, TARGETS[0],
        positional=("self", "num_tokens", "attn_metadata", "slot_mappings",
                    "num_tokens_across_dp", "cudagraph_runtime_mode"),
        defaults={"cudagraph_runtime_mode": mode_none},
    )

    @functools.wraps(original)
    def hcu_run_model(self, num_tokens, attn_metadata, slot_mappings,
                      num_tokens_across_dp, cudagraph_runtime_mode=mode_none):
        scope = getattr(getattr(self, "model", None), "_hcu_dspark_draft_scope", None)
        if scope is None:
            return original(self, num_tokens, attn_metadata, slot_mappings,
                            num_tokens_across_dp, cudagraph_runtime_mode)
        if getattr(self, "_speculator_name", None) != "DSpark":
            raise ValueError("DSpark draft scope is attached to a non-DSpark speculator")
        if cudagraph_runtime_mode != mode_none:
            raise ValueError("DSpark draft context scope currently supports eager execution only")
        with scope.activate():
            previous = self.vllm_config
            self.vllm_config = scope.config
            try:
                # Native set_forward_context now sees AGRS and constructs DP
                # metadata before the model can reach its first dispatch.
                return original(self, num_tokens, attn_metadata, slot_mappings,
                                num_tokens_across_dp, cudagraph_runtime_mode)
            finally:
                self.vllm_config = previous

    setattr(hcu_run_model, _WRAPPER, True)
    cls._vllm_hcu_original_dspark_run_model = original
    cls._run_model = hcu_run_model
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
