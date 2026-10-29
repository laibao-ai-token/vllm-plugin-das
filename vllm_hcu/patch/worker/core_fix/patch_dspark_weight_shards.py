# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Restrict DSV4.1 DSpark tensor iteration to the indexed draft shards.

The paired loader treats allow_patterns_overrides as format alternatives (it
stops after the first match), and its safetensors index check requires the full
checkpoint file list. Keep that preparation and validation intact, then narrow
its result before any shard's tensors are read. The context belongs to a single
load_weights call; neither the target's config nor the loader instance is changed.
"""

from __future__ import annotations

import contextvars
import functools
import json
import os
from pathlib import Path
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.model_executor.model_loader.default_loader"
PATCH_ID = "worker.core_fix.deepseek_v41_dspark.weight_shards"
TARGETS = (
    f"{TARGET_MODULE}.DefaultModelLoader.load_weights",
    f"{TARGET_MODULE}.DefaultModelLoader._prepare_weights",
)
_DRAFT_MODULE = "vllm.models.deepseek_v41.amd.dspark"
_CLASS_MARKER = "_vllm_hcu_dspark_weight_shards_applied"
_LOAD_MARKER = "_vllm_hcu_dspark_weight_shards_load_wrapper"
_PREPARE_MARKER = "_vllm_hcu_dspark_weight_shards_prepare_wrapper"
_DRAFT_LOAD = contextvars.ContextVar("vllm_hcu_dspark_weight_shards", default=None)


def _is_supported_draft(model: object, model_config: object) -> bool:
    model_type = type(model)
    return (
        model_type.__module__ == _DRAFT_MODULE
        and model_type.__name__ == "DSparkDeepseekV4ForCausalLM"
        and getattr(getattr(model_config, "hf_config", None), "model_type", None)
        == "deepseek_v41"
        and callable(getattr(model, "_remap_dspark_name", None))
        and getattr(model, "allow_patterns_overrides", None) is None
        and not getattr(model, "secondary_weights", ())
    )


def _select_draft_files(
    model: object, hf_folder: str, weight_files: list[str]
) -> list[str] | None:
    """Use the draft's own acceptance rules, including its optional head aliases."""
    try:
        with (Path(hf_folder) / "model.safetensors.index.json").open(
            encoding="utf-8"
        ) as stream:
            index = json.load(stream)
    except (OSError, ValueError, UnicodeError):
        return None
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return None
    if any(
        not isinstance(name, str) or not isinstance(filename, str)
        for name, filename in weight_map.items()
    ):
        return None

    # Ask the official remapper rather than duplicating stage/head/scale rules.
    # embed/head are rejected by this loader and shared from the target later.
    required = {
        filename
        for name, filename in weight_map.items()
        if model._remap_dspark_name(name) is not None
    }
    if not required or any(
        Path(filename).is_absolute()
        or ".." in Path(filename).parts
        or not filename.endswith(".safetensors")
        for filename in required
    ):
        return None
    required = {os.path.normpath(filename) for filename in required}
    candidates = {
        os.path.normpath(os.path.relpath(filename, hf_folder))
        for filename in weight_files
    }
    if not required.issubset(candidates):
        return None
    return [
        filename
        for filename in weight_files
        if os.path.normpath(os.path.relpath(filename, hf_folder)) in required
    ]


def apply_to_module(module: ModuleType) -> bool:
    loader_module = load_exact_module(TARGET_MODULE, module)
    loader_class = require_class(
        loader_module, "DefaultModelLoader", f"{TARGET_MODULE}.DefaultModelLoader"
    )
    original_load = require_callable(loader_class, "load_weights", TARGETS[0])
    original_prepare = require_callable(loader_class, "_prepare_weights", TARGETS[1])
    if getattr(loader_class, _CLASS_MARKER, False):
        if not getattr(original_load, _LOAD_MARKER, False) or not getattr(
            original_prepare, _PREPARE_MARKER, False
        ):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {PATCH_ID} is stale"
            )
        return False
    require_exact_signature(
        original_load, TARGETS[0], positional=("self", "model", "model_config")
    )
    require_exact_signature(
        original_prepare,
        TARGETS[1],
        positional=(
            "self",
            "model_name_or_path",
            "subfolder",
            "revision",
            "fall_back_to_pt",
            "allow_patterns_overrides",
        ),
    )

    @functools.wraps(original_load)
    def hcu_load_weights(self, model, model_config):
        # Explicitly clear an enclosing draft context for nested target loads.
        context = (
            (model, model_config)
            if _is_supported_draft(model, model_config)
            else None
        )
        token = _DRAFT_LOAD.set(context)
        try:
            return original_load(self, model, model_config)
        finally:
            _DRAFT_LOAD.reset(token)

    @functools.wraps(original_prepare)
    def hcu_prepare_weights(
        self,
        model_name_or_path,
        subfolder,
        revision,
        fall_back_to_pt,
        allow_patterns_overrides,
    ):
        prepared = original_prepare(
            self,
            model_name_or_path,
            subfolder,
            revision,
            fall_back_to_pt,
            allow_patterns_overrides,
        )
        context = _DRAFT_LOAD.get()
        if context is None:
            return prepared
        model, model_config = context
        hf_folder, weight_files, use_safetensors = prepared
        if (
            not use_safetensors
            or subfolder is not None
            or model_name_or_path != model_config.model
            or self.load_config.load_format
            not in ("auto", "hf", "safetensors", "fastsafetensors", "instanttensor")
        ):
            return prepared
        selected = _select_draft_files(model, hf_folder, weight_files)
        if selected is None:
            return prepared
        return hf_folder, selected, use_safetensors

    setattr(hcu_load_weights, _LOAD_MARKER, True)
    setattr(hcu_prepare_weights, _PREPARE_MARKER, True)
    setattr(loader_class, "_vllm_hcu_original_dspark_shard_load", original_load)
    setattr(loader_class, "_vllm_hcu_original_dspark_shard_prepare", original_prepare)
    setattr(loader_class, "load_weights", hcu_load_weights)
    setattr(loader_class, "_prepare_weights", hcu_prepare_weights)
    setattr(loader_class, _CLASS_MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
