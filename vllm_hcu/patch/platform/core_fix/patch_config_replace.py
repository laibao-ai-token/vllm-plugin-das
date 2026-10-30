# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Preserve audited HCU ParallelConfig aliases across strict native replace."""
from __future__ import annotations

import copy
from dataclasses import fields
import functools
import sys
from types import ModuleType

from ._common import (PatchCompatibilityError, apply_once, load_exact_module,
                      require_callable, require_positional_signature)

TARGET_MODULE = "vllm.config.utils"
PATCH_ID = "platform.core_fix.parallel_config.hcu_replace_aliases"
_CALLERS = ("vllm.config", "vllm.v1.worker.gpu.spec_decode.dflash.speculator",
            "vllm.v1.worker.gpu.spec_decode.dspark.utils")
TARGETS = (f"{TARGET_MODULE}.replace", *(f"{name}.replace" for name in _CALLERS))
_MARKER = "_vllm_hcu_parallel_replace_applied"
_WRAPPER = "_vllm_hcu_parallel_replace_wrapper"
_PRIVATE_FIELDS = (
    "_vllm_hcu_deepep_auto",
    "_vllm_hcu_expert_map_path",
    "_vllm_hcu_expert_map_record_path",
    "_vllm_hcu_eplb_disable_rearrange",
    "_vllm_hcu_eplb_static_dispatch_policy",
)


def _extras(instance):
    parallel = sys.modules.get("vllm.config.parallel")
    cls = getattr(parallel, "ParallelConfig", None)
    if cls is None or type(instance) is not cls:
        return {}
    declared = {item.name for item in fields(cls)}
    return {name: instance.__dict__[name] for name in _PRIVATE_FIELDS
            if name not in declared and name in instance.__dict__}


def _caller_aliases(original, wrapper=None):
    result = []
    for name in _CALLERS:
        module = sys.modules.get(name)
        alias = getattr(module, "replace", None)
        if alias is None:
            continue
        if alias is not original and alias is not wrapper:
            raise PatchCompatibilityError(f"required HCU replace alias {name}.replace is incompatible")
        result.append(module)
    return result


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    current = require_callable(target, "replace", TARGETS[0])
    if getattr(target, _MARKER, False):
        if not getattr(current, _WRAPPER, False):
            raise PatchCompatibilityError("required HCU ParallelConfig replace marker is stale")
        original = target._vllm_hcu_original_parallel_replace
        for caller in _caller_aliases(original, current):
            caller.replace = current
        return False
    require_positional_signature(current, TARGETS[0], ("dataclass_instance",), var_keyword="kwargs")
    callers = _caller_aliases(current)

    def install():
        @functools.wraps(current)
        def hcu_replace(dataclass_instance, /, **kwargs):
            extras = _extras(dataclass_instance)
            if not extras:
                return current(dataclass_instance, **kwargs)
            # Keep the same real class/constructor and the exact native
            # get_field/is_init_field/kwargs behavior. Only the five audited
            # HCU attributes are hidden on a private shallow copy. Unknown
            # attributes and unknown update kwargs still fail natively.
            shadow = copy.copy(dataclass_instance)
            if shadow.__dict__ is dataclass_instance.__dict__:
                raise PatchCompatibilityError("ParallelConfig copy unexpectedly shares its attribute dictionary")
            for name in extras:
                del shadow.__dict__[name]
            updates = kwargs
            # Native ParallelConfig construction pops the port list and may
            # resolve eplb_config.communicator. Keep those source-owned mutable
            # values private as well; explicit non-alias replacements win.
            for name in ("_data_parallel_master_port_list", "eplb_config"):
                if name not in dataclass_instance.__dict__:
                    continue
                value = dataclass_instance.__dict__[name]
                if name not in kwargs:
                    shadow.__dict__[name] = copy.deepcopy(value)
                elif kwargs[name] is value:
                    if updates is kwargs:
                        updates = dict(kwargs)
                    updates[name] = copy.deepcopy(value)
            result = current(shadow, **updates)
            for name, value in extras.items():
                setattr(result, name, value)
            return result

        setattr(hcu_replace, _WRAPPER, True)
        target._vllm_hcu_original_parallel_replace = current
        target.replace = hcu_replace
        for caller in callers:
            caller.replace = hcu_replace
        setattr(target, _MARKER, True)

    install()
    return True


def apply(module: ModuleType | None = None) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    return apply_once(patch_id=PATCH_ID, targets=TARGETS, marker_owner=target,
                      marker=_MARKER, callback=lambda: apply_to_module(target))
