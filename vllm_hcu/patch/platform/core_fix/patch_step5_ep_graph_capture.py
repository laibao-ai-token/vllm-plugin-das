# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations
from types import ModuleType
from vllm_hcu.runtime_compat.step5_patch_contract import (
    Step5PatchCompatibilityError,
    already_applied,
    remember_targets,
    require_signatures,
)
from ._common import load_exact_module, require_callable

TARGET_MODULE = "vllm.distributed.parallel_state"
PATCH_ID = "platform.core_fix.step5_ep_graph_capture"
TARGETS = ("vllm.distributed.parallel_state.graph_capture",)
_MARKER = "_vllm_hcu_step5_ep_graph_capture_applied"

SIGNATURES = (
    (
        "vllm.distributed.parallel_state.graph_capture",
        ("device", "graph_capture_context"),
    ),
)
REVIEWED_SOURCES = ()


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, TARGETS):
        return False
    try:
        require_signatures(target, SIGNATURES, REVIEWED_SOURCES)
    except Step5PatchCompatibilityError:
        # Step5-only optional adapter. A core whose module-level graph_capture
        # has a different signature (e.g. the fixed v0.32 body takes only
        # ``device`` and already wraps TP/PP/DP groups) simply does not need
        # this EP capture wrapper. Skip it instead of failing platform
        # initialization for unrelated models.
        return False
    import functools
    from contextlib import contextmanager, nullcontext

    original = require_callable(target, "graph_capture", TARGETS[0])

    @contextmanager
    @functools.wraps(original)
    def graph_capture(device, graph_capture_context=None):
        with original(device, graph_capture_context) as context:
            ca = getattr(
                getattr(getattr(target, "_EP", None), "device_communicator", None),
                "ca_comm",
                None,
            )
            with ca.capture() if ca is not None else nullcontext():
                yield context

    target.graph_capture = graph_capture
    remember_targets(target, _MARKER, TARGETS)
    return True


def apply(module=None):
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
