# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Use a correct Engram hash launch geometry on HCU.

Re-hosted from the 0.29 line (``patch_deepseek_v41_engram_hash.py``, lost by the
v0.32.0 migration checkpoint).  On gfx938 with Triton 3.6 the upstream launch of
``_hash_ids_kernel`` at ``BLOCK_T=32`` and ``num_warps=4`` silently drops 2-gram
stores, so the n-gram hash table the Engram lookup reads is incomplete and the
injected embeddings are wrong without raising anything.  The fix is purely a
launch-geometry override: keep the upstream kernel, force ``num_warps=2``.

v0.32 anchor status (checked against ``vllm-community-main`` @ ``5df63a72``):
``vllm/models/deepseek_v41/common/engram.py:288`` still defines
``_hash_ids_kernel`` with ``output`` / ``num_heads`` / ``MAX_NGRAM`` / ``BLOCK_T`` /
``BLOCK_H`` parameters, and the only call site (``:562``) still launches it as
``_hash_ids_kernel[(cdiv(num_tokens, 32), num_layers)](...)`` with ``BLOCK_T=32``
and ``num_warps=4`` (``:600``).  So the defect and the remedy both carry over.

Mismatch behaviour: a signature/fingerprint mismatch means the upstream kernel no
longer looks like the one this geometry fix was written against.  The adapter
then installs nothing and says so in the log instead of blocking startup, because
this patch rewrites no state in that case.  It does not catch missing modules,
missing markers, or unreadable source; those still propagate.
"""

from __future__ import annotations

import inspect
from types import ModuleType

from vllm.logger import init_logger

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
)

TARGET_MODULE = "vllm.models.deepseek_v41.common.engram"
PATCH_ID = "worker.core_fix.deepseek_v41.engram_hash_launch"
TARGETS = (f"{TARGET_MODULE}._hash_ids_kernel",)
_MARKER = "_vllm_hcu_dsv41_hash_launch_applied"
_ORIGINAL = "_vllm_hcu_original_hash_ids_kernel"
# The launch geometry this override depends on; kept explicit so a renamed or
# restructured kernel fails the check instead of silently not applying.
_REQUIRED_PARAMS = frozenset({"output", "num_heads", "MAX_NGRAM", "BLOCK_T", "BLOCK_H"})
logger = init_logger(__name__)


class _HashKernelLaunch:
    """Transparent wrapper that only overrides the warp count of a launch."""

    def __init__(self, kernel):
        self.kernel = kernel

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def hcu_launch(*args, **kwargs):
            # gfx938 / Triton 3.6 drops 2-gram stores with BLOCK_T=32, four warps.
            kwargs["num_warps"] = 2
            return launch(*args, **kwargs)

        return hcu_launch

    def __getattr__(self, name):
        return getattr(self.kernel, name)


def _contract(kernel) -> bool:
    """True when this is still the kernel the geometry fix was written against."""
    try:
        require_callable(kernel, "__getitem__", TARGETS[0])
        function = require_callable(kernel, "fn", TARGETS[0] + ".fn")
    except PatchCompatibilityError as exc:
        logger.warning(
            "%s is not applicable to this vLLM build (%s); leaving the upstream "
            "Engram hash launch untouched. Engram n-gram hashes may be computed "
            "with the known-bad BLOCK_T=32 / num_warps=4 geometry.",
            PATCH_ID,
            exc,
        )
        return False
    if not _REQUIRED_PARAMS.issubset(inspect.signature(function).parameters):
        logger.warning(
            "%s is not applicable: %s no longer exposes %s; leaving the upstream "
            "Engram hash launch untouched.",
            PATCH_ID,
            TARGETS[0],
            sorted(_REQUIRED_PARAMS),
        )
        return False
    return True


def apply_to_module(module: ModuleType) -> bool:
    engram = load_exact_module(TARGET_MODULE, module)
    kernel = getattr(engram, "_hash_ids_kernel", None)
    if getattr(engram, _MARKER, False):
        if not isinstance(kernel, _HashKernelLaunch):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {PATCH_ID} is stale"
            )
        return False
    if not _contract(kernel):
        return False
    setattr(engram, _ORIGINAL, kernel)
    setattr(engram, "_hash_ids_kernel", _HashKernelLaunch(kernel))
    setattr(engram, _MARKER, True)
    logger.info("%s applied: %s launches with num_warps=2.", PATCH_ID, TARGETS[0])
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGETS", "TARGET_MODULE", "apply", "apply_to_module"]
