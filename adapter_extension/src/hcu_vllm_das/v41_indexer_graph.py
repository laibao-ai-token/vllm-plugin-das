# SPDX-License-Identifier: Apache-2.0
"""Graph integration contract for the owned V41 FP8 indexer reference.

The reference indexer reads host-side attention metadata while it scores the
FP8 cache.  It must therefore stay outside a capture segment.  v0.33 has two
supported ways to express that contract:

* breakable cudagraph capture uses ``eager_break_during_capture`` on the
  Python custom-op kernel; and
* the legacy Dynamo/Inductor piecewise path uses the operator's
  ``cudagraph_unsafe`` tag plus ``CompilationConfig.splitting_ops``.

This module owns the public op name and the split-list registration so the
indexer construction path and the configured compilation lifecycle stay
consistent.
"""
from __future__ import annotations

OP_NAME = 'vllm::hcu_v41_fp8_indexer_reference'


def register_splitting_op(config) -> bool:
    """Append the reference op to a finalized Inductor piecewise split list.

    Must be called with the resolved ``VllmConfig`` after
    ``CompilationConfig.set_splitting_ops_for_v1`` has finalized the list.  Do
    not use the early activation path: ``get_cached_compilation_config`` is not
    valid there on the v0.33 worker startup path.
    """
    compilation = getattr(config, 'compilation_config', None)
    if compilation is None:
        return False
    from vllm.config import CompilationMode

    if getattr(compilation, 'mode', None) != CompilationMode.VLLM_COMPILE:
        return False
    if getattr(compilation, 'use_inductor_graph_partition', False):
        # The cudagraph_unsafe op tag is authoritative in this mode.
        return False
    cudagraph_mode = getattr(compilation, 'cudagraph_mode', None)
    has_piecewise = getattr(cudagraph_mode, 'has_piecewise_cudagraphs', None)
    if not callable(has_piecewise) or not has_piecewise():
        return False
    splitting_ops = getattr(compilation, 'splitting_ops', None)
    if not isinstance(splitting_ops, list):
        raise RuntimeError(
            'CompilationConfig.splitting_ops must be finalized as a list '
            'before registering the V41 reference indexer')
    if OP_NAME not in splitting_ops:
        splitting_ops.append(OP_NAME)
    return True


def graph_capture_allowed(config) -> bool:
    """Return whether a non-eager reference indexer has a capture break.

    Full cudagraphs are intentionally rejected: they are monolithic capture
    contexts, so a host-metadata custom op cannot be broken out of the replay.
    Use PIECEWISE (or the breakable piecewise wrapper) for this reference.
    """
    from vllm.compilation.breakable_cudagraph import (
        is_breakable_cudagraph_enabled,
    )

    compilation = getattr(config, 'compilation_config', None)
    cudagraph_mode = getattr(compilation, 'cudagraph_mode', None)
    if (
        cudagraph_mode is not None
        and cudagraph_mode.has_full_cudagraphs()
    ):
        return False
    if is_breakable_cudagraph_enabled():
        return True
    if compilation is None:
        return False
    from vllm.config import CompilationMode

    return bool(
        compilation.mode == CompilationMode.VLLM_COMPILE
        and (
            getattr(compilation, 'use_inductor_graph_partition', False)
            or OP_NAME in (getattr(compilation, 'splitting_ops', None) or [])
        )
    )
