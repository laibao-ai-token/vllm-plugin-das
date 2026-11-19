# SPDX-License-Identifier: Apache-2.0
"""Explicit W4A8 draft LL experiment; AG/RS remains the default."""

KEY = "hcu_dsv41_dspark_draft_backend"


def native_draft_ll_requested(config):
    extra = getattr(config, "additional_config", None)
    if extra is None:
        return False
    if not isinstance(extra, dict):
        raise ValueError("DSpark additional_config must be a dict")
    backend = extra.get(KEY, "allgather_reducescatter")
    if backend == "allgather_reducescatter":
        return False
    if backend != "deepep_low_latency":
        raise ValueError("Unsupported HCU DSpark draft backend: " + str(backend))

    from .dspark_agrs import _owned_quant_kind, draft_config, is_owned_draft
    from .dspark_graph import native_decode_dspark

    if (_owned_quant_kind(config) != "ct_w4a8"
            or not native_decode_dspark(config) or is_owned_draft(config)):
        raise ValueError("W4A8 draft LL requires the native CT DP8/EP8 target")
    scheduler, spec = config.scheduler_config, config.speculative_config
    if (scheduler.max_num_batched_tokens, scheduler.max_num_seqs,
            spec.num_speculative_tokens) != (256, 32, 5):
        raise ValueError("W4A8 draft LL requires the fixed 256/32/5 token contract")
    # Reuse all default feature, topology and graph capacity checks without
    # constructing any model, communicator or transport resource.
    draft_config(config)
    return True
