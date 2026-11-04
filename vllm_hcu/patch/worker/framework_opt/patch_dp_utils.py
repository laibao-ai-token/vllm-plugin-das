# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Avoid unnecessary DP coordination for DeepEP low-latency mode."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import already_applied, load_exact_module, require_callable, require_exact_signature

TARGET_MODULE = "vllm.v1.worker.dp_utils"
PATCH_ID = "worker.framework_opt.dp.deepep_low_latency"
TARGETS = (f"{TARGET_MODULE}.coordinate_batch_across_dp",)
_MARKER = "_vllm_hcu_dp_low_latency_applied"
_WRAPPER = "_vllm_hcu_dp_low_latency_wrapper"


def _max_low_latency_token_slot(num_tokens_unpadded, num_tokens_padded, parallel_config):
    import torch
    import torch.distributed as dist
    from vllm.distributed.parallel_state import get_dp_group
    group = get_dp_group()
    cpu_sync = getattr(parallel_config, "disable_nccl_for_dp_synchronization", False)
    device = "cpu" if cpu_sync else group.device
    process_group = group.cpu_group if cpu_sync else group.device_group
    local = num_tokens_padded if num_tokens_padded is not None else num_tokens_unpadded
    slot = torch.tensor([local], dtype=torch.int32, device=device)
    dist.all_reduce(slot, op=dist.ReduceOp.MAX, group=process_group)
    return int(slot.item())


def apply_to_module(module: ModuleType) -> bool:
    dp = load_exact_module(TARGET_MODULE, module)
    wrapped = ((dp, "coordinate_batch_across_dp", TARGETS[0], _WRAPPER),)
    if already_applied(dp, _MARKER, wrapped):
        return False
    original = require_callable(dp, "coordinate_batch_across_dp", TARGETS[0])
    require_exact_signature(
        original,
        TARGETS[0],
        positional=(
            "num_tokens_unpadded",
            "allow_microbatching",
            "parallel_config",
            "num_tokens_padded",
            "uniform_decode",
            "cudagraph_mode",
        ),
        defaults={
            "num_tokens_padded": None,
            "uniform_decode": None,
            "cudagraph_mode": 0,
        },
    )

    @functools.wraps(original)
    def hcu_coordinate(
        num_tokens_unpadded,
        allow_microbatching,
        parallel_config,
        num_tokens_padded=None,
        uniform_decode=None,
        cudagraph_mode=0,
    ):
        if (
            parallel_config.data_parallel_size == 1
            or (
                parallel_config.all2all_backend == "deepep_low_latency"
                and not getattr(
                    parallel_config, "_vllm_hcu_deepep_auto", False
                )
            )
        ):
            if parallel_config.data_parallel_size > 1:
                from vllm.config import get_current_vllm_config_or_none
                from vllm_hcu.dspark_runtime import native_decode_dspark
                config = get_current_vllm_config_or_none()
                if config is not None and native_decode_dspark(config):
                    import torch
                    slot = _max_low_latency_token_slot(
                        num_tokens_unpadded, num_tokens_padded, parallel_config)
                    return False, torch.full(
                        (parallel_config.data_parallel_size,), slot, dtype=torch.int32,
                    ), cudagraph_mode
            return False, None, cudagraph_mode
        return original(
            num_tokens_unpadded,
            allow_microbatching,
            parallel_config,
            num_tokens_padded,
            uniform_decode,
            cudagraph_mode,
        )

    setattr(hcu_coordinate, _WRAPPER, True)
    setattr(dp, "_vllm_hcu_original_coordinate_batch_across_dp", original)
    setattr(dp, "coordinate_batch_across_dp", hcu_coordinate)
    setattr(dp, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
