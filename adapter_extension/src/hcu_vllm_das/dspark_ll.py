# SPDX-License-Identifier: Apache-2.0
"""Single native target buffer for the fixed-capacity W4A8 LL experiment.

Keep the native cache entry and destructor. The caller emits a native clean
before every target/draft dispatch so layout changes also work on graph replay.
This transport policy is limited to single-node legacy IPC.
"""
from __future__ import annotations

import os


class SharedIPCManager:
    """Native factory argument; never replaces the EP group's manager."""

    def __init__(self, manager):
        import deep_ep
        from vllm.distributed.device_communicators.all2all import DeepEPLLAll2AllManager

        if (not isinstance(manager, DeepEPLLAll2AllManager)
                or manager.internode or manager.world_size != 8
                or manager.dp_world_size != 8 or manager.support_fault_tolerance):
            raise ValueError("W4A8 shared LL requires single-node native DP8/EP8")
        if os.environ.get("ROCSHMEM_BACKEND", "ipc") != "ipc":
            raise ValueError("W4A8 shared LL requires ROCSHMEM IPC")
        for name in ("is_ll_layered_opt_enabled", "is_ll_layered_develop_enabled"):
            select = getattr(deep_ep.Buffer, name, None)
            if not callable(select) or select(8):
                raise ValueError("W4A8 shared LL requires native legacy IPC selection")
        for name in ("ROCSHMEM_IPC_MNVL", "DEEP_EP_NORMAL_MNVL"):
            if os.environ.get(name, "0").strip().lower() not in ("0", "false", "off", ""):
                raise ValueError("W4A8 shared LL does not support " + name)
        if os.environ.get("DEEPEP_TEST_FORCE_ROUTE_MODE", "").strip():
            raise ValueError("W4A8 shared LL does not support forced routes")
        self.native = manager
        self.world_size = manager.world_size
        self.dp_world_size = manager.dp_world_size

    def get_handle(self, kwargs):
        expected = dict(max_num_tokens_per_dp_rank=256, token_hidden_size=5120,
                        num_ep_ranks=8, num_global_experts=384, num_local_experts=48)
        experts = kwargs.get("num_global_experts")
        actual = dict(expected, num_global_experts=experts,
                      num_local_experts=experts // 8 if type(experts) is int else None)
        if experts not in (128, 384) or kwargs != actual:
            raise ValueError("W4A8 shared LL geometry is outside the fixed contract")
        # Request the same native cache key for both layouts. Adding a second
        # cache alias would give the native manager two destruction owners.
        return self.native.get_handle(expected)
