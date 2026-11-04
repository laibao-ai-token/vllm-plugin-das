# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Explicit, eager DSpark draft ownership for the PCP1/TP1/DP deployment.

The EP view borrows existing process groups; it owns only its Python
communicator view and AG-RS manager. It never destroys borrowed resources.
"""
from __future__ import annotations

import copy
import functools
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from threading import RLock, get_ident

from vllm_hcu.patch.config import get_hcu_config, set_hcu_config

_BACKEND = "allgather_reducescatter"
_ACTIVE_SCOPE = ContextVar("hcu_dspark_draft_scope", default=None)
_SCOPE_LOCK = RLock()


@dataclass(frozen=True)
class DSparkTokenCapacity:
    speculative_tokens: int
    max_requests: int
    query_tokens_per_request: int
    query_tokens_per_rank: int
    verify_tokens_per_rank: int
    capacity_per_rank: int


def token_capacity(vllm_config) -> DSparkTokenCapacity:
    spec = vllm_config.speculative_config
    scheduler = vllm_config.scheduler_config
    n = spec.num_speculative_tokens
    requests = scheduler.max_num_seqs
    capacity = scheduler.max_num_batched_tokens
    if any(type(value) is not int or value <= 0 for value in (n, requests, capacity)):
        raise ValueError("DSpark requires positive speculative tokens, request slots and capacity")
    anchor = getattr(spec.draft_model_config.hf_config, "sample_from_anchor", True)
    if type(anchor) is not bool:
        raise ValueError("DSpark sample_from_anchor must be bool")
    query = n if anchor else n + 1
    result = DSparkTokenCapacity(n, requests, query, requests * query,
                                requests * (n + 1), capacity)
    # Conservative full-slot contract; do not silently enlarge the scheduler,
    # native draft buffers, or the target LL handle/PF allocation.
    if max(result.query_tokens_per_rank, result.verify_tokens_per_rank) > capacity:
        raise ValueError(
            "DSpark token capacity is insufficient: "
            f"draft={result.query_tokens_per_rank}, verify={result.verify_tokens_per_rank}, "
            f"capacity={capacity} per rank"
        )
    return result


def native_decode_dspark(config):
    """Original acc_d scope; no opt-in or claim of prior GPU validation."""
    spec = getattr(config, "speculative_config", None)
    parallel = getattr(spec, "target_parallel_config", None)
    if parallel is None:
        parallel = config.parallel_config
    hf = getattr(config.model_config, "hf_config", None)
    return (
        getattr(spec, "method", None) == "dspark"
        and getattr(hf, "model_type", None) in ("deepseek_v4", "deepseek_v41")
        and (parallel.tensor_parallel_size, parallel.prefill_context_parallel_size,
             parallel.pipeline_parallel_size, parallel.data_parallel_size) == (1, 1, 1, 8)
        and parallel.enable_expert_parallel
        and parallel.all2all_backend == "deepep_low_latency"
    )


def validate_graph_capacity(config, moe=None):
    """Validate resolved token buckets; do not replace fixed LL capacity by M."""
    if config.model_config.enforce_eager:
        return
    compilation = getattr(config, "compilation_config", None)
    sizes = getattr(compilation, "cudagraph_capture_sizes", None)
    mode = getattr(getattr(compilation, "cudagraph_mode", None), "name", None)
    if (mode not in ("FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE", "PIECEWISE")
            or not sizes or any(type(n) is not int or n <= 0 for n in sizes)):
        raise ValueError("CT decode graph capture requires known mode and positive token buckets")
    budget = config.scheduler_config.max_num_batched_tokens
    if max(sizes) > budget:
        raise ValueError("CT decode graph capture exceeds scheduler capacity")
    if moe is not None:
        declared = getattr(moe, "max_capture_size", None)
        if type(declared) is not int or max(sizes) > declared or declared > budget:
            raise ValueError("CT decode graph capture does not match MoE producer capacity")


def bind_draft_prepare_finalize(prepare, manager):
    """Bind all current naive P/F EP dependencies per instance, before capture.

    The 0.29 binding covered prepare/finalize. v0.32 additionally resolves EP
    in _allocate_combine_input; leave global EP and target LL ownership intact.
    """
    from types import FunctionType, MethodType
    previous = getattr(prepare, "_hcu_dspark_bound_manager", None)
    if previous is not None:
        if previous is not manager:
            raise RuntimeError("DSpark prepare/finalize manager changed")
        return prepare
    for name in ("prepare", "finalize", "_allocate_combine_input"):
        original = getattr(prepare, name).__func__
        function = FunctionType(original.__code__,
            dict(original.__globals__, get_ep_group=lambda: manager),
            original.__name__, original.__defaults__, original.__closure__)
        function.__kwdefaults__ = original.__kwdefaults__
        functools.update_wrapper(function, original)
        setattr(prepare, name, MethodType(function, prepare))
    prepare._hcu_dspark_bound_manager = manager
    return prepare


def draft_config(vllm_config):
    spec = vllm_config.speculative_config
    target = spec.target_parallel_config
    parallel = vllm_config.parallel_config
    if (
        target.prefill_context_parallel_size != 1
        or target.tensor_parallel_size != 1
        or target.pipeline_parallel_size != 1
        or target.data_parallel_size <= 1
        or not target.enable_expert_parallel
        or getattr(target, "enable_elastic_ep", False)
    ):
        raise ValueError("DSpark AG-RS scope currently requires target PCP1/TP1/PP1 with DP+EP")
    if parallel.data_parallel_size != target.data_parallel_size:
        raise ValueError("DSpark runtime DP size must match target DP size")
    if not vllm_config.model_config.enforce_eager or not spec.draft_model_config.enforce_eager:
        if not native_decode_dspark(vllm_config):
            raise ValueError("DSpark AG-RS scope currently requires eager target and draft execution outside native DP8")
        validate_graph_capacity(vllm_config)
        token_capacity(vllm_config)
    result = copy.copy(vllm_config)
    result.parallel_config = copy.deepcopy(parallel)
    result.parallel_config.all2all_backend = _BACKEND
    result.parallel_config.prefill_context_parallel_size = 1
    result.parallel_config.enable_eplb = False
    result.parallel_config.enable_elastic_ep = False
    result.kernel_config = copy.deepcopy(vllm_config.kernel_config)
    requested = getattr(spec, "moe_backend", None)
    result.kernel_config.moe_backend = requested or result.kernel_config.moe_backend
    if result.kernel_config.moe_backend != "deep_gemm":
        raise ValueError("DSpark AG-RS CT W4A8 requires draft moe_backend='deep_gemm'")
    result.additional_config = copy.deepcopy(vllm_config.additional_config)
    # Native speculator discovers draft layers by diffing these live tables
    # on its original config. Copy pure compiler policy without copying models
    # or disconnecting the draft attention registration from that discovery.
    compilation = vllm_config.compilation_config
    live_tables = (getattr(compilation, "static_forward_context", None),
                   getattr(compilation, "static_all_moe_layers", None))
    result.compilation_config = copy.deepcopy(
        compilation, memo={id(table): table for table in live_tables if table is not None}
    )
    feature = replace(get_hcu_config(vllm_config), deepep_auto=False,
                      moe_backend="deep_gemm", expert_map_path=None,
                      expert_map_record_path=None, eplb_disable_rearrange=False)
    set_hcu_config(result, feature)
    return result


class _BorrowedCommunicatorView:
    def __init__(self, target, manager):
        self._target = target
        self.all2all_manager = manager
        self.all2all_backend = _BACKEND

    def __getattr__(self, name):
        return getattr(self._target, name)

    def dispatch_router_logits(self, *args, **kwargs):
        return type(self._target).dispatch_router_logits(self, *args, **kwargs)

    def dispatch(self, *args, **kwargs):
        return type(self._target).dispatch(self, *args, **kwargs)

    def combine(self, *args, **kwargs):
        return type(self._target).combine(self, *args, **kwargs)

    def allocate_combine_input(self, *args, **kwargs):
        return type(self._target).allocate_combine_input(self, *args, **kwargs)

    def combine_into_output(self, *args, **kwargs):
        return type(self._target).combine_into_output(self, *args, **kwargs)

    def _borrowed_resource(self, *args, **kwargs):
        raise RuntimeError("DSpark communicator view does not own borrowed resources")

    destroy = checkpoint_prepare = checkpoint_restore = suspend = resume = _borrowed_resource


class _BorrowedEpView:
    """Keep dispatch/combine ownership separate while borrowing rank/PG state."""
    def __init__(self, target, manager):
        self._target = target
        self.device_communicator = _BorrowedCommunicatorView(target.device_communicator, manager)

    def __getattr__(self, name):
        return getattr(self._target, name)

    def dispatch_router_logits(self, *args, **kwargs):
        return type(self._target).dispatch_router_logits(self, *args, **kwargs)

    def dispatch(self, *args, **kwargs):
        return type(self._target).dispatch(self, *args, **kwargs)

    def combine(self, *args, **kwargs):
        return type(self._target).combine(self, *args, **kwargs)

    def allocate_combine_input(self, *args, **kwargs):
        return type(self._target).allocate_combine_input(self, *args, **kwargs)

    def combine_into_output(self, *args, **kwargs):
        return type(self._target).combine_into_output(self, *args, **kwargs)

    def destroy(self):
        raise RuntimeError("DSpark EP view does not own its borrowed process groups")


class DSparkDraftScope:
    def __init__(self, target_config, config, target_ep, manager, parallel_state,
                 current_config, capacity):
        self.target_config = target_config
        self.config = config
        self.target_ep = target_ep
        self.manager = manager
        self.draft_ep = _BorrowedEpView(target_ep, manager)
        self.parallel_state = parallel_state
        self.current_config = current_config
        self.capacity = capacity
        self.owner_thread = get_ident()
        self.static_ep = False

    @contextmanager
    def activate(self):
        if get_ident() != self.owner_thread:
            raise RuntimeError("DSpark draft scope must run on its owning worker thread")
        # Eager worker model execution is serialized. The lock also makes
        # nested/reentrant use restore the exact previous EP view on failure.
        with _SCOPE_LOCK:
            previous = self.parallel_state._EP
            if previous is not self.target_ep and previous is not self.draft_ep:
                raise RuntimeError("DSpark EP owner changed outside the draft scope")
            token = _ACTIVE_SCOPE.set(self)
            from vllm_hcu.model_executor.layers.fused_moe.combine_runtime import set_active_ep_group
            combine_token = set_active_ep_group(self.draft_ep)
            if not self.static_ep:
                self.parallel_state._EP = self.draft_ep
            try:
                with self.current_config(self.config):
                    yield
            finally:
                from vllm_hcu.model_executor.layers.fused_moe.combine_runtime import reset_active_ep_group
                reset_active_ep_group(combine_token)
                self.parallel_state._EP = previous
                _ACTIVE_SCOPE.reset(token)


def get_dspark_token_capacity():
    """Read the active draft bound; consumers own any persistent binding."""
    scope = _ACTIVE_SCOPE.get()
    return None if scope is None else scope.capacity


def get_dspark_all2all_manager(moe):
    """Read a consumer-bound or active draft manager; never mutate the MoE."""
    bound = getattr(moe, "_hcu_dspark_all2all_manager", None)
    scope = _ACTIVE_SCOPE.get()
    if bound is not None:
        if scope is not None and bound is not scope.manager:
            raise RuntimeError("DSpark MoE manager belongs to a different draft scope")
        return bound
    if scope is None:
        return None
    return scope.manager


def load_with_draft_scope(original, target_model, vllm_config):
    feature = get_hcu_config(vllm_config)
    automatic = native_decode_dspark(vllm_config)
    if feature.dspark_draft_all2all_backend is None and not automatic:
        return original(target_model, vllm_config)
    spec = vllm_config.speculative_config
    if getattr(spec, "method", None) != "dspark":
        raise ValueError("DSpark draft backend requires speculative method='dspark'")
    model_type = getattr(spec.draft_model_config.hf_config, "model_type", None)
    if model_type not in ("deepseek_v4", "deepseek_v41"):
        raise ValueError("DSpark AG-RS scope supports DeepSeek V4/V4.1 only")
    config = draft_config(vllm_config)
    capacity = token_capacity(config)
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager

    with _SCOPE_LOCK:
        target_ep = parallel_state.get_ep_group()
        if target_ep.device_communicator is None:
            raise RuntimeError("DSpark target EP group has no device communicator")
        if target_ep.world_size != config.parallel_config.data_parallel_size:
            raise ValueError("DSpark TP1 EP group must match DP membership")
        with set_current_vllm_config(config):
            manager = AgRsAll2AllManager(target_ep.cpu_group)
        if manager.world_size != target_ep.world_size or manager.dp_world_size != config.parallel_config.data_parallel_size:
            raise ValueError("DSpark AG-RS manager does not match the draft DP/EP group")
    scope = DSparkDraftScope(vllm_config, config, target_ep, manager, parallel_state,
                            set_current_vllm_config, capacity)
    scope.static_ep = automatic
    if automatic:
        manager._hcu_dspark_static_ownership = True
        def padded_sizes(num_local_tokens, comm_group):
            if not 0 <= num_local_tokens <= capacity.capacity_per_rank:
                raise ValueError("DSpark AG-RS local token capacity exceeded")
            return [num_local_tokens] * comm_group.world_size
        manager._get_sizes = padded_sizes
    with scope.activate():
        model = original(target_model, config)
    forward = model.forward

    @functools.wraps(forward)
    def draft_forward(*args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is not None and ids.shape[0] > capacity.capacity_per_rank:
            raise ValueError("DSpark draft forward exceeds checked token capacity")
        with scope.activate():
            return forward(*args, **kwargs)

    model.forward = draft_forward
    model._hcu_dspark_draft_scope = scope
    return model
