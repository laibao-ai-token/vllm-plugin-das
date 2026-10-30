# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Read-only PCP8 native-draft construction snapshots.

No vLLM/Torch import, group/manager creation, collective, mutation, scope or
admission hook exists here. References are borrowed; frozen records do not
make the referenced production objects immutable or prove runtime readiness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

MODE = "v41-pcp8-native-draft-construction"
_NATIVE_BACKEND = "deepep_high_throughput"
_FUTURE_BACKEND = "allgather_reducescatter"
_GROUP_BACKEND = "nccl"  # HCUPlatform.dist_backend; not the CPU PG backend.


class DescriptorNotReady(ValueError):
    """A required supplied object, field or PG reference is absent."""


class DescriptorMismatch(ValueError):
    """Supplied metadata disagrees with native ordered group topology."""


class UnsupportedConstruction(ValueError):
    """The supplied construction mode is outside this narrow snapshot policy."""


@dataclass(frozen=True)
class GroupReferences:
    world: object
    tp: object
    dp: object
    pcp: object
    ep: object
    pp: object
    dcp: object


@dataclass(frozen=True)
class GroupSnapshot:
    role: str
    members: tuple[int, ...]
    global_rank: int
    rank_in_group: int
    world_size: int
    torch_distributed_backend: str
    unique_name: str
    local_rank: int
    device_index: int
    source: object = field(repr=False, compare=False)
    cpu_group: object = field(repr=False, compare=False)
    device_group: object = field(repr=False, compare=False)
    device_communicator: object | None = field(repr=False, compare=False)
    device: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class PhysicalGroups:
    world: GroupSnapshot
    tp: GroupSnapshot
    dp: GroupSnapshot
    pcp: GroupSnapshot
    ep: GroupSnapshot
    pp: GroupSnapshot
    dcp: GroupSnapshot


@dataclass(frozen=True)
class LogicalParallel:
    tp: int
    dp: int
    pcp: int
    pp: int
    dcp: int
    world_size: int
    dp_rank: int
    all2all_backend: str
    source: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class ConditionalMoE:
    """Native no-size-override result, not an observed/constructed MoE."""
    tp: int
    dp: int
    pcp: int
    ep: int
    tp_rank: int
    dp_rank: int
    pcp_rank: int
    ep_rank: int
    sp: int


@dataclass(frozen=True)
class ConditionalCapacity:
    max_requests: int
    speculative_tokens: int
    sample_from_anchor: bool
    query_per_request: int
    query_per_rank: int
    verify_per_rank: int
    local_token_buffer: int
    pcp_gather_rows_bound: int
    # No workspace sufficiency, per-step request count or graph bound implied.


@dataclass(frozen=True)
class ConstructionDescriptor:
    mode: str
    target: LogicalParallel
    native_draft: LogicalParallel
    groups: PhysicalGroups
    external_dp_index: int
    conditional_moe: ConditionalMoE
    capacity: ConditionalCapacity
    native_draft_backend: str
    future_expected_backend: str
    future_agrs_transport: GroupSnapshot
    target_model_class: type = field(repr=False, compare=False)
    draft_model_class: type = field(repr=False, compare=False)
    target_model_config: object = field(repr=False, compare=False)
    draft_model_config: object = field(repr=False, compare=False)
    target_dtype: object = field(repr=False, compare=False)
    draft_dtype: object = field(repr=False, compare=False)
    scheduler_config: object = field(repr=False, compare=False)
    speculative_config: object = field(repr=False, compare=False)


def _required(owner: object, name: str, path: str, *, allow_none: bool = False):
    if owner is None:
        raise DescriptorNotReady(f"{path}: object is absent")
    try:
        value = getattr(owner, name)
    except AttributeError as exc:
        raise DescriptorNotReady(f"{path}.{name}: field is absent") from exc
    if value is None and not allow_none:
        raise DescriptorNotReady(f"{path}.{name}: value is absent")
    return value


def _integer(value, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DescriptorMismatch(f"{path}: expected int >= {minimum}, got {value!r}")
    return value


def _boolean(value, path: str) -> bool:
    if type(value) is not bool:
        raise DescriptorMismatch(f"{path}: expected a bool")
    return value


def _group(role: str, source: object) -> GroupSnapshot:
    path = f"groups.{role}"
    members = _required(source, "ranks", path)
    if not isinstance(members, (list, tuple)) or not members:
        raise DescriptorMismatch(f"{path}.ranks: ordered nonempty list/tuple required")
    members = tuple(_integer(rank, f"{path}.ranks") for rank in members)
    if len(set(members)) != len(members):
        raise DescriptorMismatch(f"{path}.ranks: duplicate global ranks")
    size = _integer(_required(source, "world_size", path), f"{path}.world_size", minimum=1)
    rank = _integer(_required(source, "rank", path), f"{path}.rank")
    index = _integer(_required(source, "rank_in_group", path), f"{path}.rank_in_group")
    if size != len(members) or index >= size or members[index] != rank:
        raise DescriptorMismatch(f"{path}: size/global-rank/in-group-rank mismatch")
    backend = _required(source, "torch_distributed_backend", path)
    if not isinstance(backend, str) or backend != _GROUP_BACKEND:
        raise UnsupportedConstruction(f"{path}.torch_distributed_backend: expected {_GROUP_BACKEND!r}")
    cpu = _required(source, "cpu_group", path)
    device_pg = _required(source, "device_group", path)
    communicator = _required(source, "device_communicator", path, allow_none=True)
    use_device = _boolean(_required(source, "use_device_communicator", path), f"{path}.use_device_communicator")
    # Native WORLD and size-one groups may legitimately have no communicator.
    if role in ("pcp", "ep") and (not use_device or communicator is None):
        raise DescriptorNotReady(f"{path}: multi-rank PCP/EP device communicator is absent")
    unique_name = _required(source, "unique_name", path)
    if not isinstance(unique_name, str) or not unique_name:
        raise DescriptorMismatch(f"{path}.unique_name: nonempty native name required")
    return GroupSnapshot(
        role, members, rank, index, size, backend, unique_name,
        _integer(_required(source, "local_rank", path), f"{path}.local_rank"),
        _integer(_required(source, "device_index", path), f"{path}.device_index"),
        source, cpu, device_pg, communicator, _required(source, "device", path),
    )


def _parallel(config: object, path: str, pcp: int, world: int) -> LogicalParallel:
    dimensions = (("tensor_parallel_size", 1), ("data_parallel_size", 1),
                  ("prefill_context_parallel_size", pcp), ("pipeline_parallel_size", 1),
                  ("decode_context_parallel_size", 1), ("world_size", world))
    values = {}
    for name, expected in dimensions:
        value = _integer(_required(config, name, path), f"{path}.{name}", minimum=1)
        if value != expected:
            raise UnsupportedConstruction(f"{path}.{name}: expected {expected}, got {value}")
        values[name] = value
    for name, expected in (("enable_expert_parallel", True), ("enable_elastic_ep", False),
                           ("enable_eplb", False), ("use_sequence_parallel_moe", False),
                           ("_vllm_hcu_deepep_auto", False)):
        value = _boolean(_required(config, name, path), f"{path}.{name}")
        if value is not expected:
            raise UnsupportedConstruction(f"{path}.{name}: unsupported policy")
    backend = _required(config, "all2all_backend", path)
    if backend != _NATIVE_BACKEND:
        raise UnsupportedConstruction(f"{path}.all2all_backend: native draft must still inherit {_NATIVE_BACKEND!r}")
    dp_rank = _integer(_required(config, "data_parallel_rank", path), f"{path}.data_parallel_rank")
    if dp_rank != 0:
        raise DescriptorMismatch(f"{path}.data_parallel_rank: DP singleton must have rank zero")
    return LogicalParallel(values["tensor_parallel_size"], values["data_parallel_size"],
                           values["prefill_context_parallel_size"], values["pipeline_parallel_size"],
                           values["decode_context_parallel_size"], values["world_size"],
                           dp_rank, backend, config)


def describe_pcp8_native_draft(
    *, mode: str, target_parallel_config: object, native_draft_parallel_config: object,
    target_model_config: object, draft_model_config: object,
    target_model_class: type, draft_model_class: type,
    scheduler_config: object, speculative_config: object,
    groups: GroupReferences, sample_from_anchor: bool,
    factory_kwargs: Mapping[str, object],
) -> ConstructionDescriptor:
    """Snapshot supplied references; never construct configs/models/PGs/managers.

    The output is not an admission token. Native backend remains HT; AGRS
    transport/gather fields are conditional future-construction information.
    Class references are recorded, not resolved/imported or executed here.
    """
    if mode != MODE:
        raise UnsupportedConstruction(f"unsupported construction mode: {mode!r}")
    if not isinstance(groups, GroupReferences):
        raise DescriptorNotReady("groups: explicit GroupReferences required")
    for cls, path in ((target_model_class, "target_model_class"), (draft_model_class, "draft_model_class")):
        if cls is None:
            raise DescriptorNotReady(f"{path}: class reference is absent")
        if not isinstance(cls, type):
            raise DescriptorMismatch(f"{path}: a class reference is required")
    for config, path in ((target_model_config, "target_model_config"), (draft_model_config, "draft_model_config")):
        if not _boolean(_required(config, "enforce_eager", path), f"{path}.enforce_eager"):
            raise UnsupportedConstruction(f"{path}: non-eager execution is outside A1")
    target = _parallel(target_parallel_config, "target_parallel_config", 8, 8)
    draft = _parallel(native_draft_parallel_config, "native_draft_parallel_config", 1, 1)
    if _required(speculative_config, "method", "speculative_config") != "dspark":
        raise UnsupportedConstruction("speculative_config.method: expected dspark")
    if _required(speculative_config, "target_parallel_config", "speculative_config") is not target_parallel_config:
        raise DescriptorMismatch("speculative_config.target_parallel_config: reference mismatch")
    if _required(speculative_config, "target_model_config", "speculative_config") is not target_model_config:
        raise DescriptorMismatch("speculative_config.target_model_config: reference mismatch")
    if _required(speculative_config, "draft_model_config", "speculative_config") is not draft_model_config:
        raise DescriptorMismatch("speculative_config.draft_model_config: reference mismatch")
    spec_draft_parallel = _required(speculative_config, "draft_parallel_config", "speculative_config")
    spec_draft_tp = _integer(_required(spec_draft_parallel, "tensor_parallel_size",
                                     "speculative_config.draft_parallel_config"),
                             "speculative_config.draft_parallel_config.tensor_parallel_size", minimum=1)
    if spec_draft_tp != draft.tp:
        raise DescriptorMismatch("native loader draft TP differs from supplied logical draft TP")
    if not isinstance(factory_kwargs, Mapping):
        raise DescriptorNotReady("factory_kwargs: actual call mapping required")
    if any(factory_kwargs.get(name) is not None for name in ("tp_size", "dp_size", "pcp_size")):
        raise UnsupportedConstruction("factory_kwargs: explicit size overrides are outside the standard native path")
    # This is the raw native call mapping. Standard V41 omits these kwargs;
    # the fixed native signature supplies False/0 (not physical-size fallbacks).
    if _boolean(factory_kwargs.get("is_sequence_parallel", False), "factory_kwargs.is_sequence_parallel"):
        raise UnsupportedConstruction("factory_kwargs: sequence-parallel mode is outside A1")
    if _boolean(factory_kwargs.get("enable_eplb", False), "factory_kwargs.enable_eplb"):
        raise UnsupportedConstruction("factory_kwargs: EPLB override is outside A1")
    redundant = _integer(factory_kwargs.get("num_redundant_experts", 0),
                         "factory_kwargs.num_redundant_experts")
    if redundant:
        raise UnsupportedConstruction("factory_kwargs: redundant experts are outside A1")

    physical = PhysicalGroups(*(_group(role, getattr(groups, role)) for role in ("world", "tp", "dp", "pcp", "ep", "pp", "dcp")))
    world = physical.world
    if world.members != tuple(range(world.world_size)) or world.world_size % 8:
        raise UnsupportedConstruction("WORLD: expected native nonelastic ranks 0..W-1 with W divisible by 8")
    rank = world.global_rank
    external = rank // 8
    expected_members = tuple(range(external * 8, external * 8 + 8))
    for role in ("tp", "dp", "pp", "dcp"):
        group = getattr(physical, role)
        if group.global_rank != rank or group.members != (rank,) or group.rank_in_group != 0:
            raise DescriptorMismatch(f"groups.{role}: expected this global rank's singleton")
    for role in ("pcp", "ep"):
        group = getattr(physical, role)
        if group.global_rank != rank or group.members != expected_members or group.rank_in_group != rank % 8:
            raise DescriptorMismatch(f"groups.{role}: ordered native PCP8/EP8 members or rank mismatch")
    if physical.pcp.members != physical.ep.members:
        raise DescriptorMismatch("PCP and EP ordered members differ")
    if target.dp_rank != physical.dp.rank_in_group or draft.dp_rank != physical.dp.rank_in_group:
        raise DescriptorMismatch("logical DP rank differs from physical DP rank")

    requests = _integer(_required(scheduler_config, "max_num_seqs", "scheduler_config"), "max_num_seqs", minimum=1)
    capacity = _integer(_required(scheduler_config, "max_num_batched_tokens", "scheduler_config"), "max_num_batched_tokens", minimum=1)
    n = _integer(_required(speculative_config, "num_speculative_tokens", "speculative_config"), "num_speculative_tokens", minimum=1)
    anchor = _boolean(sample_from_anchor, "sample_from_anchor")
    q = n if anchor else n + 1
    query, verify = requests * q, requests * (n + 1)
    if max(query, verify) > capacity:
        raise UnsupportedConstruction("per-rank query/verify full-slot bound exceeds unchanged local token buffer")
    conditional_capacity = ConditionalCapacity(requests, n, anchor, q, query, verify, capacity,
                                                physical.pcp.world_size * query)
    return ConstructionDescriptor(
        mode, target, draft, physical, external,
        ConditionalMoE(1, physical.dp.world_size, physical.pcp.world_size, physical.ep.world_size,
                       0, physical.dp.rank_in_group, physical.pcp.rank_in_group, physical.ep.rank_in_group, 1),
        conditional_capacity, draft.all2all_backend, _FUTURE_BACKEND, physical.pcp,
        target_model_class, draft_model_class, target_model_config, draft_model_config,
        _required(target_model_config, "dtype", "target_model_config"),
        _required(draft_model_config, "dtype", "draft_model_config"), scheduler_config, speculative_config,
    )
