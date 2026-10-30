# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the adjacent, unchanged final R3 runtime source.

Configuration/accessor imports are explicit boundaries. No vLLM core package is
imported, and no handoff-specific environment variable is required after copying
this file with its adjacent implementation tree into a repository.
"""
from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("pcp_world_size", "use_pcp", "pcp_rank")


def _module(monkeypatch, name, **attrs):
    parent = None
    for count in range(1, len(name.split(".")) + 1):
        key = ".".join(name.split(".")[:count])
        module = sys.modules.get(key)
        if module is None or not getattr(module, "_builder_scope_boundary", False):
            module = ModuleType(key)
            module.__path__ = []
            module._builder_scope_boundary = True
            monkeypatch.setitem(sys.modules, key, module)
        if parent is not None:
            monkeypatch.setattr(parent, key.rsplit(".", 1)[-1], module, raising=False)
        parent = module
    for key, value in attrs.items():
        monkeypatch.setattr(module, key, value, raising=False)
    return module


def _source(monkeypatch, name, relative):
    path = ROOT / relative
    assert path.is_file(), f"missing adjacent runtime source: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def harness(monkeypatch):
    env = NS(batch=None, events=[], depth={"logical": 0, "replicated": 0}, attachments=[])

    @contextmanager
    def scope(kind, value=None):
        env.depth[kind] += 1
        env.events.append(("enter", kind, value))
        try:
            yield
        finally:
            env.depth[kind] -= 1
            env.events.append(("exit", kind, value))

    _module(monkeypatch, "vllm_hcu.deepseek_v4_runtime",
            config_field=lambda value, key, default=None: getattr(value, key, default),
            model_architectures=lambda config: tuple(config.model_config.architectures))
    _module(monkeypatch, "vllm_hcu.v1.glm53_pcp", _current_pcp_batch=lambda: env.batch)
    _module(monkeypatch, "vllm_hcu.model_executor.layers.attention.pcp",
            logical_pcp_metadata_scope=lambda width: scope("logical", width),
            replicated_mtp_batch_scope=lambda: scope("replicated"))
    _module(monkeypatch, "vllm_hcu.patch.worker.framework_opt.patch_pcp_model_state",
            _attach_pcp_cache_ownership=lambda value, owned: env.attachments.append((value, owned)))
    _module(monkeypatch, "vllm.v1.worker.gpu.attn_utils",
            build_slot_mappings_by_layer=lambda slots, config: {"global": slots})
    env.swa_cls = type("DeepseekSparseSWAMetadataBuilder", (), {})
    _module(monkeypatch, "vllm.v1.attention.backends.mla.sparse_swa",
            DeepseekSparseSWAMetadataBuilder=env.swa_cls)
    _module(monkeypatch, "vllm_hcu.runtime_compat")
    _source(monkeypatch, "vllm_hcu.runtime_compat.deepseek_v41_pcp",
            "vllm_hcu/runtime_compat/deepseek_v41_pcp.py")
    env.runtime = _source(monkeypatch, "_v41_logical_builder_scope_runtime",
                          "vllm_hcu/v1/deepseek_v41_pcp.py")
    assert callable(getattr(env.runtime, "_logical_builder_scope", None)), "R3 runtime API is not ready"
    yield env
    assert env.depth == {"logical": 0, "replicated": 0}


def _tuple(builder):
    return tuple(getattr(builder, field) for field in FIELDS)


def _group(builder, requests=None):
    def get_metadata_builder(index):
        if requests is not None:
            requests.append(index)
        return builder
    return NS(get_metadata_builder=get_metadata_builder)


@pytest.mark.parametrize("rank", range(8))
def test_all_physical_ranks_observe_consistent_logical_tuple(harness, rank):
    builder = NS(pcp_world_size=8, use_pcp=True, pcp_rank=rank)
    saved = _tuple(builder)
    requests = []
    with harness.runtime._logical_builder_scope([[_group(builder, requests)]], 3):
        assert _tuple(builder) == (1, False, 0)
    assert _tuple(builder) == saved
    assert requests == [3]


@pytest.mark.parametrize("rank", range(8))
def test_body_exception_restores_exact_tuple(harness, rank):
    builder = NS(pcp_world_size=8, use_pcp=True, pcp_rank=rank)
    saved = _tuple(builder)
    error = ValueError("native builder failed")
    with pytest.raises(ValueError) as caught:
        with harness.runtime._logical_builder_scope([[_group(builder)]], 0):
            assert _tuple(builder) == (1, False, 0)
            raise error
    assert caught.value is error
    assert _tuple(builder) == saved


def test_nested_shared_aliases_restore_outer_scope_then_original(harness):
    builder = NS(pcp_world_size=8, use_pcp=True, pcp_rank=7)
    groups = [[_group(builder), _group(builder)], [_group(builder)]]
    with harness.runtime._logical_builder_scope(groups, 2):
        assert _tuple(builder) == (1, False, 0)
        with pytest.raises(LookupError):
            with harness.runtime._logical_builder_scope(groups, 4):
                assert _tuple(builder) == (1, False, 0)
                raise LookupError("inner native build")
        assert _tuple(builder) == (1, False, 0)
    assert _tuple(builder) == (8, True, 7)


def test_unique_builders_restore_in_reverse_and_alias_is_changed_once(harness):
    events = []

    class ObservedBuilder:
        def __setattr__(self, name, value):
            if name in FIELDS and getattr(self, "observe", False):
                events.append((self.label, name, value))
            object.__setattr__(self, name, value)

    first, second = ObservedBuilder(), ObservedBuilder()
    for builder, label, rank in ((first, "first", 1), (second, "second", 7)):
        builder.label = label
        builder.pcp_world_size, builder.use_pcp, builder.pcp_rank = 8, True, rank
        builder.observe = True
    groups = [[_group(first), _group(second), _group(first)], [_group(second)]]
    with harness.runtime._logical_builder_scope(groups, 0):
        assert _tuple(first) == _tuple(second) == (1, False, 0)
        assert len(events) == 6
    assert len(events) == 12
    assert [label for label, _, _ in events[6:]] == ["second"] * 3 + ["first"] * 3
    assert _tuple(first) == (8, True, 1)
    assert _tuple(second) == (8, True, 7)


def test_exact_original_field_objects_survive(harness):
    originals = (object(), object(), object())
    builder = NS(**dict(zip(FIELDS, originals)))
    with harness.runtime._logical_builder_scope([[_group(builder)]], 0):
        assert _tuple(builder) == (1, False, 0)
    assert all(getattr(builder, key) is value for key, value in zip(FIELDS, originals))


def test_local_swa_without_pcp_fields_is_unchanged(harness):
    builder = harness.swa_cls()
    builder.no_replay_start = torch.zeros(2, dtype=torch.int32)
    saved = dict(vars(builder))
    with harness.runtime._logical_builder_scope([[_group(builder)]], 5):
        assert vars(builder) == saved
        assert not any(hasattr(builder, key) for key in FIELDS)
    assert vars(builder) == saved


@pytest.mark.parametrize("present", [("pcp_world_size",), ("use_pcp",), ("pcp_rank",),
                                     ("pcp_world_size", "use_pcp"),
                                     ("pcp_world_size", "pcp_rank"),
                                     ("use_pcp", "pcp_rank")])
def test_partial_schema_fails_closed_and_restores_prior_builder(harness, present):
    full = NS(pcp_world_size=8, use_pcp=True, pcp_rank=7)
    partial = NS(**{field: 8 if field == "pcp_world_size" else True if field == "use_pcp" else 7
                    for field in present})
    saved = dict(vars(partial))
    entered = False
    with pytest.raises(RuntimeError, match="(?i)pcp|builder|schema"):
        with harness.runtime._logical_builder_scope([[_group(full), _group(partial)]], 0):
            entered = True
    assert not entered
    assert _tuple(full) == (8, True, 7)
    assert vars(partial) == saved


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("raises", [False, True], ids=["normal", "exception"])
def test_profile_prepare_calls_native_with_logical_tuple(harness, rank, raises):
    builder = NS(pcp_world_size=8, use_pcp=True, pcp_rank=rank)
    groups, calls = [[_group(builder)]], []
    result = NS(native_result=True)
    def original(*args):
        calls.append(args)
        assert _tuple(builder) == (1, False, 0)
        assert harness.depth == {"logical": 1, "replicated": 1}
        if raises:
            raise ValueError("profile build")
        return result
    state = NS(_hcu_v41_original_prepare_attn=original)
    args = (state, object(), object(), object(), object(), groups, object())
    if raises:
        with pytest.raises(ValueError, match="profile build"):
            harness.runtime._prepare_attn(*args, ubatch_idx=4)
        assert harness.attachments == []
    else:
        assert harness.runtime._prepare_attn(*args, ubatch_idx=4) is result
        assert harness.attachments == [(result, False)]
    assert len(calls) == 1 and calls[0][7] == 4
    assert _tuple(builder) == (8, True, rank)


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("raises", [False, True], ids=["normal", "exception"])
def test_global_prepare_scopes_indexer_and_preserves_local_swa(harness, rank, raises):
    builder = NS(pcp_world_size=8, use_pcp=True, pcp_rank=rank)
    swa = harness.swa_cls()
    swa.no_replay_start = torch.zeros(1, dtype=torch.int32)
    swa_group, index_group = _group(swa), _group(builder)
    groups = [[swa_group], [index_group, index_group]]
    local = NS(num_tokens_after_padding=4, num_reqs=1,
               positions=torch.arange(4), is_padding=torch.tensor([False] * 4))
    global_batch = NS(num_tokens=4, num_reqs=1, positions=torch.arange(4))
    tables, slots = torch.tensor([[3]]), torch.arange(4).reshape(1, 4)
    harness.batch = NS(local_batch=local, global_batch=global_batch,
                       manager=NS(prepare_global_attn=lambda: (tables, slots)),
                       local_indices=torch.arange(4))
    calls = []
    def original(*args):
        calls.append(args)
        assert harness.depth == {"logical": 1, "replicated": 1}
        assert not any(hasattr(swa, key) for key in FIELDS)
        if args[0] is global_batch:
            assert _tuple(builder) == (1, False, 0)
            if raises:
                raise ValueError("global build")
        else:
            assert args[0] is local and args[4] == [[swa_group], []]
            assert _tuple(builder) == (8, True, rank)
        return {"metadata": NS(value=torch.tensor([len(calls)]))}
    state = NS(_hcu_v41_original_prepare_attn=original)
    args = (state, local, object(), tables, slots, groups, object())
    if raises:
        with pytest.raises(ValueError, match="global build"):
            harness.runtime._prepare_attn(*args, ubatch_idx=6)
        assert len(calls) == 1
    else:
        result = harness.runtime._prepare_attn(*args, ubatch_idx=6)
        assert len(calls) == 2 and result is harness.batch._hcu_v41_metadata[0]
        assert len(harness.attachments) == 2
    assert _tuple(builder) == (8, True, rank)


@pytest.fixture
def compressed_builder():
    """Native CR2 fields consumed by the final production build guard."""
    calls = []

    class Builder:
        def __init__(self):
            self.pcp_world_size, self.use_pcp, self.pcp_rank = 8, True, 7
            self.compress_ratio = 2
            self.compressed_slot_mapping_buffer = torch.full((16,), -1, dtype=torch.int64)

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
            calls.append((self, common_prefix_len, common_attn_metadata, fast_build, kwargs))
            return NS(native_tuple=_tuple(self), slot_mapping=common_attn_metadata.slot_mapping)

    metadata = NS(num_reqs=1, num_actual_tokens=4,
                  slot_mapping=torch.arange(4, dtype=torch.int64),
                  query_start_loc=torch.tensor([0, 4], dtype=torch.int32))
    return NS(builder=Builder(), metadata=metadata, calls=calls)


@pytest.mark.parametrize("slot_count", [0, 1, 3])
def test_compressed_guard_rejects_nonzero_short_slots_before_native(harness, compressed_builder, slot_count):
    fixture = compressed_builder
    builder, metadata = fixture.builder, fixture.metadata
    metadata.slot_mapping = torch.arange(slot_count, dtype=torch.int64)
    groups = [[_group(builder), _group(builder)]]
    with pytest.raises(RuntimeError, match="compressed-slot metadata has insufficient rows or capacity"):
        with harness.runtime._logical_builder_scope(groups, 0), harness.runtime._compressed_slot_guard_scope(groups, 0):
            builder.build(0, metadata)
    assert fixture.calls == []
    assert "build" not in vars(builder)
    assert _tuple(builder) == (8, True, 7)


@pytest.mark.parametrize("malformed", ["output_capacity", "output_dtype", "slot_dtype", "query_rows"])
def test_compressed_guard_checks_actual_capacity_and_schema(harness, compressed_builder, malformed):
    fixture = compressed_builder
    builder, metadata = fixture.builder, fixture.metadata
    if malformed == "output_capacity":
        builder.compressed_slot_mapping_buffer = torch.empty(3, dtype=torch.int64)
    elif malformed == "output_dtype":
        builder.compressed_slot_mapping_buffer = torch.empty(16, dtype=torch.int32)
    elif malformed == "slot_dtype":
        metadata.slot_mapping = metadata.slot_mapping.to(torch.int32)
    else:
        metadata.query_start_loc = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="compressed-slot metadata has insufficient rows or capacity"):
        with harness.runtime._compressed_slot_guard_scope([[_group(builder)]], 0):
            builder.build(0, metadata)
    assert fixture.calls == []
    assert "build" not in vars(builder)


@pytest.mark.parametrize("instance_override", [False, True], ids=["class_method", "instance_method"])
def test_compressed_guard_restores_exact_method_binding_and_forwards(harness, compressed_builder, instance_override):
    fixture = compressed_builder
    builder, metadata = fixture.builder, fixture.metadata
    original = builder.build
    if instance_override:
        builder.build = MethodType(original.__func__, builder)
        original = builder.build
    groups = [[_group(builder), _group(builder)]]
    with harness.runtime._logical_builder_scope(groups, 2), harness.runtime._compressed_slot_guard_scope(groups, 2):
        assert builder.build is not original
        result = builder.build(9, metadata, True, marker="native")
        assert result.native_tuple == (1, False, 0)
        assert result.slot_mapping is metadata.slot_mapping
    assert fixture.calls == [(builder, 9, metadata, True, {"marker": "native"})]
    assert builder.build.__self__ is builder and builder.build.__func__ is original.__func__
    if instance_override:
        assert vars(builder)["build"] is original
    else:
        assert "build" not in vars(builder)
    assert _tuple(builder) == (8, True, 7)


def test_compressed_guard_nested_exception_restores_outer_binding_then_class(harness, compressed_builder):
    fixture = compressed_builder
    builder, metadata = fixture.builder, fixture.metadata
    original = builder.build
    groups = [[_group(builder)], [_group(builder)]]
    error = LookupError("inner native build")
    with harness.runtime._logical_builder_scope(groups, 0), harness.runtime._compressed_slot_guard_scope(groups, 0):
        outer = builder.build
        with pytest.raises(LookupError) as caught:
            with harness.runtime._logical_builder_scope(groups, 1), harness.runtime._compressed_slot_guard_scope(groups, 1):
                assert builder.build is not outer
                assert builder.build(0, metadata).native_tuple == (1, False, 0)
                raise error
        assert caught.value is error
        assert builder.build is outer and _tuple(builder) == (1, False, 0)
        assert builder.build(0, metadata).native_tuple == (1, False, 0)
    assert len(fixture.calls) == 2
    assert "build" not in vars(builder)
    assert builder.build.__self__ is builder and builder.build.__func__ is original.__func__
    assert _tuple(builder) == (8, True, 7)


def test_compressed_guard_setup_exception_restores_prior_wrapped_builder(harness, compressed_builder):
    builder = compressed_builder.builder
    malformed = NS(compress_ratio=2, build=lambda *args: None)
    with pytest.raises(RuntimeError, match="CR2 metadata builder has no compressed-slot contract"):
        with harness.runtime._compressed_slot_guard_scope([[_group(builder), _group(malformed)]], 0):
            pytest.fail("incomplete native CR2 builder entered body")
    assert "build" not in vars(builder)
    assert compressed_builder.calls == []
