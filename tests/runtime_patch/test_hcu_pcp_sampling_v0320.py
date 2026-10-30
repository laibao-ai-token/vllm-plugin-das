# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Portable CPU checks for the v0.32.0 HCU sampling ABI.

Adjacent source is the default: Path(__file__).resolve().parents[2]. The optional
HCU_SAMPLING_SOURCE_ROOT selects a pre-change control. Manager runtime bodies
and runner sample_tokens are AST-loaded unchanged, without full core/HCU imports
or constructors. CPU all_gather, metadata/model/slot builders, replicated scope
and super().sample_tokens are explicit unit boundaries. The super fake models
the versioned triple caller and nonfinal global_batch access; it is NOT the real
native sampling method or a native integration claim. The lead's separate full
native caller probe owns that evidence. Actual runner warmup helpers are loaded.

Eight serial CPU ranks use literal global rows and a fixed permutation with
uneven padding, not a mirrored restore algorithm as oracle. No real collective,
GPU, worker startup, draft/gate acceptance, accuracy or performance is claimed.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
import enum
import os
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest
import torch

ROOT = Path(os.environ.get("HCU_SAMPLING_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
MANAGER_PATH = ROOT / "vllm_hcu/v1/pcp_manager.py"
RUNNER_PATH = ROOT / "vllm_hcu/v1/hcu_model_runner_v2.py"
BOUNDARIES = (
    "AST-selected adjacent source with unchanged runtime bodies",
    "manager constructor bypass and serial CPU process-group boundary",
    "fake native triple caller and nonfinal global_batch reader",
    "controlled model/metadata/slot-mapping/replicated-scope boundaries",
    "no full core/HCU import, GPU or real distributed collective",
)


class PatchCompatibilityError(RuntimeError):
    pass


class GraphMode(enum.Enum):
    NONE = 0


def execute(nodes, scope, filename):
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), scope)


@pytest.fixture
def manager_api():
    tree = ast.parse(MANAGER_PATH.read_text(), filename=str(MANAGER_PATH))
    # Keep only the sampling methods/properties, without changing their bodies.
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == "HcuPCPManager")
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and
               n.name in {"restore_hidden_states", "restore_for_sampling", "global_batch"}]
    selected = ast.ClassDef(name=original.name, bases=[], keywords=[],
                           body=methods, decorator_list=[])
    nodes = [selected if n is original else n for n in tree.body if n is original or (
        isinstance(n, ast.FunctionDef) and n.name == "maybe_restore_pcp_for_sampling"
    ) or (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
            t.id == "_AUX_HIDDEN_UNSET" for t in n.targets)
    ) or (
        isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and
        n.target.id == "_AUX_HIDDEN_UNSET"
    )]
    scope = dict(torch=torch, np=np, dataclass=dataclass, replace=replace,
                 NamedTuple=NamedTuple, PatchCompatibilityError=PatchCompatibilityError)
    execute(nodes, scope, MANAGER_PATH)
    return SimpleNamespace(**scope)


# Numbers are global row IDs; -1 is padding. Equal gather width, uneven useful
# row counts, and deliberately non-monotone order within and between ranks.
RANK_ROWS = (
    (0, 15, -1, -1), (2, 16, -1, -1), (4, 17, 18, -1),
    (6, 13, -1, -1), (8, 11, -1, -1), (10, 9, -1, -1),
    (12, 7, -1, -1), (14, 5, 3, 1),
)
RESTORE_INDEX = (0, 31, 4, 30, 8, 29, 12, 25, 16, 21, 20, 17, 24, 13, 28, 1, 5, 9, 10)


def global_tensors():
    hidden = torch.tensor([
        [101, 201], [102, 202], [103, 203], [104, 204], [105, 205],
        [106, 206], [107, 207], [108, 208], [109, 209], [110, 210],
        [111, 211], [112, 212], [113, 213], [114, 214], [115, 215],
        [116, 216], [117, 217], [118, 218], [119, 219]], dtype=torch.float32)
    integers = torch.tensor([41, -7, 90, 12, 55, 3, 81, 24, 67, 18,
                             99, 5, 36, 72, 14, 63, 8, 47, 29], dtype=torch.int64)
    cubes = torch.tensor([
        [[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]],
        [[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]],
        [[25, 26], [27, 28]], [[29, 30], [31, 32]], [[33, 34], [35, 36]],
        [[37, 38], [39, 40]], [[41, 42], [43, 44]], [[45, 46], [47, 48]],
        [[49, 50], [51, 52]], [[53, 54], [55, 56]], [[57, 58], [59, 60]],
        [[61, 62], [63, 64]], [[65, 66], [67, 68]], [[69, 70], [71, 72]],
        [[73, 74], [75, 76]]], dtype=torch.float16)
    flags = torch.tensor([[True], [False], [True], [True], [False], [False],
                          [True], [False], [True], [False], [False], [True],
                          [True], [False], [True], [False], [True], [True],
                          [False]], dtype=torch.bool)
    return [hidden, integers, cubes, flags]


def local_rows(global_tensor, rows):
    padding = torch.zeros_like(global_tensor[0])
    return torch.stack([padding if row == -1 else global_tensor[row] for row in rows])


class SerialGather:
    """Known CPU collective boundary; rank-ordered equal-width outputs."""
    def __init__(self, rank, globals_):
        self.rank_in_group = rank
        self.world_size = 8
        self.locals = [[local_rows(t, rows) for rows in RANK_ROWS] for t in globals_]
        self.calls = []

    def all_gather(self, tensor, dim):
        assert dim == 0
        self.calls.append(tensor)
        for ranks in self.locals:
            if tensor is ranks[self.rank_in_group]:
                return torch.cat(ranks, dim=0)
        raise AssertionError("unexpected tensor at the CPU collective boundary")


def make_manager(api, rank=0):
    expected = global_tensors()
    group = SerialGather(rank, expected)
    manager = object.__new__(api.HcuPCPManager)
    manager._pcp_group = group
    manager._hidden_restore_idx = torch.tensor(RESTORE_INDEX, dtype=torch.int64)
    manager._global_batch = SimpleNamespace(
        req_ids=["request-a", "request-b", "request-c"], num_tokens=19,
        positions=torch.tensor([31, 32, 33, 34, 35]),
        idx_mapping_np=np.array([7, 2, 9], dtype=np.int32))
    calls = []
    actual_restore = manager.restore_hidden_states

    def counted_restore(tensor):
        calls.append(tensor)
        return actual_restore(tensor)

    manager.restore_hidden_states = counted_restore
    return SimpleNamespace(manager=manager, group=group, calls=calls,
        local=[ranks[rank] for ranks in group.locals], expected=expected)


def assert_calls(actual, expected):
    assert len(actual) == len(expected)
    assert all(a is e for a, e in zip(actual, expected))


def batch_snapshot(batch):
    return (list(batch.req_ids), batch.num_tokens, batch.positions.clone(),
            batch.idx_mapping_np.copy())


def assert_batch_unchanged(batch, saved):
    assert batch.req_ids == saved[0] and batch.num_tokens == saved[1]
    torch.testing.assert_close(batch.positions, saved[2])
    np.testing.assert_array_equal(batch.idx_mapping_np, saved[3])


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("aux_kind", ["none", "empty", "list"])
def test_explicit_aux_triple_restores_each_tensor_once_in_global_order(manager_api, rank, aux_kind):
    h = make_manager(manager_api, rank)
    aux = None if aux_kind == "none" else ([] if aux_kind == "empty" else h.local[1:])
    inputs = [h.local[0]] + ([] if aux is None else aux)
    saved_inputs = [t.clone() for t in inputs]
    saved_batch = batch_snapshot(h.manager._global_batch)
    result = h.manager.restore_for_sampling(h.local[0], aux)
    assert isinstance(result, tuple) and len(result) == 3
    restored, restored_aux, batch = result
    torch.testing.assert_close(restored, h.expected[0])
    if aux is None:
        assert restored_aux is None
    else:
        assert isinstance(restored_aux, list) and len(restored_aux) == len(aux)
        for actual, expected in zip(restored_aux, h.expected[1:]):
            torch.testing.assert_close(actual, expected)
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert_calls(h.calls, inputs)
    assert_calls(h.group.calls, inputs)
    assert batch is h.manager._global_batch
    assert_batch_unchanged(batch, saved_batch)
    for actual, saved in zip(inputs, saved_inputs):
        torch.testing.assert_close(actual, saved)


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("hidden_none", [False, True])
def test_legacy_omitted_aux_is_pair_and_none_pp_does_not_gather(manager_api, rank, hidden_none):
    h = make_manager(manager_api, rank)
    hidden = None if hidden_none else h.local[0]
    result = h.manager.restore_for_sampling(hidden)
    assert isinstance(result, tuple) and len(result) == 2
    restored, batch = result
    assert batch is h.manager._global_batch
    if hidden_none:
        assert restored is None
    else:
        torch.testing.assert_close(restored, h.expected[0])
    assert_calls(h.calls, [] if hidden_none else [hidden])
    assert_calls(h.group.calls, [] if hidden_none else [hidden])


@pytest.mark.parametrize("aux_kind", ["none", "empty", "list"])
def test_explicit_none_hidden_retains_triple_and_restores_only_aux(manager_api, aux_kind):
    h = make_manager(manager_api)
    aux = None if aux_kind == "none" else ([] if aux_kind == "empty" else h.local[1:])
    result = h.manager.restore_for_sampling(None, aux)
    assert len(result) == 3 and result[0] is None and result[2] is h.manager._global_batch
    if aux is None:
        assert result[1] is None
    else:
        assert isinstance(result[1], list) and len(result[1]) == len(aux)
        for actual, expected in zip(result[1], h.expected[1:]):
            torch.testing.assert_close(actual, expected)
    assert_calls(h.calls, [] if aux is None else aux)
    assert_calls(h.group.calls, [] if aux is None else aux)


def test_explicit_none_is_distinct_from_omitted_sentinel(manager_api):
    assert "_AUX_HIDDEN_UNSET" in vars(manager_api)
    h = make_manager(manager_api)
    omitted = h.manager.restore_for_sampling(None)
    explicit = h.manager.restore_for_sampling(None, None)
    assert len(omitted) == 2 and len(explicit) == 3
    assert omitted[0] is explicit[0] is explicit[1] is None
    assert omitted[1] is explicit[2] is h.manager._global_batch
    assert h.calls == h.group.calls == []


@pytest.mark.parametrize("malformed", ["tensor", "tuple", "dict", "scalar", "bad-first", "bad-late", "none-item", "scalar-tensor", "mismatched-rows"])
def test_malformed_aux_is_validated_before_any_restore_or_gather(manager_api, malformed):
    h = make_manager(manager_api)
    invalid = {"tensor": h.local[1], "tuple": (h.local[1],), "dict": {"aux": h.local[1]},
               "scalar": 7, "bad-first": [object(), h.local[1]],
               "bad-late": [h.local[1], h.local[2], object()], "none-item": [None],
               "scalar-tensor": [h.local[1], torch.tensor(2)],
               "mismatched-rows": [h.local[1], h.local[2][:2]]}[malformed]
    saved = batch_snapshot(h.manager._global_batch)
    with pytest.raises((PatchCompatibilityError, TypeError, ValueError)):
        h.manager.restore_for_sampling(h.local[0], invalid)
    assert h.calls == h.group.calls == []
    assert_batch_unchanged(h.manager._global_batch, saved)


@pytest.mark.parametrize("hidden_none", [False, True])
def test_legacy_helper_stays_pair_and_manager_none_is_identity(manager_api, hidden_none):
    h = make_manager(manager_api)
    local_batch = object()
    hidden = None if hidden_none else h.local[0]
    direct = manager_api.maybe_restore_pcp_for_sampling(None, hidden, local_batch)
    assert len(direct) == 2 and direct[0] is hidden and direct[1] is local_batch
    restored = manager_api.maybe_restore_pcp_for_sampling(h.manager, hidden, local_batch)
    assert len(restored) == 2 and restored[1] is h.manager._global_batch
    if hidden_none:
        assert restored[0] is None
    else:
        torch.testing.assert_close(restored[0], h.expected[0])
    assert_calls(h.calls, [] if hidden_none else [hidden])


def test_global_batch_property_exposes_saved_batch_without_restoring_hidden(manager_api):
    h = make_manager(manager_api)
    assert isinstance(vars(manager_api.HcuPCPManager).get("global_batch"), property)
    assert h.manager.global_batch is h.manager._global_batch
    assert h.manager.global_batch is h.manager._global_batch
    assert h.calls == h.group.calls == []


class State(NamedTuple):
    input_batch: object
    hidden_states: object
    aux_hidden_states: object
    attn_metadata: object
    slot_mappings_by_layer: object
    finished_req_ids: object


class LegacyState(NamedTuple):
    input_batch: object
    hidden_states: object
    attn_metadata: object
    slot_mappings_by_layer: object
    finished_req_ids: object


class InjectedFailure(RuntimeError):
    pass


@pytest.fixture
def runner_factory(manager_api):
    """Actual HCU sample_tokens over a deliberately fake v0.32 native caller."""
    def factory(*, aux_kind="list", legacy=False, final=True, speculative=True,
                v41=True, pcp=True, failure=None):
        h = make_manager(manager_api)
        events = []
        active = ContextVar("test_sampling_scope", default=False)
        warmup = ContextVar("test_sampling_warmup", default=False)
        observations, api_calls = [], []
        actual_api = h.manager.restore_for_sampling

        def record_restore(*args):
            events.append("restore")
            api_calls.append(args)
            return actual_api(*args)

        h.manager.restore_for_sampling = record_restore

        @contextmanager
        def replicated_scope():
            token = active.set(True)
            events.append("scope.enter")
            try:
                yield
            finally:
                active.reset(token)
                events.append("scope.exit")

        class NativeCallerBoundary:
            # This is a versioned known fake, NOT native sample_tokens code.
            def sample_tokens(self, grammar_output):
                events.append("super")
                assert grammar_output == "grammar"
                manager = self.pcp_manager
                if self.is_last_pp_rank and manager is not None:
                    state = self.execute_model_state
                    hidden, aux, batch = manager.restore_for_sampling(
                        state.hidden_states, getattr(state, "aux_hidden_states", None))
                    changes = dict(hidden_states=hidden, input_batch=batch)
                    if hasattr(state, "aux_hidden_states"):
                        changes["aux_hidden_states"] = aux
                    self.execute_model_state = state._replace(**changes)
                elif not self.is_last_pp_rank and manager is not None:
                    assert manager.global_batch is h.manager._global_batch
                    assert self.execute_model_state.input_batch is manager.global_batch
                observations.append((manager, self.execute_model_state, active.get()))
                if failure == "sampler":
                    raise InjectedFailure("sampler")
                return "sampled" if self.is_last_pp_rank else "received"

        def prepare_global():
            events.append("global_attn")
            assert active.get()
            if failure == "global-attn":
                raise InjectedFailure("global-attn")
            return "global-block-tables", "global-slot-mappings"

        def slot_map(slots, config):
            events.append("slot-map")
            assert slots == "global-slot-mappings"
            return {"layer-0": slots}

        def prepare_metadata(batch, mode, blocks, slots, groups, config):
            events.append("metadata")
            assert batch is h.manager._global_batch and mode is GraphMode.NONE
            assert blocks == "global-block-tables" and slots == "global-slot-mappings"
            assert active.get()
            if failure == "metadata":
                raise InjectedFailure("metadata")
            return "global-metadata"

        def record_index(runner, batch):
            events.append("record-index")

        h.manager.prepare_global_attn = prepare_global
        tree = ast.parse(RUNNER_PATH.read_text(), filename=str(RUNNER_PATH))
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                        and n.name == "HcuGPUModelRunnerV2")
        method = next(n for n in original.body if isinstance(n, ast.FunctionDef)
                      and n.name == "sample_tokens")
        cls = ast.ClassDef(name=original.name, bases=[ast.Name(id="GPUModelRunner", ctx=ast.Load())],
                           keywords=[], body=[method], decorator_list=[])
        helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                   {"suppress_sampler_warmup_logprobs", "sampler_warmup_scope"}]
        namespace = dict(GPUModelRunner=NativeCallerBoundary, torch=torch, np=np,
            contextmanager=contextmanager, nullcontext=nullcontext,
            _SAMPLER_WARMUP=warmup, replicated_mtp_batch_scope=replicated_scope,
            record_pp_spec_draft_index_stream=record_index,
            is_deepseek_v4=lambda config: config.v41,
            build_slot_mappings_by_layer=slot_map, CUDAGraphMode=GraphMode)
        execute(helpers + [cls], namespace, RUNNER_PATH)
        runner = object.__new__(namespace[original.name])
        runner.pcp_manager = h.manager if pcp else None
        runner.vllm_config = SimpleNamespace(v41=v41,
            parallel_config=SimpleNamespace(prefill_context_parallel_size=8 if pcp else 1))
        runner.speculator = object() if speculative else None
        runner.is_last_pp_rank = final
        runner.kv_cache_config, runner.attn_groups = object(), object()
        runner.model_state = SimpleNamespace(prepare_attn=prepare_metadata)
        runner.model = SimpleNamespace(get_mtp_target_hidden_states=lambda: None)
        runner.sampler = SimpleNamespace(sampling_states=SimpleNamespace(
            num_logprobs=np.array([5, 9, 7, 3, 1, 8, 6, 4, 2, 10], dtype=np.int32)))
        aux = None if aux_kind == "none" else ([] if aux_kind == "empty" else h.local[1:])
        hidden = h.local[0] if final else None
        local_batch = SimpleNamespace(req_ids=["local"], num_tokens=4,
            idx_mapping_np=np.array([7, 2, 9], dtype=np.int32), num_reqs=3)
        h.manager._global_batch.num_reqs = 3
        tail = ("local-metadata", {"local": "slots"}, ["finished-request"])
        runner.execute_model_state = (LegacyState(local_batch, hidden, *tail) if legacy
                                     else State(local_batch, hidden, aux, *tail))
        return SimpleNamespace(runner=runner, manager=h.manager, h=h, events=events,
            active=active, warmup=warmup, namespace=namespace, observations=observations,
            api_calls=api_calls, original=runner.execute_model_state, aux=aux)
    return factory


@pytest.mark.parametrize("aux_kind", ["none", "empty", "list"])
@pytest.mark.parametrize("legacy", [False, True])
def test_mtp_early_triple_preserves_aux_rebuilds_metadata_and_hides_manager(runner_factory, aux_kind, legacy):
    f = runner_factory(aux_kind=aux_kind, legacy=legacy)
    assert f.runner.sample_tokens("grammar") == "sampled"
    assert len(f.api_calls) == 1 and len(f.api_calls[0]) == 2
    assert f.api_calls[0][0] is f.original.hidden_states
    assert f.api_calls[0][1] is (None if legacy else f.aux)
    state = f.runner.execute_model_state
    assert state is not f.original and state.input_batch is f.manager.global_batch
    torch.testing.assert_close(state.hidden_states, f.h.expected[0])
    if legacy:
        assert not hasattr(state, "aux_hidden_states")
    elif f.aux is None:
        assert state.aux_hidden_states is None
    else:
        assert isinstance(state.aux_hidden_states, list)
        assert len(state.aux_hidden_states) == len(f.aux)
        for actual, expected in zip(state.aux_hidden_states, f.h.expected[1:]):
            torch.testing.assert_close(actual, expected)
    assert state.attn_metadata == "global-metadata"
    assert state.slot_mappings_by_layer == {"layer-0": "global-slot-mappings"}
    assert state.finished_req_ids is f.original.finished_req_ids
    assert f.original.input_batch is not state.input_batch
    assert f.original.attn_metadata == "local-metadata"
    assert f.observations[0][0] is None and f.observations[0][2] is True
    assert f.runner.pcp_manager is f.manager and f.active.get() is False
    assert f.events.index("restore") < f.events.index("metadata") < f.events.index("super")
    expected_calls = [f.h.local[0]] + ([] if legacy or f.aux is None else f.aux)
    assert_calls(f.h.calls, expected_calls)
    assert_calls(f.h.group.calls, expected_calls)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("aux_kind", ["none", "empty"])
def test_nonfinal_pp_restores_global_batch_without_hidden_operations(runner_factory, legacy, aux_kind):
    f = runner_factory(final=False, speculative=False, legacy=legacy, aux_kind=aux_kind)
    assert f.runner.sample_tokens("grammar") == "received"
    assert len(f.api_calls) == 1 and len(f.api_calls[0]) == 2
    assert f.api_calls[0][0] is None
    assert f.api_calls[0][1] is (None if legacy else f.aux)
    assert f.runner.execute_model_state.hidden_states is None
    assert f.runner.execute_model_state.input_batch is f.manager.global_batch
    assert f.h.calls == f.h.group.calls == []
    assert "global_attn" not in f.events and "metadata" not in f.events
    assert "scope.enter" not in f.events
    assert f.runner.pcp_manager is f.manager and f.active.get() is False


@pytest.mark.parametrize("aux_kind", ["none", "empty", "list"])
def test_regular_final_native_fake_owns_single_explicit_triple_restore(runner_factory, aux_kind):
    f = runner_factory(speculative=False, aux_kind=aux_kind)
    assert f.runner.sample_tokens("grammar") == "sampled"
    assert f.events.index("super") < f.events.index("restore")
    assert len(f.api_calls) == 1 and len(f.api_calls[0]) == 2
    assert f.observations[0][0] is f.manager and f.observations[0][2] is False
    torch.testing.assert_close(f.runner.execute_model_state.hidden_states, f.h.expected[0])
    assert "metadata" not in f.events and "scope.enter" not in f.events
    assert f.runner.pcp_manager is f.manager and f.active.get() is False
    expected_calls = [f.h.local[0]] + ([] if f.aux is None else f.aux)
    assert_calls(f.h.calls, expected_calls)


@pytest.mark.parametrize("failure", ["global-attn", "metadata", "sampler"])
def test_failure_unwinds_manager_scope_and_real_warmup_suppression(runner_factory, failure):
    f = runner_factory(failure=failure)
    logprobs = f.runner.sampler.sampling_states.num_logprobs
    saved = logprobs.copy()
    with f.namespace["sampler_warmup_scope"]():
        with pytest.raises(InjectedFailure, match=failure):
            f.runner.sample_tokens("grammar")
        np.testing.assert_array_equal(logprobs, saved)
        assert f.runner.pcp_manager is f.manager and f.active.get() is False
    assert f.warmup.get() is False
    assert f.events.count("scope.enter") == f.events.count("scope.exit") == 1
    assert len(f.api_calls) == 1
    # Restoration precedes late errors: no transactional rollback or general
    # absence of side effects is promised here.
    assert f.runner.execute_model_state is not f.original
    assert f.runner.execute_model_state.input_batch is f.manager.global_batch
    if failure == "sampler":
        assert f.observations[0][0] is None and f.observations[0][2] is True


@pytest.mark.parametrize("v41,pcp,speculative", [(True, False, False), (False, False, True), (False, True, False)])
def test_default_pcp1_and_nonv41_delegation_has_no_scope_leaks(runner_factory, v41, pcp, speculative):
    f = runner_factory(v41=v41, pcp=pcp, speculative=speculative, aux_kind="none")
    assert f.runner.sample_tokens("grammar") == "sampled"
    assert f.active.get() is False and "scope.enter" not in f.events
    assert "global_attn" not in f.events and "metadata" not in f.events
    assert f.runner.pcp_manager is (f.manager if pcp else None)
    if not pcp:
        assert f.api_calls == [] and f.h.calls == []
        assert f.runner.execute_model_state is f.original
    else:
        assert len(f.api_calls) == 1


def test_execute_state_none_delegates_without_pcp_restore_or_scope(runner_factory):
    f = runner_factory(pcp=False, speculative=False)
    f.runner.execute_model_state = None
    assert f.runner.sample_tokens("grammar") == "sampled"
    assert f.api_calls == [] and f.h.calls == []
    assert f.active.get() is False and "scope.enter" not in f.events
