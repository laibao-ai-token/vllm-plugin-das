# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the versioned V41 global-metadata overlay.

Only the two adjacent implementation sources are executed.  Configuration
accessors, PCP scopes, forward context, ownership attachment, and the native
slot-by-layer utility are explicit import boundaries; no installed vLLM package
or target-core source is needed.  The serial attention body in forward tests is
a numerical CPU fixture, not a native-attention integration claim.  Independent
native AST/source integration remains the Lead's responsibility.
"""
from __future__ import annotations

import importlib.util
import os
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch


# When copied into a repository, the adjacent overlay root remains the default.
_SOURCE_ROOT = Path(os.environ.get(
    "VLLM_HCU_SOURCE_ROOT", str(Path(__file__).resolve().parents[2])
)).resolve()


def _load_source(name, relative_path, monkeypatch):
    path = _SOURCE_ROOT / relative_path
    assert path.is_file(), f"missing adjacent V41 implementation: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _boundary_module(monkeypatch, name, **attributes):
    """Replace the entire import ancestry; never execute a package __init__."""
    parent = None
    for component_count in range(1, len(name.split(".")) + 1):
        current_name = ".".join(name.split(".")[:component_count])
        module = sys.modules.get(current_name)
        if module is None or not getattr(module, "_v41_cpu_boundary", False):
            module = ModuleType(current_name)
            module.__path__ = []
            module._v41_cpu_boundary = True
            monkeypatch.setitem(sys.modules, current_name, module)
        if parent is not None:
            setattr(parent, current_name.rsplit(".", 1)[-1], module)
        parent = module
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture
def harness(monkeypatch):
    """Load real adapter functions with narrow, observable CPU boundaries."""
    env = NS(
        batch=None,
        context=NS(attn_metadata=object(), slot_mapping=object(),
                   is_padding=torch.tensor([False])),
        events=[], depth={"replicated": 0, "logical": 0},
        ownership_calls=[], slot_builder_calls=[],
    )

    @contextmanager
    def scope(name, value=None):
        env.events.append(("enter", name, value))
        env.depth[name] += 1
        try:
            yield
        finally:
            env.depth[name] -= 1
            env.events.append(("exit", name, value))

    def attach(metadata, source_owned):
        env.ownership_calls.append((metadata, source_owned))

    def slots_by_layer(slots, cache_config):
        env.slot_builder_calls.append((slots, cache_config))
        # The real utility is a boundary; its returned object must survive intact.
        result = {"swa": slots[0], "indexer": slots[1]}
        env.built_slots = result
        return result

    # These are configuration data accessors, not copies of the admission
    # predicates under test.  Fixtures use this minimal attribute-based schema.
    _boundary_module(
        monkeypatch, "vllm_hcu.deepseek_v4_runtime",
        config_field=lambda value, name, default=None: getattr(value, name, default),
        model_architectures=lambda config: tuple(config.model_config.architectures),
    )
    _boundary_module(monkeypatch, "vllm.forward_context",
                     get_forward_context=lambda: env.context)
    _boundary_module(monkeypatch, "vllm_hcu.v1.glm53_pcp",
                     _current_pcp_batch=lambda: env.batch)
    _boundary_module(
        monkeypatch, "vllm_hcu.model_executor.layers.attention.pcp",
        replicated_mtp_batch_scope=lambda: scope("replicated"),
        logical_pcp_metadata_scope=lambda width: scope("logical", width),
    )
    _boundary_module(
        monkeypatch, "vllm_hcu.patch.worker.framework_opt.patch_pcp_model_state",
        _attach_pcp_cache_ownership=attach,
    )
    _boundary_module(monkeypatch, "vllm.v1.worker.gpu.attn_utils",
                     build_slot_mappings_by_layer=slots_by_layer)

    env.swa_builder_cls = type("DeepseekSparseSWAMetadataBuilder", (), {})
    _boundary_module(monkeypatch, "vllm.v1.attention.backends.mla.sparse_swa",
                     DeepseekSparseSWAMetadataBuilder=env.swa_builder_cls)

    private = f"_v41_global_metadata_cpu_{id(env)}"
    _boundary_module(monkeypatch, private)
    env.contract = _load_source(
        private + ".pureconfig",
        "vllm_hcu/runtime_compat/deepseek_v41_pcp.py", monkeypatch,
    )
    canonical = "vllm_hcu.runtime_compat.deepseek_v41_pcp"
    _boundary_module(monkeypatch, "vllm_hcu.runtime_compat")
    monkeypatch.setitem(sys.modules, canonical, env.contract)
    env.runtime = _load_source(
        private + ".runtime", "vllm_hcu/v1/deepseek_v41_pcp.py", monkeypatch,
    )
    yield env
    assert env.runtime._ACTIVE_ROWS.get() is None
    assert env.depth == {"replicated": 0, "logical": 0}


def _scope_only_layer(harness, **attributes):
    # These tests isolate transport/context restoration around a numerical toy
    # serial body. Native writer preflight is covered separately by portable
    # structural tests and the actual-source writer probe, not this toy layer.
    harness.runtime._preflight_attention = lambda layer, hidden, positions: None
    return NS(**attributes)


def _config():
    return NS(
        model_config=NS(architectures=["DeepseekV41ForCausalLM"],
                        use_mla=True, enforce_eager=True,
                        is_multimodal_model=False),
        parallel_config=NS(
            tensor_parallel_size=1, prefill_context_parallel_size=8,
            data_parallel_size=1, pipeline_parallel_size=1,
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1,
            enable_expert_parallel=True, enable_eplb=False,
            all2all_backend="deepep_high_throughput",
        ),
        use_v2_model_runner=True,
        attention_config=NS(indexer_kv_dtype="fp8", indexer_sparse_logits=False),
        cache_config=NS(kv_offloading_size=None), speculative_config=None,
        lora_config=None, kv_transfer_config=None,
    )


def _set_config(config, path, value):
    owner = config
    for name in path.split(".")[:-1]:
        owner = getattr(owner, name)
    setattr(owner, path.split(".")[-1], value)


def _batch(local_indices=(0, 1, 4, 0), local_positions=(5, 8, 2, 0)):
    global_batch = NS(
        num_tokens=10, num_tokens_after_padding=10,
        query_start_loc=torch.tensor([0, 1, 2, 10], dtype=torch.int32),
        positions=torch.tensor([5, 8, 0, 1, 2, 3, 4, 5, 6, 7]),
        input_ids=torch.tensor([15, 18, 20, 21, 22, 23, 24, 25, 26, 27]),
        is_padding=torch.tensor([False] * 10),
    )
    local_batch = NS(
        num_tokens=3, num_tokens_after_padding=4, num_reqs=3,
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        positions=torch.tensor(local_positions),
        input_ids=torch.tensor([15, 18, 22, 0]),
        is_padding=torch.tensor([False, False, False, True]),
    )
    global_tables = torch.tensor([[10, 11], [20, 21], [30, 31]])
    global_slots = torch.tensor([
        [100, 101, 200, 201, 202, 203, 204, 205, 206, 207],
        [300, 301, 400, 401, 402, 403, 404, 405, 406, 407],
    ])
    manager = NS(_global_has_prefill=True,
                 prepare_global_attn=lambda: (global_tables, global_slots))
    return NS(
        local_batch=local_batch, global_batch=global_batch, manager=manager,
        local_indices=torch.tensor(local_indices), global_tables=global_tables,
        global_slots=global_slots,
    )


def _indexer(rows=4, candidates=None):
    return NS(skip_k_cache_insert=True, k_cache=NS(prefix="indexer"),
              topk_indices_buffer=torch.empty(rows, 2, dtype=torch.int32),
              candidate_blocks=candidates)


def _global_context(harness, rows=4):
    metadata = NS(pcp_world_size=1, num_decode_tokens=1,
                  num_prefill_tokens=rows - 1)
    harness.context.attn_metadata = {"indexer": metadata}
    return metadata


def _native_cache(module="vllm.models.deepseek_v41.attention",
                  name="DeepseekV4IndexerCache"):
    return type(name, (), {"__module__": module})()


@pytest.mark.parametrize(
    "rank,prefill_row,positions,expected_slots",
    [
        (0, 2, [5, 8, 0, 0], [[100, 101, 200, -1], [300, 301, 400, -1]]),
        (1, 3, [5, 8, 1, 0], [[100, 101, 201, -1], [300, 301, 401, -1]]),
        (2, 4, [5, 8, 2, 0], [[100, 101, 202, -1], [300, 301, 402, -1]]),
        (3, 5, [5, 8, 3, 0], [[100, 101, 203, -1], [300, 301, 403, -1]]),
        (4, 6, [5, 8, 4, 0], [[100, 101, 204, -1], [300, 301, 404, -1]]),
        (5, 7, [5, 8, 5, 0], [[100, 101, 205, -1], [300, 301, 405, -1]]),
        (6, 8, [5, 8, 6, 0], [[100, 101, 206, -1], [300, 301, 406, -1]]),
        (7, 9, [5, 8, 7, 0], [[100, 101, 207, -1], [300, 301, 407, -1]]),
    ],
    ids=[f"rank-{rank}" for rank in range(8)],
)
def test_prepare_dual_metadata_snapshots_reused_scratch_and_keeps_local_view(
    harness, rank, prefill_row, positions, expected_slots,
):
    batch = harness.batch = _batch((0, 1, prefill_row, 0), positions)
    batch.rank = rank
    caller_tables = torch.tensor([[90, 91], [92, 93], [94, 95]])
    caller_slots = torch.tensor([[900, 901, 902, -1], [910, 911, 912, -1]])
    model_tensor = torch.tensor([451, 452])
    model_specific = NS(tensor=model_tensor, aliases={"same": model_tensor})
    swa_builder = harness.swa_builder_cls()
    swa_builder.no_replay_start = torch.zeros(2, dtype=torch.int32)
    swa_group = NS(get_metadata_builder=lambda index: swa_builder)
    other_group = NS(get_metadata_builder=lambda index: object())
    mode, groups, cache_config = object(), [[swa_group], [other_group]], object()
    scratch = torch.empty(4, dtype=torch.int64)
    calls, outputs = [], []

    def original(*args):
        assert harness.depth == {"replicated": 1, "logical": 1}
        calls.append(args)
        current, _, tables, slots, _, _, _, _, model_metadata = args
        # Deliberately reuse one native-builder scratch allocation in both calls.
        scratch.copy_(torch.tensor([71, 72, 73, 74] if len(calls) == 1
                                   else [81, 82, 83, 84]))
        metadata = NS(
            builder_scratch=scratch, scratch_aliases=[scratch, {"same": scratch}],
            query_start_loc=current.query_start_loc, positions=current.positions,
            input_ids=current.input_ids, tables=tables, slots=slots,
            model_specific=model_metadata, model_tensor=model_metadata.tensor,
        )
        result = {"layer": metadata, "layer_alias": metadata,
                  "scratch": scratch, "model_specific": model_metadata}
        outputs.append(result)
        return result

    state = NS(_hcu_v41_original_prepare_attn=original)
    result = harness.runtime._prepare_attn(
        state, batch.local_batch, mode, caller_tables, caller_slots,
        groups, cache_config, ubatch_idx=3,
        model_specific_attn_metadata=model_specific,
    )
    assert len(calls) == 2
    assert calls[0][0] is batch.global_batch
    assert calls[0][1] is mode
    assert calls[0][2] is batch.global_tables
    assert calls[0][3] is batch.global_slots
    assert calls[1][0] is batch.local_batch
    assert calls[1][2] is caller_tables
    assert calls[0][4] is groups
    assert calls[1][4] == [[swa_group], []]
    assert calls[1][4] is not groups
    assert swa_builder.no_replay_start.tolist() == [0, 0, 0]
    assert calls[0][5] is calls[1][5] is cache_config
    assert calls[0][6:8] == calls[1][6:8] == (False, 3)
    assert calls[0][8] is calls[1][8] is model_specific
    assert calls[1][3] is not caller_slots
    assert calls[1][3] is not batch.global_slots
    assert calls[1][3].tolist() == expected_slots
    assert caller_slots.tolist() == [[900, 901, 902, -1], [910, 911, 912, -1]]
    assert batch.global_slots.tolist() == [
        [100, 101, 200, 201, 202, 203, 204, 205, 206, 207],
        [300, 301, 400, 401, 402, 403, 404, 405, 406, 407],
    ]

    local, full, full_slots = batch._hcu_v41_metadata
    assert result is local is outputs[1]
    assert full is not outputs[0]
    assert full["layer"] is full["layer_alias"]
    assert full["layer"].builder_scratch is full["scratch"]
    assert full["layer"].scratch_aliases[0] is full["scratch"]
    assert full["layer"].scratch_aliases[1]["same"] is full["scratch"]
    assert full["scratch"] is not scratch
    assert full["scratch"].tolist() == [71, 72, 73, 74]
    assert local["scratch"] is scratch
    assert scratch.tolist() == [81, 82, 83, 84]
    assert full["layer"].query_start_loc is batch.global_batch.query_start_loc
    assert full["layer"].positions is batch.global_batch.positions
    assert full["layer"].input_ids is batch.global_batch.input_ids
    assert full["layer"].tables is batch.global_tables
    assert full["layer"].slots is batch.global_slots
    assert local["layer"].query_start_loc is batch.local_batch.query_start_loc
    assert local["layer"].query_start_loc.tolist() == [0, 1, 2, 3]
    assert local["layer"].positions is batch.local_batch.positions
    assert local["layer"].positions.tolist() == positions
    assert local["layer"].input_ids is batch.local_batch.input_ids
    assert local["layer"].tables is caller_tables
    assert local["layer"].slots is calls[1][3]
    for view in (local, full):
        assert view["model_specific"] is model_specific
        assert view["layer"].model_specific is model_specific
        assert view["layer"].model_tensor is model_tensor
    assert model_specific.aliases["same"] is model_tensor
    assert model_tensor.tolist() == [451, 452]
    assert full_slots is harness.built_slots
    assert harness.slot_builder_calls[0][0] is batch.global_slots
    assert harness.slot_builder_calls[0][1] is cache_config
    assert len(harness.ownership_calls) == 2
    assert harness.ownership_calls[0][0] is outputs[0]
    assert harness.ownership_calls[1][0] is local
    assert [owned for _, owned in harness.ownership_calls] == [False, False]
    assert harness.events == [
        ("enter", "replicated", None), ("enter", "logical", 1),
        ("exit", "logical", 1), ("exit", "replicated", None),
        ("enter", "replicated", None), ("enter", "logical", 1),
        ("exit", "logical", 1), ("exit", "replicated", None),
    ]


@pytest.mark.parametrize("explicit", [False, True], ids=["defaults", "explicit"])
def test_no_batch_profiling_delegates_exact_arguments_and_original_dict(harness, explicit):
    inputs = (object(), object(), object(), object(), [], object())
    metadata = {"original": torch.tensor([17])}
    model_specific = object()
    calls = []

    def original(*args):
        assert harness.depth == {"replicated": 1, "logical": 1}
        calls.append(args)
        return metadata

    state = NS(_hcu_v41_original_prepare_attn=original)
    keywords = (dict(for_capture=True, ubatch_idx=9,
                     model_specific_attn_metadata=model_specific) if explicit else {})
    result = harness.runtime._prepare_attn(state, *inputs, **keywords)
    assert result is metadata
    assert len(calls) == 1
    assert len(calls[0]) == 9
    assert all(actual is expected for actual, expected in zip(calls[0][:6], inputs))
    if explicit:
        assert calls[0][6:8] == (True, 9)
        assert calls[0][8] is model_specific
    else:
        assert calls[0][6:9] == (False, 0, None)
    assert len(harness.ownership_calls) == 1
    assert harness.ownership_calls[0][0] is metadata
    assert harness.ownership_calls[0][1] is False
    assert harness.slot_builder_calls == []


@pytest.mark.parametrize("bad_case", ["foreign-batch", "capture"])
def test_prepare_rejects_foreign_batch_and_capture(harness, bad_case):
    batch = harness.batch = _batch()
    calls = []
    state = NS(_hcu_v41_original_prepare_attn=lambda *args: calls.append(args))
    input_batch = NS() if bad_case == "foreign-batch" else batch.local_batch
    with pytest.raises(RuntimeError, match="current eager local batch"):
        harness.runtime._prepare_attn(
            state, input_batch, object(), object(), object(), object(), object(),
            for_capture=bad_case == "capture",
        )
    assert calls == []


@pytest.mark.parametrize("bad_case", ["indices-width", "mask-width", "mask-dtype"])
def test_local_slot_map_rejects_incomplete_or_non_boolean_fragments(harness, bad_case):
    batch = _batch()
    if bad_case == "indices-width":
        batch.local_indices = torch.tensor([0, 1, 4])
    elif bad_case == "mask-width":
        batch.local_batch.is_padding = torch.tensor([False, False])
    else:
        batch.local_batch.is_padding = torch.tensor([0, 0, 0, 1])
    with pytest.raises(RuntimeError, match="row map|padding mask"):
        harness.runtime._local_slots(batch, batch.global_slots)


def test_snapshot_preserves_caller_tensors_model_identity_cycles_and_scratch_aliases(harness):
    source_tensor = torch.tensor([11, 12])
    other_tensor = torch.tensor([21, 22])
    caller = NS(values=[source_tensor, {"other": other_tensor}])
    caller_tensor = torch.tensor([31, 32])
    model_specific = {"tensor": caller_tensor, "alias": caller_tensor}
    scratch = torch.tensor([41, 42])
    builder = NS(scratch=scratch, alias=scratch, source=source_tensor,
                 other=other_tensor, caller_tensor=caller_tensor,
                 model_specific=model_specific)
    metadata = {"builder": builder, "builder_alias": builder, "scratch": scratch}
    metadata["cycle"] = metadata
    snapshot = harness.runtime._snapshot_metadata(metadata, caller, model_specific)
    assert snapshot is not metadata
    assert snapshot["cycle"] is snapshot
    assert snapshot["builder"] is snapshot["builder_alias"]
    assert snapshot["builder"] is not builder
    assert snapshot["scratch"] is snapshot["builder"].scratch
    assert snapshot["scratch"] is snapshot["builder"].alias
    assert snapshot["scratch"] is not scratch
    assert snapshot["builder"].source is source_tensor
    assert snapshot["builder"].other is other_tensor
    assert snapshot["builder"].caller_tensor is caller_tensor
    assert snapshot["builder"].model_specific is model_specific
    assert model_specific["tensor"] is model_specific["alias"] is caller_tensor
    scratch.fill_(99)
    assert snapshot["scratch"].tolist() == [41, 42]
    assert source_tensor.tolist() == [11, 12]
    assert other_tensor.tolist() == [21, 22]
    assert caller_tensor.tolist() == [31, 32]


def _bind_views(harness, batch):
    local_metadata = {"local": object()}
    full_metadata = {"indexer": NS(pcp_world_size=1, num_decode_tokens=2,
                                   num_prefill_tokens=batch.global_batch.num_tokens - 2)}
    local_slots = {"local": torch.tensor([101, 102, -1, -1])}
    full_slots = {"global": batch.global_slots}
    local_padding = batch.local_batch.is_padding
    harness.context.attn_metadata = local_metadata
    harness.context.slot_mapping = local_slots
    harness.context.is_padding = local_padding
    batch._hcu_v41_metadata = (local_metadata, full_metadata, full_slots)
    return local_metadata, local_slots, local_padding


def _assert_restored(harness, saved):
    assert harness.context.attn_metadata is saved[0]
    assert harness.context.slot_mapping is saved[1]
    assert harness.context.is_padding is saved[2]
    assert harness.runtime._ACTIVE_ROWS.get() is None
    assert harness.depth == {"replicated": 0, "logical": 0}


class _SerialFailure(Exception):
    pass


@pytest.mark.parametrize("fail", [False, True], ids=["normal", "exception"])
def test_attention_scope_restores_exact_objects_and_nested_scope(harness, fail):
    batch = _batch()
    saved = _bind_views(harness, batch)
    local_positions = batch.local_batch.positions
    values_before = local_positions.clone()
    error = _SerialFailure("attention failed")

    def run():
        with harness.runtime._attention_scope(batch, 10):
            assert harness.context.attn_metadata is batch._hcu_v41_metadata[1]
            assert harness.context.slot_mapping is batch._hcu_v41_metadata[2]
            assert harness.context.is_padding.tolist() == [False] * 10
            assert harness.context.is_padding.data_ptr() == batch.global_batch.is_padding.data_ptr()
            outer_active = harness.runtime._ACTIVE_ROWS.get()
            assert outer_active[0] == 10
            assert outer_active[1] is batch._hcu_v41_metadata[1]
            assert harness.depth == {"replicated": 1, "logical": 1}
            with harness.runtime._attention_scope(None, 2):
                inner = harness.runtime._ACTIVE_ROWS.get()
                assert inner[0] == 2
                assert inner[1] is batch._hcu_v41_metadata[1]
            assert harness.runtime._ACTIVE_ROWS.get() is outer_active
            if fail:
                raise error

    if fail:
        with pytest.raises(_SerialFailure) as caught:
            run()
        assert caught.value is error
    else:
        run()
    _assert_restored(harness, saved)
    assert batch.local_batch.positions is local_positions
    assert torch.equal(local_positions, values_before)


@pytest.mark.parametrize("bad_case", ["missing", "stale"])
def test_attention_scope_rejects_missing_or_stale_views_without_mutation(harness, bad_case):
    batch = _batch()
    saved = _bind_views(harness, batch)
    if bad_case == "missing":
        del batch._hcu_v41_metadata
    else:
        batch._hcu_v41_metadata = ({"other-step": object()}, object(), object())
    with pytest.raises(RuntimeError, match="stale or missing metadata"):
        with harness.runtime._attention_scope(batch, 10):
            pytest.fail("stale metadata must not enter attention")
    _assert_restored(harness, saved)


@pytest.mark.parametrize("fail", [False, True], ids=["normal", "exception"])
def test_forward_gathers_local_shards_restores_scope_and_localizes_once(harness, fail):
    batch = harness.batch = _batch()
    saved = _bind_views(harness, batch)
    local_positions = batch.local_batch.positions
    local_hidden = torch.tensor([[1., 2.], [3., 4.], [9., 10.], [900., 901.]])
    global_hidden = torch.tensor([
        [1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.],
        [11., 12.], [13., 14.], [15., 16.], [17., 18.], [19., 20.],
    ])
    scaling = torch.tensor(0.5)
    aux_streams = [object(), object()]
    layer = _scope_only_layer(harness, aux_stream_list=aux_streams)
    gathers, serial_calls, localizations = [], [], []
    error = _SerialFailure("numeric serial boundary failed")

    def gather(hidden):
        assert hidden is local_hidden
        assert hidden.tolist() == [[1., 2.], [3., 4.], [9., 10.], [900., 901.]]
        gathers.append(hidden)
        return global_hidden

    def serial(positions, hidden, supplied_scaling):
        # Explicit native serial-body boundary: real wrapper, CPU arithmetic body.
        serial_calls.append((positions, hidden, supplied_scaling))
        assert layer.aux_stream_list is None
        assert hidden is global_hidden
        assert supplied_scaling is scaling
        assert positions.tolist() == [5, 8, 0, 1, 2, 3, 4, 5, 6, 7]
        assert positions.data_ptr() == batch.global_batch.positions.data_ptr()
        assert positions is not local_positions
        assert harness.context.attn_metadata is batch._hcu_v41_metadata[1]
        assert harness.context.slot_mapping is batch._hcu_v41_metadata[2]
        assert harness.context.is_padding.tolist() == [False] * 10
        assert harness.runtime._ACTIVE_ROWS.get()[0] == 10
        assert harness.depth == {"replicated": 1, "logical": 1}
        if fail:
            raise error
        return hidden + positions[:, None] + supplied_scaling

    def localize(output):
        _assert_restored(harness, saved)
        assert layer.aux_stream_list is aux_streams
        assert output.tolist() == [
            [6.5, 7.5], [11.5, 12.5], [5.5, 6.5], [8.5, 9.5], [11.5, 12.5],
            [14.5, 15.5], [17.5, 18.5], [20.5, 21.5], [23.5, 24.5], [26.5, 27.5],
        ]
        shard = torch.stack([output[0], output[1], output[4], output[0]])
        assert shard[-1].tolist() == [6.5, 7.5]
        localizations.append((output, shard))
        return shard

    batch.gather, batch.localize = gather, localize
    layer._hcu_v41_serial_forward = serial
    if fail:
        with pytest.raises(_SerialFailure) as caught:
            harness.runtime._forward(layer, local_positions, local_hidden, scaling)
        assert caught.value is error
        assert localizations == []
    else:
        output = harness.runtime._forward(layer, local_positions, local_hidden, scaling)
        assert len(localizations) == 1
        assert output is localizations[0][1]
        assert output.tolist() == [[6.5, 7.5], [11.5, 12.5], [11.5, 12.5], [0., 0.]]
    assert len(gathers) == len(serial_calls) == 1
    assert layer.aux_stream_list is aux_streams
    _assert_restored(harness, saved)
    assert batch.local_batch.positions is local_positions
    assert local_positions.tolist() == [5, 8, 2, 0]
    assert local_hidden.tolist() == [[1., 2.], [3., 4.], [9., 10.], [900., 901.]]
    assert batch.global_batch.positions.tolist() == [5, 8, 0, 1, 2, 3, 4, 5, 6, 7]


def test_forward_pure_decode_never_gathers_replicated_rows(harness):
    batch = harness.batch = _batch()
    batch.manager._global_has_prefill = False
    batch.global_batch.num_tokens = 2
    batch.global_batch.positions = torch.tensor([5, 8])
    batch.global_batch.is_padding = torch.tensor([False, False])
    batch.local_batch.num_tokens = 2
    batch.local_batch.positions = torch.tensor([5, 8, 0, 0])
    batch.local_batch.is_padding = torch.tensor([False, False, True, True])
    batch.local_indices = torch.tensor([0, 1, 0, 1])
    saved = _bind_views(harness, batch)
    local_positions = batch.local_batch.positions
    hidden = torch.tensor([[1., 2.], [3., 4.], [700., 701.], [800., 801.]])
    aux_streams = object()
    layer = _scope_only_layer(harness, aux_stream_list=aux_streams)
    calls, localizations = [], []

    def gather(_):
        pytest.fail("pure replicated decode must not issue a gather")

    def serial(positions, full_hidden, scaling):
        assert scaling is None
        assert full_hidden.shape == (2, 2)
        assert full_hidden.data_ptr() == hidden.data_ptr()
        assert full_hidden.tolist() == [[1., 2.], [3., 4.]]
        assert positions.tolist() == [5, 8]
        assert layer.aux_stream_list is None
        assert harness.runtime._ACTIVE_ROWS.get()[0] == 2
        assert harness.context.attn_metadata is batch._hcu_v41_metadata[1]
        calls.append(full_hidden)
        return full_hidden + positions[:, None]

    def localize(output):
        assert output.tolist() == [[6., 7.], [11., 12.]]
        shard = torch.stack([output[0], output[1], output[0], output[1]])
        localizations.append(shard)
        return shard

    batch.gather, batch.localize = gather, localize
    layer._hcu_v41_serial_forward = serial
    output = harness.runtime._forward(layer, local_positions, hidden)
    assert len(calls) == len(localizations) == 1
    assert output is localizations[0]
    assert output.tolist() == [[6., 7.], [11., 12.], [0., 0.], [0., 0.]]
    assert layer.aux_stream_list is aux_streams
    assert batch.local_batch.positions is local_positions
    assert local_positions.tolist() == [5, 8, 0, 0]
    _assert_restored(harness, saved)


@pytest.mark.parametrize("fail", [False, True], ids=["normal", "exception"])
def test_forward_without_batch_passes_exact_inputs_and_restores_aux_streams(harness, fail):
    saved = (harness.context.attn_metadata, harness.context.slot_mapping,
             harness.context.is_padding)
    positions, hidden, scaling = torch.tensor([4, 7]), torch.ones(2, 2), object()
    result = torch.tensor([[51., 52.], [61., 62.]])
    aux_streams = [object()]
    layer = _scope_only_layer(harness, aux_stream_list=aux_streams)
    calls = []
    error = _SerialFailure("profile serial failure")

    def serial(actual_positions, actual_hidden, actual_scaling):
        calls.append(True)
        assert actual_positions is positions
        assert actual_hidden is hidden
        assert actual_scaling is scaling
        assert layer.aux_stream_list is None
        assert harness.context.attn_metadata is saved[0]
        assert harness.context.slot_mapping is saved[1]
        assert harness.context.is_padding is saved[2]
        assert harness.runtime._ACTIVE_ROWS.get()[0] == 2
        if fail:
            raise error
        return result

    layer._hcu_v41_serial_forward = serial
    if fail:
        with pytest.raises(_SerialFailure) as caught:
            harness.runtime._forward(layer, positions, hidden, scaling)
        assert caught.value is error
    else:
        assert harness.runtime._forward(layer, positions, hidden, scaling) is result
    assert calls == [True]
    assert layer.aux_stream_list is aux_streams
    assert positions.tolist() == [4, 7]
    _assert_restored(harness, saved)


@pytest.mark.parametrize("bad_case", ["hidden-width", "position-width", "gather-count"])
def test_forward_rejects_incomplete_local_or_global_rows(harness, bad_case):
    batch = harness.batch = _batch()
    saved = _bind_views(harness, batch)
    hidden = torch.ones(3 if bad_case == "hidden-width" else 4, 2)
    positions = torch.arange(3 if bad_case == "position-width" else 4)
    gather_calls = []

    def gather(actual):
        assert actual is hidden
        gather_calls.append(actual)
        return torch.ones(9, 2)

    batch.gather = gather
    layer = _scope_only_layer(harness, aux_stream_list=object(),
               _hcu_v41_serial_forward=lambda *args: pytest.fail("invalid row count reached serial"))
    original_aux = layer.aux_stream_list
    with pytest.raises(RuntimeError, match="local fragment rows|complete global batch"):
        harness.runtime._forward(layer, positions, hidden)
    assert len(gather_calls) == (1 if bad_case == "gather-count" else 0)
    assert layer.aux_stream_list is original_aux
    _assert_restored(harness, saved)


@pytest.mark.parametrize("bad_output", ["not-tensor", "wrong-rows"])
def test_forward_rejects_invalid_native_output_after_restoring_context(harness, bad_output):
    batch = harness.batch = _batch()
    saved = _bind_views(harness, batch)
    batch.gather = lambda hidden: torch.ones(10, 2)
    batch.localize = lambda output: pytest.fail("invalid output reached localization")
    aux_streams = object()
    layer = _scope_only_layer(harness, aux_stream_list=aux_streams,
               _hcu_v41_serial_forward=lambda *args: (
                   {"output": torch.ones(10, 2)} if bad_output == "not-tensor"
                   else torch.ones(9, 2)))
    with pytest.raises(RuntimeError, match="invalid global output"):
        harness.runtime._forward(layer, batch.local_batch.positions, torch.ones(4, 2))
    assert layer.aux_stream_list is aux_streams
    _assert_restored(harness, saved)


@pytest.mark.parametrize("candidate_capacity", [None, 4, 7],
                         ids=["candidate-free", "candidate-exact", "candidate-spare"])
def test_global_indexer_guard_accepts_complete_global_rows(harness, candidate_capacity):
    _global_context(harness)
    candidates = (None if candidate_capacity is None else
                  torch.empty(candidate_capacity, 2, dtype=torch.int32))
    indexer = _indexer(candidates=candidates)
    hidden, q, weights = torch.ones(4, 3), torch.ones(4, 2, 3), torch.ones(4, 2)
    with harness.runtime._attention_scope(None, 4):
        assert harness.runtime.require_global_indexer_rows(indexer, hidden, q, None, weights) is None
    with pytest.raises(RuntimeError, match="active global attention scope"):
        harness.runtime.require_global_indexer_rows(indexer, hidden, q, None, weights)


@pytest.mark.parametrize("bad_scope", ["none", "different-metadata"])
def test_global_indexer_guard_requires_scope_bound_to_current_metadata(harness, bad_scope):
    _global_context(harness)
    args = (_indexer(), torch.ones(4, 3), torch.ones(4, 2, 3), None, torch.ones(4, 2))
    if bad_scope == "none":
        with pytest.raises(RuntimeError, match="active global attention scope"):
            harness.runtime.require_global_indexer_rows(*args)
    else:
        with harness.runtime._attention_scope(None, 4):
            harness.context.attn_metadata = {
                "indexer": NS(pcp_world_size=1, num_decode_tokens=1, num_prefill_tokens=3)
            }
            with pytest.raises(RuntimeError, match="active global attention scope"):
                harness.runtime.require_global_indexer_rows(*args)


@pytest.mark.parametrize(
    "bad_case",
    ["q-not-tensor", "q-rows", "hidden-rows", "weight-rows", "produced-k", "source-insert"],
)
def test_global_indexer_guard_rejects_misaligned_rows_or_source_cache_mode(harness, bad_case):
    _global_context(harness)
    indexer = _indexer()
    hidden, q, weights, k = torch.ones(4, 3), torch.ones(4, 2, 3), torch.ones(4, 2), None
    if bad_case == "q-not-tensor":
        q = None
    elif bad_case == "q-rows":
        q = torch.ones(3, 2, 3)
    elif bad_case == "hidden-rows":
        hidden = torch.ones(3, 3)
    elif bad_case == "weight-rows":
        weights = torch.ones(3, 2)
    elif bad_case == "produced-k":
        k = torch.ones(4, 3)
    else:
        indexer.skip_k_cache_insert = False
    with harness.runtime._attention_scope(None, 4):
        with pytest.raises(RuntimeError, match="Q/weights/source-cache rows"):
            harness.runtime.require_global_indexer_rows(indexer, hidden, q, k, weights)


@pytest.mark.parametrize(
    "pcp_width,decode_rows,prefill_rows",
    [(8, 1, 3), (1, 1, 2), (1, 2, 3)],
    ids=["physical-pcp-metadata", "short-total", "long-total"],
)
def test_global_indexer_guard_rejects_wrong_logical_width_or_total_rows(
    harness, pcp_width, decode_rows, prefill_rows,
):
    metadata = _global_context(harness)
    metadata.pcp_world_size = pcp_width
    metadata.num_decode_tokens = decode_rows
    metadata.num_prefill_tokens = prefill_rows
    with harness.runtime._attention_scope(None, 4):
        with pytest.raises(RuntimeError, match="complete logical-PCP1 metadata"):
            harness.runtime.require_global_indexer_rows(
                _indexer(), torch.ones(4, 3), torch.ones(4, 2, 3), None, torch.ones(4, 2),
            )


@pytest.mark.parametrize("limited_buffer", ["topk", "candidate"])
def test_global_indexer_guard_rejects_insufficient_shared_capacity(harness, limited_buffer):
    _global_context(harness)
    if limited_buffer == "topk":
        indexer = _indexer(rows=3)
    else:
        indexer = _indexer(candidates=torch.empty(3, 2, dtype=torch.int32))
    with harness.runtime._attention_scope(None, 4):
        with pytest.raises(RuntimeError, match=f"shared {limited_buffer} capacity"):
            harness.runtime.require_global_indexer_rows(
                indexer, torch.ones(4, 3), torch.ones(4, 2, 3), None, torch.ones(4, 2),
            )


@pytest.mark.parametrize("compress_ratio", [1, 2], ids=["CR1", "CR2"])
def test_logical_indexer_width_is_one_for_native_source_owned_cache(harness, compress_ratio):
    config = _config()
    cache = _native_cache()
    assert harness.runtime.logical_indexer_pcp_width(config, cache, True, compress_ratio) == 1
    assert config.parallel_config.prefill_context_parallel_size == 8


@pytest.mark.parametrize(
    "module,name,skip_insert,ratio",
    [
        ("vllm.models.deepseek_v41.attention", "DeepseekV4IndexerCache", False, 1),
        ("vllm.models.deepseek_v41.attention", "DeepseekV4IndexerCache", True, 0),
        ("vllm.models.deepseek_v41.attention", "DeepseekV4IndexerCache", True, 4),
        ("vllm.models.deepseek_v4.attention", "DeepseekV4IndexerCache", True, 1),
        ("tests.native_fixture", "DeepseekV4IndexerCache", True, 2),
        ("vllm.models.deepseek_v41.attention", "GenericIndexerCache", True, 1),
        ("vllm.models.deepseek_v41.attention", "DeepseekV41IndexerCache", True, 2),
    ],
    ids=["inserts-k", "CR0", "CR4", "V4-module", "foreign-module", "generic-class", "wrong-name"],
)
def test_logical_indexer_width_rejects_non_native_or_non_source_owned_cache(
    harness, module, name, skip_insert, ratio,
):
    with pytest.raises(ValueError, match="native source-owned FP8 index cache"):
        harness.runtime.logical_indexer_pcp_width(_config(), _native_cache(module, name),
                                                skip_insert, ratio)


def test_logical_indexer_width_rejects_native_cache_subclass(harness):
    native_type = type(_native_cache())
    derived_type = type("DerivedCache", (native_type,),
                        {"__module__": "vllm.models.deepseek_v41.attention"})
    with pytest.raises(ValueError, match="native source-owned FP8 index cache"):
        harness.runtime.logical_indexer_pcp_width(_config(), derived_type(), True, 2)


@pytest.mark.parametrize("architectures", [
    ["DeepseekV4ForCausalLM"], ["OtherForCausalLM"],
    ["DeepseekV41ForCausalLM", "OtherForCausalLM"],
], ids=["older-V4", "generic-model", "mixed-architectures"])
def test_generic_indexer_keeps_physical_pcp8_without_native_cache_restriction(harness, architectures):
    config = _config()
    config.model_config.architectures = architectures
    assert harness.runtime.logical_indexer_pcp_width(config, object(), False, 99) == 8
    assert config.parallel_config.prefill_context_parallel_size == 8


def test_logical_indexer_width_applies_full_target_contract(harness):
    config = _config()
    config.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="TP1/PCP8/DP1/PP1/DCP1"):
        harness.runtime.logical_indexer_pcp_width(config, _native_cache(), True, 1)


@pytest.mark.parametrize("indexer_dtype", ["auto", "fp8"])
def test_pureconfig_admits_only_eager_target_slice(harness, indexer_dtype):
    config = _config()
    config.attention_config.indexer_kv_dtype = indexer_dtype
    assert harness.contract.is_v41_pcp(config) is True
    assert harness.contract.require_v41_pcp_contract(config) is None
    assert config.parallel_config.prefill_context_parallel_size == 8


@pytest.mark.parametrize(
    "architectures,pcp,is_target_pcp",
    [
        (["DeepseekV41ForCausalLM"], 8, True),
        (["DeepseekV41ForCausalLM"], 2, True),
        (["DeepseekV41ForCausalLM"], 1, False),
        (["DeepseekV4ForCausalLM"], 8, False),
        (["DeepseekV41ForCausalLM", "DeepseekV4ForCausalLM"], 8, False),
        ([], 8, False),
    ],
    ids=["target-PCP8", "predicate-PCP2", "target-PCP1", "V4", "mixed", "no-architecture"],
)
def test_pureconfig_target_predicate_does_not_widen_other_models(
    harness, architectures, pcp, is_target_pcp,
):
    config = _config()
    config.model_config.architectures = architectures
    config.parallel_config.prefill_context_parallel_size = pcp
    assert harness.contract.is_v41_pcp(config) is is_target_pcp
    if not is_target_pcp:
        with pytest.raises(ValueError, match="native V41 target with PCP>1"):
            harness.contract.require_v41_pcp_contract(config)


@pytest.mark.parametrize(
    "path,value,error",
    [
        ("parallel_config.tensor_parallel_size", 2, "TP1/PCP8/DP1/PP1/DCP1"),
        ("parallel_config.prefill_context_parallel_size", 2, "TP1/PCP8/DP1/PP1/DCP1"),
        ("parallel_config.data_parallel_size", 2, "TP1/PCP8/DP1/PP1/DCP1"),
        ("parallel_config.pipeline_parallel_size", 2, "TP1/PCP8/DP1/PP1/DCP1"),
        ("parallel_config.decode_context_parallel_size", 2, "TP1/PCP8/DP1/PP1/DCP1"),
        ("parallel_config.cp_kv_cache_interleave_size", 2, "without interleaving"),
        ("parallel_config.enable_expert_parallel", False, "expert parallelism"),
        ("parallel_config.enable_eplb", True, "does not support EPLB"),
        ("use_v2_model_runner", False, "Model Runner V2"),
        ("model_config.use_mla", False, "native sparse MLA"),
        ("model_config.enforce_eager", False, "eager execution"),
        ("model_config.is_multimodal_model", True, "multimodal inputs or LoRA"),
        ("lora_config", NS(max_loras=1), "multimodal inputs or LoRA"),
        ("kv_transfer_config", NS(kv_connector="fixture"), "standalone caches"),
        ("cache_config.kv_offloading_size", 1, "without KV offload"),
        ("cache_config.kv_offloading_size", 0, "without KV offload"),
        ("attention_config.indexer_kv_dtype", "fp4", "FP8 indexer only"),
        ("attention_config.indexer_kv_dtype", "bf16", "FP8 indexer only"),
        ("attention_config.indexer_sparse_logits", True, "sparse-logits indexers"),
    ],
    ids=["TP2", "PCP2", "DP2", "PP2", "DCP2", "interleaving", "no-EP", "EPLB",
         "runner-v1", "no-MLA", "graphs", "multimodal", "LoRA", "KV-transfer",
         "offload", "configured-zero-offload", "FP4", "BF16", "sparse-logits"],
)
def test_pureconfig_rejects_unsupported_topology_and_features(harness, path, value, error):
    config = _config()
    _set_config(config, path, value)
    with pytest.raises(ValueError, match=error):
        harness.contract.require_v41_pcp_contract(config)


@pytest.mark.parametrize("speculative", [
    NS(method="dspark", num_speculative_tokens=5, use_dspark=lambda: True),
    NS(method="dspark", num_speculative_tokens=1, use_dspark=lambda: True),
    NS(method="mtp", num_speculative_tokens=1),
    NS(method="ngram", num_speculative_tokens=5),
    NS(),
], ids=["DSpark5", "DSpark1", "MTP", "ngram", "non-null-empty"])
def test_pureconfig_rejects_every_speculative_config_including_dspark5(harness, speculative):
    config = _config()
    config.speculative_config = speculative
    with pytest.raises(ValueError, match="speculative execution requires the separate draft/group integration"):
        harness.contract.require_v41_pcp_contract(config)


def test_install_backend_is_per_instance_idempotent_and_preserves_base_class(harness):
    class BaseBackend:
        sentinel = object()

        @classmethod
        def supports_pcp(cls):
            return False

        @classmethod
        def name(cls):
            return "fixture-native-backend"

    class Cache:
        def get_attn_backend(self):
            return BaseBackend

    cache, peer, untouched = Cache(), Cache(), Cache()
    original_method = Cache.get_attn_backend
    harness.runtime._install_cache_backend(cache)
    installed_method = cache.get_attn_backend
    saved_method = cache._hcu_v41_original_backend
    adapted = cache.get_attn_backend()
    assert saved_method.__self__ is cache
    assert saved_method.__func__ is original_method
    assert saved_method() is BaseBackend
    assert installed_method.__self__ is cache
    assert adapted is not BaseBackend
    assert issubclass(adapted, BaseBackend)
    assert adapted.__bases__ == (BaseBackend,)
    assert adapted.supports_pcp() is True
    assert adapted.sentinel is BaseBackend.sentinel
    assert adapted.name() == "fixture-native-backend"
    assert BaseBackend.supports_pcp() is False
    assert Cache.get_attn_backend is original_method
    assert peer.get_attn_backend() is untouched.get_attn_backend() is BaseBackend
    assert "get_attn_backend" not in vars(peer)
    harness.runtime._install_cache_backend(cache)
    assert cache.get_attn_backend is installed_method
    assert cache._hcu_v41_original_backend is saved_method
    harness.runtime._install_cache_backend(peer)
    assert peer.get_attn_backend() is adapted
    assert peer.get_attn_backend.__self__ is peer
    assert untouched.get_attn_backend() is BaseBackend
    assert "_hcu_v41_original_backend" not in vars(untouched)


def test_fragment_replay_zero_capacity_grows_only_local_builder(harness):
    builder = harness.swa_builder_cls()
    old = builder.no_replay_start = torch.zeros(5, dtype=torch.int32)
    group = NS(get_metadata_builder=lambda index: builder)
    nonswa = NS(get_metadata_builder=lambda index: object())
    filtered = harness.runtime._local_swa_groups([[group], [nonswa]], 0)
    assert filtered == [[group], []]
    harness.runtime._ensure_fragment_replay_capacity(filtered, 12, 0)
    assert builder.no_replay_start.shape == (12,)
    assert builder.no_replay_start.dtype == torch.int32
    assert builder.no_replay_start.tolist() == [0] * 12
    assert builder.no_replay_start is not old and old.tolist() == [0] * 5
    current = builder.no_replay_start
    harness.runtime._ensure_fragment_replay_capacity(filtered, 3, 0)
    assert builder.no_replay_start is current


def test_fragment_replay_zero_capacity_rejects_invalid_dtype(harness):
    builder = harness.swa_builder_cls()
    builder.no_replay_start = torch.zeros(5, dtype=torch.float32)
    group = NS(get_metadata_builder=lambda index: builder)
    with pytest.raises(RuntimeError, match="int32 default replay vector"):
        harness.runtime._ensure_fragment_replay_capacity([[group]], 12, 0)


def test_no_local_swa_group_is_rejected(harness):
    group = NS(get_metadata_builder=lambda index: object())
    with pytest.raises(RuntimeError, match="local SWA metadata for Engram"):
        harness.runtime._local_swa_groups([[group]], 0)


def test_bounded_replay_cannot_enter_local_fragment_build(harness):
    batch = harness.batch = _batch()
    calls = []
    def original(*args):
        calls.append(args)
        return {"native": torch.tensor([11])}
    state = NS(_hcu_v41_original_prepare_attn=original, _replay=(128, None))
    with pytest.raises(RuntimeError, match="bounded replay requires a separate fragment adapter"):
        harness.runtime._prepare_attn(state, batch.local_batch, object(), object(),
                                    object(), [], object())
    assert len(calls) == 1 and calls[0][0] is batch.global_batch
    assert not hasattr(batch, "_hcu_v41_metadata")


def test_low_latency_backend_cannot_use_target_pcp_contract(harness):
    config = _config()
    config.parallel_config.all2all_backend = "deepep_low_latency"
    with pytest.raises(ValueError, match="explicit DeepEP HT"):
        harness.contract.require_v41_pcp_contract(config)


@pytest.mark.parametrize("length,dtype,nonzero", [(2, torch.int32, False), (4, torch.int64, False), (4, torch.int32, True)])
def test_caller_replay_input_must_cover_fragments_and_stay_zero(harness, length, dtype, nonzero):
    batch = harness.batch = _batch()
    batch.global_batch.num_reqs = 3
    starts = torch.zeros(length, dtype=dtype)
    if nonzero:
        starts[0] = 1
    calls = []
    state = NS(_hcu_v41_original_prepare_attn=lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="fragment capacity|bounded replay"):
        harness.runtime._prepare_attn(state, batch.local_batch, object(), object(),
            object(), [], object(), model_specific_attn_metadata=NS(replay_start=starts))
    assert calls == []


def test_valid_zero_caller_replay_metadata_identity_survives_both_builds(harness):
    batch = harness.batch = _batch()
    batch.global_batch.num_reqs = 3
    starts = torch.zeros(4, dtype=torch.int32)
    specific = NS(replay_start=starts)
    calls = []
    def original(*args):
        calls.append(args)
        return {"specific": args[-1], "positions": args[0].positions}
    builder = harness.swa_builder_cls()
    builder.no_replay_start = torch.zeros(2, dtype=torch.int32)
    groups = [[NS(get_metadata_builder=lambda index: builder)]]
    state = NS(_hcu_v41_original_prepare_attn=original, _replay=(0, None))
    result = harness.runtime._prepare_attn(state, batch.local_batch, object(), object(),
        object(), groups, object(), model_specific_attn_metadata=specific)
    assert len(calls) == 2 and all(call[-1] is specific for call in calls)
    assert result["specific"] is batch._hcu_v41_metadata[1]["specific"] is specific
    assert specific.replay_start is starts and starts.tolist() == [0, 0, 0, 0]


def test_scope_enter_error_restores_original_context_objects(harness):
    batch = _batch()
    local, global_metadata, global_slots = {"local": object()}, {"global": object()}, {"slot": object()}
    batch._hcu_v41_metadata = (local, global_metadata, global_slots)
    harness.context.attn_metadata = local
    saved = (harness.context.attn_metadata, harness.context.slot_mapping, harness.context.is_padding)
    batch.global_batch.is_padding = None
    with pytest.raises(TypeError):
        with harness.runtime._attention_scope(batch, 10):
            pytest.fail("invalid global padding must fail before native execution")
    assert harness.context.attn_metadata is saved[0]
    assert harness.context.slot_mapping is saved[1]
    assert harness.context.is_padding is saved[2]
    assert harness.runtime._ACTIVE_ROWS.get() is None
