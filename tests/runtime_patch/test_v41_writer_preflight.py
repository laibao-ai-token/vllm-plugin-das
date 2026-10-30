# SPDX-License-Identifier: Apache-2.0
"""Portable CPU writer-order tests for the adjacent R3 V41 adapter.

The actual runtime module and pure configuration module are loaded from this
file's overlay root. Only import boundaries (configuration accessors, PCP scopes,
current batch, and forward context) are replaced. In particular, _forward,
_attention_scope, and _preflight_attention execute their adjacent source bodies.

FakeSerialBody is an explicitly TOY serial writer, not native attention, native
CR1/CR2 branch coverage, numerical parity, or device-kernel integration. Its
source/consumer objects follow the native ownership graph: model.layers.20 owns
main/index caches, model.layers.24 consumes them; only owners have compressors,
and only CR2 compressors have state caches. CPU cache records retain block and
slot capacity but their toy writes touch just one payload element. LONG/SHORT
name toy input cases, not a claim about native indexer branch selection. Native
full-body/source integration is a separate probe maintained by the Lead.
"""
from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_OWNER = "model.layers.20.self_attn"
_CONSUMER = "model.layers.24.self_attn"


def _boundary(monkeypatch, name, **attributes):
    """Install every parent stub so no installed package __init__ is imported."""
    parent = None
    parts = name.split(".")
    for count in range(1, len(parts) + 1):
        current = ".".join(parts[:count])
        module = sys.modules.get(current)
        if module is None or not getattr(module, "_writer_cpu_boundary", False):
            module = ModuleType(current)
            module.__path__ = []
            module._writer_cpu_boundary = True
            monkeypatch.setitem(sys.modules, current, module)
        if parent is not None:
            setattr(parent, parts[count - 1], module)
        parent = module
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _source(monkeypatch, name, relative):
    path = _SOURCE_ROOT / relative
    assert path.is_file(), f"missing adjacent R3 source: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _config():
    return NS(
        model_config=NS(architectures=["DeepseekV41ForCausalLM"], use_mla=True,
                        enforce_eager=True, is_multimodal_model=False),
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


class ScopeHarness:
    """Dependency-inverted loading of the two actual adjacent source modules."""

    def __init__(self, monkeypatch):
        self.batch = None
        self.context = NS(attn_metadata=None, slot_mapping=None, is_padding=None)
        self.depth = {"replicated": 0, "logical": 0}
        self.events = []
        self.on_logical_enter = None
        _boundary(
            monkeypatch, "vllm_hcu.deepseek_v4_runtime",
            config_field=lambda value, name, default=None: getattr(value, name, default),
            model_architectures=lambda config: tuple(config.model_config.architectures),
        )
        _boundary(monkeypatch, "vllm.forward_context",
                  get_forward_context=lambda: self.context)
        _boundary(monkeypatch, "vllm_hcu.v1.glm53_pcp",
                  _current_pcp_batch=lambda: self.batch)
        _boundary(
            monkeypatch, "vllm_hcu.model_executor.layers.attention.pcp",
            replicated_mtp_batch_scope=lambda: self.scope("replicated"),
            logical_pcp_metadata_scope=lambda width: self.scope("logical", width),
        )
        private = f"_v41_writer_cpu_{id(self)}"
        _boundary(monkeypatch, private)
        self.contract = _source(monkeypatch, private + ".pureconfig",
                                "vllm_hcu/runtime_compat/deepseek_v41_pcp.py")
        _boundary(monkeypatch, "vllm_hcu.runtime_compat")
        monkeypatch.setitem(sys.modules, "vllm_hcu.runtime_compat.deepseek_v41_pcp",
                            self.contract)
        self.runtime = _source(monkeypatch, private + ".runtime",
                               "vllm_hcu/v1/deepseek_v41_pcp.py")
        self.original_preflight = self.runtime._preflight_attention
        self.config = _config()
        assert self.contract.is_v41_pcp(self.config)
        assert self.contract.require_v41_pcp_contract(self.config) is None

    @contextmanager
    def scope(self, name, width=None):
        self.events.append(("enter", name, width))
        self.depth[name] += 1
        try:
            if name == "logical":
                assert width == 1
                if self.on_logical_enter is not None:
                    self.on_logical_enter()
            yield
        finally:
            self.depth[name] -= 1
            self.events.append(("exit", name, width))


@pytest.fixture
def harness(monkeypatch):
    env = ScopeHarness(monkeypatch)
    yield env
    assert env.runtime._preflight_attention is env.original_preflight
    assert env.runtime._ACTIVE_ROWS.get() is None
    assert env.depth == {"replicated": 0, "logical": 0}


# Literal input/output and writer-row oracles. Expectations are independent of
# preflight predicates; no expected value is derived by executing those checks.
_ORACLE = {
    "LONG": NS(
        hidden=[[1., 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.], [11., 12.]],
        positions=[5, 8, 0, 1, 2, 3], qsl=[0, 1, 2, 6], seq=[6, 9, 4],
        local_positions=[5, 8, 2, 0], local_indices=[0, 1, 4, 0],
        local_hidden=[[1., 2.], [3., 4.], [9., 10.], [700., 701.]],
        padding=[False, False,False, True], requests=3, rows=6, prefill=4,
        swa_slots=[1, 2, 8, 9, 10, 11], swa_values=[11, 12, 13, 14, 15, 16],
        main_slots={1: [3, 4, 16, 17, 18, 19], 2: [3, -1, -1, 16, -1, 17]},
        main_written={1: [3, 4, 16, 17, 18, 19], 2: [3, 16, 17]},
        main_values={1: [21, 22, 23, 24, 25, 26], 2: [21, 24, 26]},
        index_values={1: [41, 42, 43, 44, 45, 46], 2: [41, 44, 46]},
        state_slots=[0, 1, 2, 2, 2, 2], state_written=[0, 1, 2],
        state_values=[31., 32., 36.],
        topk=[[101, 201], [102, 202], [103, 203], [104, 204], [105, 205], [106, 206]],
        candidates=[[51, 61], [52, 62], [53, 63], [54, 64], [55, 65], [56, 66]],
        output=[[6.5, 7.5], [11.5, 12.5], [11.5, 12.5], [0., 0.]],
    ),
    "SHORT": NS(
        hidden=[[1., 2.], [3., 4.]], positions=[5, 8], qsl=[0, 1, 2], seq=[6, 9],
        local_positions=[5, 8, 0, 0], local_indices=[0, 1, 0, 1],
        local_hidden=[[1., 2.], [3., 4.], [700., 701.], [800., 801.]],
        padding=[False, False, True, True], requests=2, rows=2, prefill=0,
        swa_slots=[1, 2], swa_values=[11, 12],
        main_slots={1: [3, 4], 2: [3, -1]},
        main_written={1: [3, 4], 2: [3]},
        main_values={1: [21, 22], 2: [21]}, index_values={1: [41, 42], 2: [41]},
        state_slots=[0, 1], state_written=[0, 1], state_values=[31., 32.],
        topk=[[101, 201], [102, 202]], candidates=[[51, 61], [52, 62]],
        output=[[6.5, 7.5], [11.5, 12.5], [0., 0.], [0., 0.]],
    ),
}
_CASES = [(length, ratio, owner) for length in ("LONG", "SHORT")
          for ratio in (1, 2) for owner in (True, False)]
_CASE_IDS = [f"{length}-CR{ratio}-{'owner20' if owner else 'consumer24'}"
             for length, ratio, owner in _CASES]


def _cache(prefix, fill, width=584):
    # Two native-sized 32-token blocks. Payload arithmetic is intentionally toy.
    return NS(prefix=prefix, kv_cache=torch.full((2, 32, width), fill, dtype=torch.uint8))


def _entry(slots, oracle, **fields):
    result = NS(slot_mapping=torch.tensor(slots, dtype=torch.int64),
                pcp_world_size=1, num_actual_tokens=oracle.rows)
    vars(result).update(fields)
    return result


def _fixture(env, length="LONG", ratio=2, owner=True):
    oracle = _ORACLE[length]
    source = NS(prefix=_OWNER, is_kv_source=True,
                kv_cache=_cache(_OWNER, 72).kv_cache)
    layer = source if owner else NS(prefix=_CONSUMER, is_kv_source=False,
                                    kv_cache=torch.empty(0, dtype=torch.uint8))
    index_cache = _cache(_OWNER + ".indexer.k_cache", 73, width=136)
    swa = _cache(layer.prefix + ".swa_cache", 71)
    static = {_OWNER: source, index_cache.prefix: index_cache, swa.prefix: swa}
    static[layer.prefix] = layer
    state = None
    if owner and ratio == 2:
        state = NS(prefix=_OWNER + ".compressor.state_cache",
                   kv_cache=torch.full((8, 1024), 74., dtype=torch.float32))
        static[state.prefix] = state
    compressor = (NS(k_cache_prefix=_OWNER, _static_forward_context=static,
                     state_cache=state) if owner else None)
    topk = torch.full((8, 2), -91, dtype=torch.int32)
    candidates = torch.full((8, 2), -92, dtype=torch.int32)
    op = NS(k_cache=index_cache, skip_k_cache_insert=True,
            topk_indices_buffer=topk, candidate_blocks=candidates)
    indexer = NS(owns_k=owner, k_cache=index_cache, topk_tokens=2,
                 topk_indices_buffer=topk, indexer_op=op)
    vars(layer).update(
        compress_ratio=ratio, kv_source_layer_id=20, is_index_source=True,
        compressed_cache_prefix=_OWNER, compressor=compressor, indexer=indexer,
        swa_cache_layer=swa, topk_indices_buffer=topk,
        candidate_block_buffer=candidates, candidate_topk_blocks=2,
        _static_forward_context=static, _hcu_v41_topk_tokens=2,
        _hcu_v41_shared_topk_buffer=topk, _hcu_v41_shared_candidate_buffer=candidates,
        _hcu_v41_main_cache_prefix=_OWNER,
        _hcu_v41_index_cache_prefix=index_cache.prefix, aux_stream_list=[object(), object()],
    )
    qsl = torch.tensor(oracle.qsl, dtype=torch.int32)
    metadata = {
        swa.prefix: _entry(
            oracle.swa_slots, oracle, num_decodes=2,
            num_prefills=1 if length == "LONG" else 0,
            num_decode_tokens=2, num_prefill_tokens=oracle.prefill,
            query_start_loc=qsl, seq_lens=torch.tensor(oracle.seq, dtype=torch.int32),
            block_table=torch.tensor([[0, 1]] * oracle.requests, dtype=torch.int32)),
        _OWNER: _entry(oracle.main_slots[ratio], oracle, num_reqs=oracle.requests,
                       query_start_loc=qsl),
        index_cache.prefix: _entry(
            oracle.main_slots[ratio], oracle, num_decode_tokens=2,
            num_prefill_tokens=oracle.prefill,
            max_seq_len=8192 if length == "LONG" else 16),
    }
    if state is not None:
        metadata[state.prefix] = _entry(
            oracle.state_slots, oracle, query_start_loc=qsl,
            token_to_req_indices=torch.tensor(oracle.state_slots, dtype=torch.int32))
    local_metadata, local_slots = {"local-step": object()}, {"local-slots": object()}
    local_padding = torch.tensor(oracle.padding)
    global_slots = {prefix: md.slot_mapping for prefix, md in metadata.items()}
    batch = NS(
        local_batch=NS(num_tokens=3 if length == "LONG" else 2,
                       num_tokens_after_padding=4, positions=torch.tensor(oracle.local_positions),
                       is_padding=local_padding),
        global_batch=NS(num_tokens=oracle.rows, positions=torch.tensor(oracle.positions),
                        is_padding=torch.zeros(oracle.rows, dtype=torch.bool)),
        manager=NS(_global_has_prefill=length == "LONG"),
        _hcu_v41_metadata=(local_metadata, metadata, global_slots),
    )
    env.batch = batch
    env.context.attn_metadata, env.context.slot_mapping = local_metadata, local_slots
    env.context.is_padding = local_padding
    fixture = NS(
        env=env, oracle=oracle, length=length, ratio=ratio, owner=owner,
        layer=layer, source=source, index_cache=index_cache, state=state,
        metadata=metadata, batch=batch, topk=topk, candidates=candidates,
        hidden=torch.tensor(oracle.local_hidden), scaling=torch.tensor(0.5),
        saved=(local_metadata, local_slots, local_padding), aux=layer.aux_stream_list,
        gathers=[], localizations=[], serial_calls=[], writer_calls=[],
    )
    fixture.original_buffers = _buffers(fixture)

    def gather(hidden):
        assert length == "LONG", "replicated SHORT decode must never gather"
        assert hidden is fixture.hidden
        fixture.gathers.append(hidden)
        return torch.tensor(oracle.hidden)

    def localize(output):
        _restored(fixture)
        fixture.localizations.append(output)
        return output.index_select(0, torch.tensor(oracle.local_indices))

    batch.gather, batch.localize = gather, localize
    fixture.body = FakeSerialBody(fixture)
    layer._hcu_v41_serial_forward = fixture.body
    return fixture


def _buffers(fixture):
    layer = fixture.layer
    values = [layer.swa_cache_layer.kv_cache, fixture.source.kv_cache,
              fixture.index_cache.kv_cache, fixture.topk, fixture.candidates,
              layer.topk_indices_buffer, layer.candidate_block_buffer,
              layer.indexer.topk_indices_buffer, layer.indexer.indexer_op.topk_indices_buffer,
              layer.indexer.indexer_op.candidate_blocks]
    if fixture.state is not None:
        values.append(fixture.state.kv_cache)
    unique = {}
    for value in values:
        if isinstance(value, torch.Tensor):
            unique[id(value)] = value
    return list(unique.values())


def _snapshot(fixture):
    values = {id(value): value for value in fixture.original_buffers + _buffers(fixture)}
    # Meta-device tensors are schema-only invalid inputs, never writer storage.
    return [(value, value.clone()) for value in values.values() if value.device.type == "cpu"]


def _unchanged(snapshot):
    for actual, before in snapshot:
        assert torch.equal(actual, before), "a writer buffer changed before serial admission"


def _restored(fixture):
    env = fixture.env
    assert env.context.attn_metadata is fixture.saved[0]
    assert env.context.slot_mapping is fixture.saved[1]
    assert env.context.is_padding is fixture.saved[2]
    assert env.runtime._ACTIVE_ROWS.get() is None
    assert env.depth == {"replicated": 0, "logical": 0}
    assert fixture.layer.aux_stream_list is fixture.aux
    assert env.runtime._preflight_attention is env.original_preflight


class ToyBackendFailure(RuntimeError):
    pass


class FakeSerialBody:
    """Spy with toy writes; called only after the actual runtime's preflight."""

    def __init__(self, fixture):
        self.fixture = fixture
        self.fail_after_write = False

    def _cache_write(self, name, cache, slots, start):
        records = cache.reshape(-1, cache.shape[-1])
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0:
                records[slot, 0] = start + row
        self.fixture.writer_calls.append(name)

    def __call__(self, positions, hidden, scaling):
        f = self.fixture
        env, layer, oracle = f.env, f.layer, f.oracle
        f.serial_calls.append((positions, hidden, scaling))
        assert layer.aux_stream_list is None
        assert positions.tolist() == oracle.positions
        assert hidden.tolist() == oracle.hidden
        assert scaling is f.scaling
        active = env.runtime._ACTIVE_ROWS.get()
        assert active[0] == oracle.rows and active[1] is f.metadata
        assert env.context.attn_metadata is f.metadata
        assert env.context.slot_mapping is f.batch._hcu_v41_metadata[2]
        assert env.context.is_padding.tolist() == [False] * oracle.rows
        assert env.depth == {"replicated": 1, "logical": 1}
        self._cache_write("toy_swa", layer.swa_cache_layer.kv_cache,
                          f.metadata[layer.swa_cache_layer.prefix].slot_mapping, 11)
        if f.owner:
            self._cache_write("toy_main", f.source.kv_cache,
                              f.metadata[_OWNER].slot_mapping, 21)
            self._cache_write("toy_index", f.index_cache.kv_cache,
                              f.metadata[f.index_cache.prefix].slot_mapping, 41)
            if f.state is not None:
                self._cache_write("toy_state", f.state.kv_cache,
                                  f.metadata[f.state.prefix].slot_mapping, 31)
        row_ids = torch.arange(hidden.shape[0], dtype=torch.int32)
        layer.topk_indices_buffer[:hidden.shape[0]] = torch.stack((row_ids + 101, row_ids + 201), dim=1)
        f.writer_calls.append("toy_topk")
        layer.candidate_block_buffer[:hidden.shape[0]] = torch.stack((row_ids + 51, row_ids + 61), dim=1)
        f.writer_calls.append("toy_candidates")
        if self.fail_after_write:
            raise ToyBackendFailure("injected toy backend failure after cache/topk writes")
        return hidden + positions[:, None] + scaling


def _assert_writes(f):
    oracle = f.oracle
    expected_swa = torch.full_like(f.layer.swa_cache_layer.kv_cache, 71)
    expected_swa.reshape(-1, 584)[oracle.swa_slots, 0] = torch.tensor(oracle.swa_values, dtype=torch.uint8)
    assert torch.equal(f.layer.swa_cache_layer.kv_cache, expected_swa)
    expected_main = torch.full_like(f.source.kv_cache, 72)
    expected_index = torch.full_like(f.index_cache.kv_cache, 73)
    if f.owner:
        expected_main.reshape(-1, 584)[oracle.main_written[f.ratio], 0] = torch.tensor(
            oracle.main_values[f.ratio], dtype=torch.uint8)
        expected_index.reshape(-1, 136)[oracle.main_written[f.ratio], 0] = torch.tensor(
            oracle.index_values[f.ratio], dtype=torch.uint8)
    assert torch.equal(f.source.kv_cache, expected_main)
    assert torch.equal(f.index_cache.kv_cache, expected_index)
    if f.state is not None:
        expected_state = torch.full_like(f.state.kv_cache, 74.)
        expected_state[oracle.state_written, 0] = torch.tensor(oracle.state_values)
        assert torch.equal(f.state.kv_cache, expected_state)
    expected_topk, expected_candidates = torch.full_like(f.topk, -91), torch.full_like(f.candidates, -92)
    expected_topk[:oracle.rows] = torch.tensor(oracle.topk, dtype=torch.int32)
    expected_candidates[:oracle.rows] = torch.tensor(oracle.candidates, dtype=torch.int32)
    assert torch.equal(f.topk, expected_topk)
    assert torch.equal(f.candidates, expected_candidates)
    expected_calls = ["toy_swa"]
    if f.owner:
        expected_calls += ["toy_main", "toy_index"]
        if f.ratio == 2:
            expected_calls += ["toy_state"]
    expected_calls += ["toy_topk", "toy_candidates"]
    assert f.writer_calls == expected_calls


@pytest.mark.parametrize("length,ratio,owner", _CASES, ids=_CASE_IDS)
def test_actual_forward_admits_explicit_toy_serial_writes(harness, length, ratio, owner):
    f = _fixture(harness, length, ratio, owner)
    result = harness.runtime._forward(f.layer, f.batch.local_batch.positions, f.hidden, f.scaling)
    assert result.tolist() == f.oracle.output
    assert len(f.serial_calls) == len(f.localizations) == 1
    assert len(f.gathers) == (1 if length == "LONG" else 0)
    _assert_writes(f)
    _restored(f)


@pytest.mark.parametrize("length,ratio,owner", _CASES, ids=_CASE_IDS)
def test_post_write_toy_backend_failure_is_nontransactional_and_restores_scope(
    harness, length, ratio, owner,
):
    f = _fixture(harness, length, ratio, owner)
    f.body.fail_after_write = True
    with pytest.raises(ToyBackendFailure, match="after cache/topk writes"):
        harness.runtime._forward(f.layer, f.batch.local_batch.positions, f.hidden, f.scaling)
    assert len(f.serial_calls) == 1
    assert len(f.gathers) == (1 if length == "LONG" else 0)
    assert f.localizations == []
    _assert_writes(f)  # Completed writes survive; runtime promises no rollback.
    _restored(f)


def _bind_matrix(f, kind, value, *, repin):
    layer = f.layer
    if kind == "topk":
        layer.topk_indices_buffer = layer.indexer.topk_indices_buffer = value
        layer.indexer.indexer_op.topk_indices_buffer = value
        if repin:
            layer._hcu_v41_shared_topk_buffer = value
    else:
        layer.candidate_block_buffer = layer.indexer.indexer_op.candidate_blocks = value
        if repin:
            layer._hcu_v41_shared_candidate_buffer = value


_BAD_CASES = [
    "topk-rows", "topk-columns", "topk-dtype", "topk-rank", "topk-device",
    "candidate-rows", "candidate-columns", "candidate-dtype", "candidate-device",
    "coordinated-topk-replacement", "coordinated-candidate-replacement",
    "indexer-topk-pointer", "op-topk-pointer", "op-candidate-pointer", "op-cache-pointer",
    "index-source-pointer", "owns-k-flipped-owner", "owns-k-flipped-consumer",
    "main-source-pointer", "main-prefix", "compressor-key", "op-insert-enabled",
    "main-count-absent", "swa-count-absent", "index-count-absent",
    "swa-slots-short", "main-slots-short", "index-slots-short", "state-slots-short",
    "physical-world", "tokens-mismatch", "swa-qsl-short", "main-qsl-dtype",
    "state-qsl-short", "state-map-dtype", "swa-sequence-short", "swa-block-table-short",
    "active-context-stale", "positions-dtype", "state-cache-empty",
]


def _mutate(f, name):
    layer, metadata, rows = f.layer, f.metadata, f.oracle.rows
    swa, main = metadata[layer.swa_cache_layer.prefix], metadata[_OWNER]
    index = metadata[f.index_cache.prefix]
    if name.startswith("topk-") or name.startswith("candidate-"):
        kind, fault = name.split("-", 1)
        shape = (rows - 1, 2) if fault == "rows" else (8, 1) if fault == "columns" else (8, 2)
        if fault == "rank":
            shape = (16,)
        dtype = torch.int64 if fault == "dtype" else torch.int32
        value = torch.zeros(shape, dtype=dtype, device="meta" if fault == "device" else "cpu")
        # Model installation with insufficient schema to reach capacity checks;
        # all ownership references agree, including the installed pin.
        _bind_matrix(f, kind, value, repin=True)
    elif name.startswith("coordinated-"):
        kind = "topk" if "topk" in name else "candidate"
        _bind_matrix(f, kind, torch.zeros(8, 2, dtype=torch.int32), repin=False)
    elif name == "indexer-topk-pointer":
        layer.indexer.topk_indices_buffer = f.topk.clone()
    elif name == "op-topk-pointer":
        layer.indexer.indexer_op.topk_indices_buffer = f.topk.clone()
    elif name == "op-candidate-pointer":
        layer.indexer.indexer_op.candidate_blocks = f.candidates.clone()
    elif name == "op-cache-pointer":
        layer.indexer.indexer_op.k_cache = NS(prefix=f.index_cache.prefix, kv_cache=f.index_cache.kv_cache)
    elif name == "index-source-pointer":
        replacement = NS(prefix=f.index_cache.prefix, kv_cache=f.index_cache.kv_cache)
        layer.indexer.k_cache = layer.indexer.indexer_op.k_cache = replacement
    elif name.startswith("owns-k-flipped-"):
        layer.indexer.owns_k = not f.owner
    elif name == "main-source-pointer":
        layer._static_forward_context[_OWNER] = NS(
            prefix=_OWNER, is_kv_source=True, kv_cache=f.source.kv_cache)
    elif name == "main-prefix":
        layer.compressed_cache_prefix = _CONSUMER
    elif name == "compressor-key":
        layer.compressor.k_cache_prefix = _CONSUMER
    elif name == "op-insert-enabled":
        layer.indexer.indexer_op.skip_k_cache_insert = False
    elif name == "main-count-absent":
        del main.num_actual_tokens
    elif name == "swa-count-absent":
        del swa.num_decodes
    elif name == "index-count-absent":
        del index.num_prefill_tokens
    elif name.endswith("-slots-short"):
        target = {"swa": swa, "main": main, "index": index,
                  "state": metadata[f.state.prefix]}[name.split("-")[0]]
        target.slot_mapping = torch.zeros(rows - 1, dtype=torch.int64)
    elif name == "physical-world":
        index.pcp_world_size = 8
    elif name == "tokens-mismatch":
        main.num_actual_tokens = rows - 1
    elif name == "swa-qsl-short":
        swa.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    elif name == "main-qsl-dtype":
        main.query_start_loc = torch.tensor([0., 1., 2., 6.])
    elif name == "state-qsl-short":
        metadata[f.state.prefix].query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    elif name == "state-map-dtype":
        metadata[f.state.prefix].token_to_req_indices = torch.zeros(rows, dtype=torch.float32)
    elif name == "swa-sequence-short":
        swa.seq_lens = torch.tensor([6, 9], dtype=torch.int32)
    elif name == "swa-block-table-short":
        swa.block_table = torch.zeros(2, 2, dtype=torch.int32)
    elif name == "active-context-stale":
        # Switch the view AFTER actual _attention_scope binds _ACTIVE_ROWS,
        # through the explicit PCP-scope dependency boundary, not a preflight stub.
        f.env.on_logical_enter = lambda: setattr(f.env.context, "attn_metadata", dict(metadata))
    elif name == "positions-dtype":
        f.batch.global_batch.positions = f.batch.global_batch.positions.to(torch.int32)
    elif name == "state-cache-empty":
        f.state.kv_cache = torch.empty(0, dtype=torch.float32)
    else:
        raise AssertionError(f"unhandled mutation: {name}")


@pytest.mark.parametrize("bad_case", _BAD_CASES)
def test_actual_forward_rejects_structural_fault_before_any_toy_writer(harness, bad_case):
    f = _fixture(harness, owner=bad_case != "owns-k-flipped-consumer")
    _mutate(f, bad_case)
    before = _snapshot(f)
    with pytest.raises(RuntimeError, match="V41"):
        harness.runtime._forward(f.layer, f.batch.local_batch.positions, f.hidden, f.scaling)
    assert f.serial_calls == []
    assert f.writer_calls == []
    assert f.localizations == []
    assert len(f.gathers) == 1
    _unchanged(before)
    _restored(f)
