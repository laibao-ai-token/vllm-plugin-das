# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU source/semantic SWA contracts; no vLLM import or GPU kernel launch.

Framework bases and launch scheduling are explicit substitutes. The SWA and
prefill Triton function bodies themselves execute against a NumPy pointer
model, including masks and paged address arithmetic. This is not Triton JIT.
"""
import ast
import dataclasses
import inspect
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from typing import Any, ClassVar, cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(os.environ.get("VLLM_HCU_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
REL = "vllm_hcu/v1/attention/backends/mla/sparse_swa.py"


def test_warmup_decorator_uses_current_export_module():
    tree = ast.parse((ROOT / REL).read_text())
    origins = {alias.name: node.module for node in tree.body if isinstance(node, ast.ImportFrom)
               for alias in node.names}
    assert origins["kernel_launcher"] == "vllm.model_executor.warmup.jit_warmup"
    assert origins["VllmTritonJitKernel"] == "vllm.model_executor.warmup.jit_warmup_triton_helper"


class Pointer:
    def __init__(self, array, offsets=0):
        self.array = array.detach().numpy().reshape(-1) if isinstance(array, torch.Tensor) else array
        self.offsets = np.asarray(offsets)

    def __add__(self, offsets):
        return Pointer(self.array, self.offsets + offsets)

    __radd__ = __add__


class TensorLanguage:
    constexpr = int
    pid = 0
    arange = staticmethod(np.arange)
    minimum = staticmethod(np.minimum)
    maximum = staticmethod(np.maximum)
    where = staticmethod(np.where)

    def program_id(self, axis):
        assert axis == 0
        return self.pid

    def load(self, ptr, mask=True, other=0):
        offsets, mask = np.broadcast_arrays(ptr.offsets, mask)
        assert np.all((offsets[mask] >= 0) & (offsets[mask] < len(ptr.array)))
        return np.where(mask, ptr.array[offsets.clip(0, len(ptr.array) - 1)], other)

    def store(self, ptr, values, mask=True):
        offsets, values, mask = np.broadcast_arrays(ptr.offsets, values, mask)
        assert np.all((offsets[mask] >= 0) & (offsets[mask] < len(ptr.array)))
        ptr.array[offsets[mask]] = values[mask]


class KernelBase:
    @classmethod
    def __class_getitem__(cls, item):
        return cls


class FrameworkBuilder:
    def __init__(self, kv_cache_spec, vllm_config, device="cpu"):
        self.kv_cache_spec, self.vllm_config, self.device = kv_cache_spec, vllm_config, device


class Spec(NS):
    pass


def _decode_width(config):
    spec = config.speculative_config
    return 1 if spec is None else 1 + spec.num_speculative_tokens * (2 if spec.parallel_drafting and not config.use_v2_model_runner else 1)


def _config(ratios=(0, 1, 2), spec=None):
    return NS(model_config=NS(max_model_len=256, hf_config=NS(sliding_window=4, compress_ratios=ratios)),
              scheduler_config=NS(max_num_batched_tokens=32, max_num_seqs=4),
              compilation_config=NS(static_forward_context={}),
              speculative_config=spec, use_v2_model_runner=True)


class Launch:
    """Validate the actual launcher signature, then execute its actual body."""
    def __init__(self, wrapped, tl):
        self.wrapped, self.tl, self.calls = wrapped, tl, []

    def __call__(self, *args, **kwargs):
        bound = inspect.signature(self.wrapped.__call__).bind(*args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        self.calls.append(dict(p))
        grid, launch = self.wrapped.__call__(*args, **kwargs)
        kernel_params = inspect.signature(self.wrapped.kernel).parameters
        values = {}
        for name in kernel_params:
            if name in launch:
                values[name] = launch[name]
            elif name.endswith("_ptr"):
                values[name] = Pointer(p[name[:-4]])
            elif name in p:
                values[name] = p[name]
            else:
                raise AssertionError("unbound kernel operand: " + name)
        for pid in range(grid[0]):
            self.tl.pid = pid
            self.wrapped.kernel(**values)


def _load(path=None):
    tree = ast.parse((ROOT / REL if path is None else Path(path)).read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    tl = TensorLanguage()
    cfg = _config()
    module = ModuleType("_hcu_swa_cpu_contract_" + str(id(tree)))
    module.__dict__.update({"torch": torch, "dataclass": dataclasses.dataclass,
        "field": dataclasses.field, "Any": Any, "ClassVar": ClassVar, "cast": cast,
        "AttentionLayerBase": type("AttentionLayerBase", (), {}),
        "AttentionBackend": type("AttentionBackend", (), {}),
        "AttentionMetadataBuilder": FrameworkBuilder,
        "AttentionCGSupport": NS(UNIFORM_BATCH=1), "SlidingWindowMLASpec": Spec,
        "MLAAttentionSpec": Spec, "MultipleOf": lambda value: value,
        "get_current_vllm_config": lambda: cfg, "get_kv_quant_mode": lambda value: value,
        "max_decode_query_len": _decode_width,
        "current_platform": NS(is_rocm=lambda: True, is_xpu=lambda: False,
                               is_device_capability_family=lambda value: False),
        "triton": NS(jit=lambda *args, **kwargs: (args[0] if args else lambda fn: fn)),
        "tl": tl, "VllmTritonJitKernel": KernelBase, "kernel_launcher": lambda fn: fn,
        "get_dspark_swa_index_width": lambda window, drafts: ((max(int(window), 0) + max(int(drafts), 0) + 63) // 64) * 64,
        "get_mla_metadata": lambda: (NS(have_initialized=False), None),
        "next_power_of_2": lambda n: 1 << (n - 1).bit_length(), "cdiv": lambda a, b: (a + b - 1) // b,
        "split_decodes_and_prefills": lambda common, decode_threshold: common.split,
        "compute_token_to_req_indices": lambda q, out, n, padded: out[:n].copy_(torch.repeat_interleave(torch.arange(q.numel()-1, dtype=out.dtype), q[1:]-q[:-1])[:n]),
    })
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / REL), "exec"), module.__dict__)
    for name in ("_COMPUTE_PREFILL_METADATA_KERNEL", "_COMPUTE_SWA_INDICES_AND_LENS_KERNEL", "_COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL"):
        module.__dict__[name] = Launch(module.__dict__[name], tl)
    module._test_config = cfg
    return module


@pytest.fixture
def swa():
    module = _load()
    yield module
    sys.modules.pop(module.__name__, None)


def _builder(swa, ratios=(0, 1, 2), spec=None):
    return swa.DeepseekSparseSWAMetadataBuilder(Spec(head_size=512, tokens_per_state=1, block_size=32), _config(ratios, spec))


def _batch(qsl=(0, 1, 4), seq=(6, 9), split=(1, 1, 1, 3), causal=True):
    qsl = torch.tensor(qsl, dtype=torch.int32)
    seq = torch.tensor(seq, dtype=torch.int32)
    slots = torch.arange(int(qsl[-1]), dtype=torch.int64)
    mapping = torch.repeat_interleave(torch.arange(len(seq), dtype=torch.int32), qsl[1:]-qsl[:-1])
    def token_to_req(out):
        out[:mapping.numel()].copy_(mapping)
        return out[:mapping.numel()]
    return NS(seq_lens=seq, seq_lens_cpu_upper_bound=seq,
              query_start_loc=qsl, query_start_loc_cpu=qsl,
              block_table_tensor=torch.tensor([[3, 5], [7, 11]], dtype=torch.int32)[:len(seq)],
              slot_mapping=slots, num_reqs=len(seq), token_to_req_indices=token_to_req,
              max_query_len=int((qsl[1:]-qsl[:-1]).max()) if qsl.numel() > 1 else 0,
              causal=causal, mm_req_doc_ranges=None, split=split)


@pytest.mark.parametrize("packed,alignment,dtype", [(528, 256, torch.uint8), (584, 576, torch.uint8), (528, 256, torch.bfloat16)])
def test_native_v41_cache_geometry_and_spec(swa, packed, alignment, dtype):
    cache = swa.DeepseekV4SWACache(512, 128, dtype, "swa", NS(cache_dtype="fp8_ds_mla"),
                                  block_size=32, packed_bytes_per_token=packed,
                                  packed_page_alignment=alignment, bounded_replay=True)
    spec = cache.get_kv_cache_spec(_config())
    assert (spec.block_size, spec.state_content_bytes, spec.alignment) == (32, packed, alignment)
    assert spec.bounded_replay and spec.max_tp_shards == 1
    assert swa._test_config.compilation_config.static_forward_context["swa"] is cache
    physical = torch.empty(2, 1, 32, 528, dtype=torch.uint8)
    cache.bind_kv_cache(physical)
    assert cache.kv_cache.shape == (2, 32, 528) and cache.kv_cache.data_ptr() == physical.data_ptr()


def test_legacy_cache_defaults_and_plain_layout(swa):
    cache = swa.DeepseekV4SWACache(512, 128, torch.uint8, "old", NS(cache_dtype="fp8_ds_mla"))
    spec = cache.get_kv_cache_spec(_config())
    assert (spec.block_size, spec.state_content_bytes, spec.alignment, spec.bounded_replay) == (64, 584, 576, False)
    cache.cache_config.cache_dtype = "auto"
    spec = cache.get_kv_cache_spec(_config())
    assert spec.state_content_bytes is None and spec.alignment == 512
    assert swa.DeepseekSparseSWABackend.get_supported_kernel_block_sizes(spec) == [32]


@pytest.mark.parametrize("ratio,expected", [(0,"swaonly"), (1,"swaonly"), (2,"c2a"), (4,"c4a"), (128,"c128a")])
def test_shared_ratio_classification_and_v41_exports(swa, ratio, expected):
    assert swa._LAYER_TYPE_C1A == "c1a" and swa._LAYER_TYPE_C2A == "c2a"
    assert swa._layer_type_for(ratio) == expected
    _builder(swa, (ratio,))


def test_unknown_ratio_is_rejected(swa):
    with pytest.raises(ValueError, match="compress_ratio=3"):
        _builder(swa, (3,))


@pytest.mark.parametrize("runner_v2,parallel,expected", [(True, True, 6), (False, True, 11), (True, False, 6)])
def test_decode_threshold_tracks_current_runner_contract(swa, runner_v2, parallel, expected):
    cfg = _config(spec=NS(num_speculative_tokens=5, parallel_drafting=parallel, use_dspark=lambda: False))
    cfg.use_v2_model_runner = runner_v2
    result = swa.DeepseekSparseSWAMetadataBuilder(Spec(head_size=512, tokens_per_state=1, block_size=32), cfg)
    assert result.decode_threshold == expected


def test_ratio_one_chunk_plan_includes_real_compressed_cache(swa):
    md = swa.DeepseekSparseSWAMetadata(torch.empty(0), torch.empty(0), 32, num_prefills=2,
        prefill_seq_lens_cpu=torch.tensor([10, 12]), prefill_query_lens_cpu=torch.tensor([2, 3]),
        prefill_max_model_len=32, prefill_window_size=4, prefill_max_num_batched_tokens=8)
    assert md.get_prefill_chunk_plan(1, 1, has_compressed=True) == [(0, 2, 12, 18)]
    assert md.get_prefill_chunk_plan(1, 1) == [(0, 2, 0, 6)]
    assert md.get_prefill_chunk_plan(0, 1, has_compressed=False) == [(0, 2, 0, 6)]


@pytest.mark.parametrize("replay", [None, (0, 5), (5, 6)])
def test_mixed_batch_replay_reaches_paged_indices_and_gather_lengths(swa, replay):
    builder, common = _builder(swa), _batch()
    lower = None if replay is None else torch.tensor(replay, dtype=torch.int32)
    md = builder.build(0, common, replay_start=lower)
    assert md.replay_start is (builder.no_replay_start if lower is None else lower)
    # Request 0 query position=5; request 1 positions=6,7,8. Physical pages 3/7.
    expected_decode = [98, 99, 100, 101] if replay in (None, (0, 5)) else [101, -1, -1, -1]
    assert md.decode_swa_indices.reshape(1, 4).tolist() == [expected_decode]
    expected_prefill = ([[227, 228, 229, 230], [228, 229, 230, 231], [229, 230, 231, 232]] if replay is None else
                        [[229, 230, -1, -1], [229, 230, 231, -1], [229, 230, 231, 232]] if replay == (0, 5) else
                        [[230, -1, -1, -1], [230, 231, -1, -1], [230, 231, 232, -1]])
    assert md.prefill_swa_indices.reshape(3, 4).tolist() == expected_prefill
    assert md.prefill_gather_lens.tolist() == ([6] if replay is None else [4] if replay == (0, 5) else [3])
    calls = swa._COMPUTE_SWA_INDICES_AND_LENS_KERNEL.calls
    assert len(calls) == 2 and all(c["replay_start"] is md.replay_start for c in calls)
    assert md.query_start_loc is common.query_start_loc and md.slot_mapping is common.slot_mapping


@pytest.mark.parametrize("lower,expected", [(0, [[227,228,229,230], [228,229,230,231], [229,230,231,232]]),
    (5, [[229,230,-1,-1], [229,230,231,-1], [229,230,231,232]]),
    (6, [[230,-1,-1,-1], [230,231,-1,-1], [230,231,232,-1]])])
def test_prefill_legal_replay_lower_bound_exact_slots(swa, lower, expected):
    md = _builder(swa).build(0, _batch(), replay_start=torch.tensor([0, lower], dtype=torch.int32))
    assert md.prefill_swa_indices.reshape(3, 4).tolist() == expected
    assert md.prefill_gather_lens.tolist() == ([6] if lower == 0 else [4] if lower == 5 else [3])


def test_padding_and_page_boundary_do_not_read_stale_slots(swa):
    common = _batch(qsl=(0, 2), seq=(34,), split=(0, 1, 0, 2))
    common.slot_mapping[1] = -1
    md = _builder(swa).build(0, common)
    assert md.prefill_swa_indices.reshape(2, 4).tolist() == [[125,126,127,160], [-1,-1,-1,-1]]
    assert md.prefill_swa_lens.tolist() == [4, 0]


def test_dspark_noncausal_refresh_preserves_future_query_tokens(swa):
    spec = NS(num_speculative_tokens=2, parallel_drafting=False, use_dspark=lambda: True)
    builder = _builder(swa, spec=spec)
    common = _batch(qsl=(0, 3), seq=(7,), split=(1, 0, 3, 0), causal=False)
    md = builder.build(0, common)
    expected_row = [96,97,98,99,100,101,102] + [-1] * 57
    assert md.decode_swa_indices.reshape(3, 64).tolist() == [expected_row] * 3
    pointers = (md.decode_swa_indices.data_ptr(), md.token_to_req_indices.data_ptr(), md.is_valid_token.data_ptr())
    md.slot_mapping[1] = -1
    builder.update_draft_decode_metadata(md)
    assert md.decode_swa_indices.reshape(3, 64).tolist()[0] == expected_row
    assert md.decode_swa_indices.reshape(3, 64).tolist()[1] == [-1] * 64
    assert (md.decode_swa_indices.data_ptr(), md.token_to_req_indices.data_ptr(), md.is_valid_token.data_ptr()) == pointers
    assert len(swa._COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL.calls) == 2


def test_causal_refresh_updates_device_request_map_and_replay(swa):
    common = _batch(qsl=(0, 1, 2), seq=(6, 9), split=(2, 0, 2, 0))
    builder = _builder(swa)
    lower = torch.tensor([5, 8], dtype=torch.int32)
    md = builder.build(0, common, replay_start=lower)
    md.token_to_req_indices.fill_(0)
    builder.update_draft_decode_metadata(md)
    assert md.token_to_req_indices.tolist() == [0, 1]
    assert md.decode_swa_indices.reshape(2, 4).tolist() == [[101,-1,-1,-1], [232,-1,-1,-1]]
    assert not md.flashinfer_sparse_index_cache


def test_tile_scheduler_retains_v4_and_v41_distinct_holders(swa):
    builder = _builder(swa)
    builder._layer_types = {"swaonly", "c1a", "c2a", "c4a", "c128a"}
    plans = builder.build_tile_scheduler(1)
    assert set(plans) == builder._layer_types and all(v is not None for v in plans.values())
    assert len({id(v) for v in plans.values()}) == 5
    assert all(v is None for v in builder.build_tile_scheduler(0).values())


def test_warmup_launcher_binds_replay_operand(swa):
    for name, fields in [("ComputePrefillMetadataKernel", {"block_size":4}),
                         ("ComputeSWAIndicesAndLensKernel", {"window_size":4,"index_width":4,"has_image":0,"block_size":32,"triton_block_size":1024})]:
        cls = getattr(swa, name)
        launcher = cls()
        # TensorWarmup is a framework substitute; signatures and warmup
        # dictionaries are the unmodified production method bodies.
        swa.TritonWarmupTensor = lambda dtype, **kw: NS(dtype=dtype, **kw)
        inputs = launcher.warmup_inputs(cls.CompileKey(**fields))
        assert "replay_start" in inputs
        inspect.signature(launcher.__call__).bind(**inputs)


@pytest.mark.parametrize("padded", [False, True])
def test_empty_or_padding_only_batch_does_not_read_kv(swa, padded):
    builder = _builder(swa)
    common = (_batch(qsl=(0, 1), seq=(1,), split=(1, 0, 1, 0)) if padded else
              _batch(qsl=(0,), seq=(), split=(0, 0, 0, 0)))
    common.slot_mapping.fill_(-1)
    md = builder.build(0, common)
    if padded:
        assert md.decode_swa_indices.reshape(1, 4).tolist() == [[-1] * 4]
        assert md.decode_swa_lens.tolist() == [0]
    else:
        assert md.decode_swa_indices.numel() == 0 and md.prefill_swa_indices is None
        assert not swa._COMPUTE_SWA_INDICES_AND_LENS_KERNEL.calls
    assert not swa._COMPUTE_PREFILL_METADATA_KERNEL.calls
