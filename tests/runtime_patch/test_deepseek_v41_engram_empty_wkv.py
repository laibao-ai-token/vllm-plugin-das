# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Zero-row WKV contracts using actual V41/Linear/CT/FP8 source bodies on CPU.

Quantization, GEMM, lookup, GPU post/row-selection kernels and collectives are
explicit CPU boundaries. Tests assert their order, not real multi-rank progress.
"""
from __future__ import annotations
import ast
import copy
import functools
from functools import cache
import importlib
import inspect
import logging
from math import prod
import os
from pathlib import Path
from threading import RLock
from types import ModuleType, SimpleNamespace

import pytest
import torch

HCU_ROOT = Path(os.environ.get("VLLM_HCU_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
CORE_ROOT = Path(os.environ.get("VLLM_SOURCE_ROOT", HCU_ROOT.parent.parent / "vllm-community-main"))
PATCH_ROOT = Path(os.environ.get("HCU_EMPTY_WKV_PATCH_ROOT", HCU_ROOT))
RUNTIME_REL = "vllm_hcu/runtime_compat/scaled_mm.py"
PATCH_REL = "vllm_hcu/patch/worker/op_opt/patch_scaled_mm_linear_kernel.py"


def _exec(nodes, ns, path):
    tree = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *copy.deepcopy(nodes)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
    return ns


def _load(path, ns):
    return _exec([n for n in ast.parse(path.read_text()).body if not isinstance(n, (ast.Import, ast.ImportFrom))], ns, path)


def _class(path, name, methods):
    cls = copy.deepcopy(next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name))
    cls.bases = []
    cls.decorator_list = []
    cls.body = [n for n in cls.body if getattr(n, "name", None) in methods]
    assert {n.name for n in cls.body} == set(methods)
    return cls


@pytest.fixture
def route():
    assert not torch.cuda.is_initialized()
    events = []
    common = _load(HCU_ROOT / "vllm_hcu/patch/worker/op_opt/_common.py", {"inspect": inspect, "importlib": importlib})
    patch_ns = {key: common[key] for key in ("PatchCompatibilityError", "already_applied", "load_exact_module",
                                            "require_callable", "require_class", "require_exact_signature")}
    patch_ns.update({"torch": torch, "functools": functools, "prod": prod, "ModuleType": ModuleType})
    _load(PATCH_ROOT / PATCH_REL, patch_ns)
    core_kernel = CORE_ROOT / "vllm/model_executor/kernels/linear/scaled_mm/ScaledMMLinearKernel.py"
    kernel_ns = {"torch": torch, "as_quantized_activation": lambda x, key: None}
    _exec([_class(core_kernel, "FP8ScaledMMLinearKernel", {"_get_layer_params", "apply_weights", "get_output_padding"})], kernel_ns, core_kernel)
    base = kernel_ns["FP8ScaledMMLinearKernel"]
    base.input_quant_key = lambda self: None
    module = ModuleType(patch_ns["TARGET_MODULE"])
    module.FP8ScaledMMLinearKernel = base
    assert patch_ns["apply_to_module"](module)
    assert patch_ns["apply_to_module"](module) is False

    class ChannelWiseTorchFP8ScaledMMLinearKernel(base):
        @classmethod
        def is_supported(cls, compute_capability=None):
            return True, None
        def apply_scaled_mm(self, *, A, B, out_dtype, As, Bs, bias, output_shape):
            raise AssertionError("unpatched backend")

    runtime_ns = {"torch": torch, "logging": logging, "cache": cache, "signature": inspect.signature,
                  "prod": prod, "RLock": RLock, "ModuleType": ModuleType}
    _load(PATCH_ROOT / RUNTIME_REL, runtime_ns)
    def backend(A, B, As, Bs, out_dtype, bias):
        events.append("gemm")
        assert A.shape[0] > 0, "positive-M vendor ABI reached"
        output = (A.float() * As) @ B.float()
        output = output * Bs.reshape(1, -1)
        if bias is not None:
            output += bias
        return output.to(out_dtype)
    runtime_ns["_resolve_scaled_mm_backend"] = lambda: ("target-triton", backend)
    runtime_ns["_ensure_channel_fp8_custom_op"] = lambda selected: selected
    module = ModuleType("vllm.model_executor.kernels.linear.scaled_mm.pytorch")
    module.ChannelWiseTorchFP8ScaledMMLinearKernel = ChannelWiseTorchFP8ScaledMMLinearKernel
    runtime_ns["install_fp8_scaled_mm_compat"](module)
    runtime_ns["install_fp8_scaled_mm_compat"](module)

    def kernel(out_dtype=None):
        value = ChannelWiseTorchFP8ScaledMMLinearKernel()
        value.config = SimpleNamespace(out_dtype=out_dtype)
        value.fp8_dtype = torch.float8_e4m3fn
        value.layer_param_names = ("weight", "weight_scale", "input_scale", "input_scale_ub")
        def quant(x, *args):
            events.append("quant")
            assert x.shape[0] > 0, "zero-row quantizer reached"
            return x.to(torch.float8_e4m3fn), torch.ones((x.shape[0], 1), dtype=torch.float32)
        value.quant_fp8 = quant
        return value

    ns = {"torch": torch}
    ct = CORE_ROOT / "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py"
    scheme = CORE_ROOT / "vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py"
    linear = CORE_ROOT / "vllm/model_executor/layers/linear.py"
    _exec([_class(ct, "CompressedTensorsLinearMethod", {"apply"})], ns, ct)
    _exec([_class(scheme, "CompressedTensorsW8A8Fp8", {"apply_weights"})], ns, scheme)
    _exec([_class(linear, "ColumnParallelLinear", {"forward"})], ns, linear)
    def gather(x, dim=-1):
        events.append("tp_gather_output" if dim == -1 else "tp_gather_heads")
        return torch.cat((x, x), dim=dim)
    ns["tensor_model_parallel_all_gather"] = gather
    def layer(k=4, n=6, tp=1, sp=False, dtype=torch.bfloat16, out_dtype=None):
        value = ns["ColumnParallelLinear"]()
        value.bias = None
        value.skip_bias_add = False
        value.return_bias = False
        value.gather_output = True
        value.tp_size = 1 if sp else tp
        local_n = n // value.tp_size
        value.weight = torch.ones((local_n, k), dtype=torch.float8_e4m3fn).t()
        value.weight_scale = torch.ones((local_n, 1), dtype=torch.float32)
        value.quant_method = ns["CompressedTensorsLinearMethod"]()
        value.scheme = ns["CompressedTensorsW8A8Fp8"]()
        value.scheme.fp8_linear = kernel(out_dtype)
        value.__class__.__call__ = value.__class__.forward
        return value

    engram_path = CORE_ROOT / "vllm/models/deepseek_v41/common/engram.py"
    class PostKernel:
        def __getitem__(self, grid):
            def run(hidden, kv, q, k, mask, output, *args, **kwargs):
                events.append("post")
                output.copy_(hidden + 1)
            return run
    class SelectKernel:
        def __getitem__(self, grid):
            def run(gathered, output, source_tokens, start, count, local_width, width, **kwargs):
                for token in range(output.shape[0]):
                    for col in range(width):
                        index = (col // local_width * source_tokens + start + token) * local_width + col % local_width
                        output.reshape(-1)[token * width + col] = gathered.reshape(-1)[index] if start + token < source_tokens else 0
            return run
    class DP:
        world_size = 2
        rank_in_group = 0
        def all_gather(self, x, dim=0):
            events.append("dp_gather")
            return torch.cat((x, x), dim=dim)
    ens = {"torch": torch, "triton": SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b, next_power_of_2=lambda v:1 << (v-1).bit_length()),
           "get_engram_dp_group": lambda: DP(), "engram_gathered_num_tokens": lambda: 2,
           "tensor_model_parallel_all_gather": gather, "get_tensor_model_parallel_rank": lambda: 0,
           "get_tensor_model_parallel_world_size": lambda: 2,
           "_fused_engram_post_wkv_kernel": PostKernel(), "_engram_select_rows_kernel": SelectKernel()}
    tree = ast.parse(engram_path.read_text())
    functions = [n for n in tree.body if getattr(n, "name", None) in ("_gather_engram_rows", "_engram_select_rows")]
    _exec([*functions, _class(engram_path, "Engram", {"prepare_embeddings", "_ready_rows", "embed", "forward"})], ens, engram_path)
    def engram(tokens=0, dp=1, tp=1, shared=False, offload=False, sp=False, dtype=torch.bfloat16):
        value = ens["Engram"]()
        effective_dp = 1 if shared else dp
        value.dim = 2
        value.hc_mult = 2
        value.use_sequence_parallel = sp
        value.eps = value.clamp_value = 1e-6
        value._batch = None
        value._prefetch_stream = object() if offload else None
        value._prefetch_done = object() if offload else None
        value._start_prefetch = lambda hashes, rows: events.append("lookup")
        value._finish_prefetch = lambda event: events.append("prefetch_wait")
        heads = effective_dp * tp
        value.embed_tokens = SimpleNamespace(dp_size=effective_dp, tp_size=tp, n_hash_cols=heads,
            cpu_offload=offload, dp_shared_memory=shared,
            lookup=lambda hashes, rows: events.append("lookup"))
        value.staged_rows = torch.ones((max(2 * effective_dp, tokens), 1, 2), dtype=dtype)
        value.wkv = layer(k=heads * 2, n=6, tp=tp, sp=sp, dtype=dtype)
        value.q_weight = value.k_weight = torch.zeros((2, 2), dtype=dtype)
        hidden_tokens = (tokens + tp - 1) // tp if sp else tokens
        hidden = torch.ones((hidden_tokens, 2, 2), dtype=dtype)
        hashes = torch.empty((tokens, heads), dtype=torch.int32)
        return value, hidden, hashes
    return SimpleNamespace(kernel=kernel, layer=layer, engram=engram, events=events,
        runtime=runtime_ns, patch=patch_ns, channel=ChannelWiseTorchFP8ScaledMMLinearKernel,
        engram_ns=ens, original_fp8=base._vllm_hcu_original_apply_weights)


@pytest.mark.parametrize("dp,tp,shared,offload,sp", [
    (1,1,False,False,False), (2,1,False,True,False),
    (2,2,False,True,False), (2,2,True,True,False),
    (2,2,False,False,True), (2,2,True,True,True),
])
@pytest.mark.parametrize("tokens", [0, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_engram_preserves_collectives_before_zero_row_guard(route, dp, tp, shared, offload, sp, tokens, dtype):
    engram, hidden, hashes = route.engram(tokens, dp, tp, shared, offload, sp, dtype)
    engram.prepare_embeddings(hashes)
    output = engram.forward(hidden, hashes)
    assert output.shape == hidden.shape and output.dtype == dtype and output.device == hidden.device
    expected = ["lookup"]
    if offload:
        expected.append("prefetch_wait")
    if dp > 1 and not shared:
        expected.append("dp_gather")
    if tp > 1:
        expected.append("tp_gather_heads")
    if tokens:
        expected += ["quant", "gemm"]
    if tp > 1 and not sp:
        expected.append("tp_gather_output")
    if tokens:
        expected.append("post")
        torch.testing.assert_close(output, hidden + 1)
    assert route.events == expected


@pytest.mark.parametrize("shape", [(0,4), (2,0,4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("out_dtype", [None, torch.float32])
def test_empty_linear_skips_quantization_and_vendor_gemm(route, shape, dtype, out_dtype):
    layer = route.layer(k=4, out_dtype=out_dtype)
    output = layer(torch.empty(shape, dtype=dtype))
    assert output.shape == (*shape[:-1], 6)
    assert output.dtype == (dtype if out_dtype is None else out_dtype)
    assert output.numel() == 0 and route.events == []


@pytest.mark.parametrize("prequantized", [False, True])
def test_empty_malformed_input_still_fails_weight_and_scale_contracts(route, prequantized):
    layer = route.layer(k=4)
    if prequantized:
        layer.weight_scale = torch.ones((7,1))
        with pytest.raises(ValueError, match="weight scale"):
            layer.scheme.fp8_linear.apply_weights(layer, torch.empty((0,4)),
                x_and_scale_quanted=(torch.empty((0,4), dtype=torch.float8_e4m3fn), torch.empty((0,1))))
    else:
        with pytest.raises(ValueError, match="weight K"):
            layer(torch.empty((0,5)))
    assert route.events == []


@pytest.mark.parametrize("backend", ["target-triton", "lightop"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_valid_empty_prequantized_input_skips_backend(route, backend, dtype):
    route.channel._hcu_fp8_backend = backend
    layer = route.layer(k=4)
    output = layer.scheme.fp8_linear.apply_weights(layer, torch.empty((0,4), dtype=dtype),
        x_and_scale_quanted=(torch.empty((0,4), dtype=torch.float8_e4m3fn), torch.empty((0,1))))
    assert output.shape == (0,6) and output.dtype == dtype and route.events == []


def test_direct_empty_scaled_mm_validates_before_short_circuit(route):
    layer = route.layer(k=4)
    kernel = layer.scheme.fp8_linear
    args = dict(A=torch.empty((0,4), dtype=torch.float8_e4m3fn), B=layer.weight,
                As=torch.empty((0,1)), Bs=layer.weight_scale, out_dtype=torch.bfloat16,
                bias=None, output_shape=(0,6))
    output = kernel.apply_scaled_mm(**args)
    assert output.shape == (0,6) and output.dtype == torch.bfloat16
    for field, value, match in (("Bs", torch.empty((7,1)), "weight scale"),
                                ("output_shape", (1,6), "output_shape"),
                                ("bias", torch.empty((7,)), "bias")):
        invalid = {**args, field:value}
        with pytest.raises(ValueError, match=match):
            kernel.apply_scaled_mm(**invalid)
    assert route.events == []


def test_nonzero_native_fp8_path_is_unchanged(route):
    layer = route.layer(k=4)
    x = torch.tensor([[1.,2.,3.,4.]], dtype=torch.bfloat16)
    expected = route.original_fp8(layer.scheme.fp8_linear, layer, x)
    before = list(route.events)
    route.events.clear()
    actual = layer(x)
    assert route.events == before == ["quant", "gemm"]
    torch.testing.assert_close(actual, expected)


def test_unreviewed_kernel_keeps_original_empty_behavior(route):
    route.channel._hcu_fp8_backend = "unreviewed"
    with pytest.raises(AssertionError, match="zero-row quantizer"):
        route.layer()(torch.empty((0,4), dtype=torch.bfloat16))
    assert route.events == ["quant"]


def test_native_target_triton_positive_m_rejection_is_real_source(route):
    path = CORE_ROOT / "vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py"
    function = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "triton_scaled_mm")
    ns = {"torch": torch}
    _exec([function], ns, path)
    with pytest.raises(AssertionError):
        ns["triton_scaled_mm"](torch.empty((0,4), dtype=torch.float8_e4m3fn),
            torch.empty((6,4), dtype=torch.float8_e4m3fn).t(), torch.empty((0,1)), torch.ones((6,1)), torch.bfloat16)
    assert route.events == []
