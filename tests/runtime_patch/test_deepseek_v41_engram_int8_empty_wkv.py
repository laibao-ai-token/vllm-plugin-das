# SPDX-License-Identifier: Apache-2.0
"""CPU source-body contracts; no core/package import or real GPU collectives."""
from __future__ import annotations

import ast
import copy
import functools
import importlib
import inspect
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

HCU = Path(os.environ.get('VLLM_HCU_SOURCE_ROOT', Path(__file__).resolve().parents[2]))
CORE = Path(os.environ.get('VLLM_SOURCE_ROOT', HCU.parent.parent / 'vllm-community-main'))
CANDIDATE = Path(os.environ.get('HCU_INT8_PATCH_ROOT', str(HCU)))
REL = 'vllm_hcu/model_executor/layers/quantization/int8_runtime.py'


def body(path, ns):
    nodes = [n for n in ast.parse(path.read_text()).body
             if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[
        ast.parse('from __future__ import annotations').body[0], *nodes
    ], type_ignores=[])), str(path), 'exec'), ns)
    return ns


def cls(path, name, names):
    n = copy.deepcopy(next(n for n in ast.parse(path.read_text()).body
                          if isinstance(n, ast.ClassDef) and n.name == name))
    n.bases = []
    n.decorator_list = []
    n.body = [m for m in n.body if getattr(m, 'name', '') in names]
    assert {m.name for m in n.body} == set(names)
    return n


def execute(nodes, ns, path):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[
        ast.parse('from __future__ import annotations').body[0], *nodes
    ], type_ignores=[])), str(path), 'exec'), ns)


@pytest.fixture
def route(monkeypatch):
    assert not torch.cuda.is_initialized()
    events = []
    envs = ModuleType('vllm_hcu.platforms.envs')
    envs.VLLM_HCU_USE_CUSTOM_OPS = True
    envs.VLLM_HCU_USE_FUSED_SILU_MUL_QUANT = True
    envs.VLLM_HCU_USE_FUSED_RMS_QUANT = False
    envs.custom_quantization_gemm_enabled = lambda: True
    for name in ('vllm_hcu', 'vllm_hcu.platforms'):
        pkg = ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    sys.modules['vllm_hcu.platforms'].envs = envs
    monkeypatch.setitem(sys.modules, envs.__name__, envs)
    ns = body(CANDIDATE / REL, {'torch': torch})
    runtime = ModuleType('vllm_hcu.model_executor.layers.quantization.int8_runtime')
    runtime.__dict__.update(ns)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    lightop = ModuleType('lightop')
    lightop.__path__ = []
    quant = ModuleType('lightop.quant')
    gemm = ModuleType('lightop.gemm_ops')
    def q(x):
        events.append('quant')
        assert x.numel() > 0, 'zero-row vendor quantizer reached'
        return x.to(torch.int8), torch.ones((*x.shape[:-1], 1))
    def g(a, b, sa, sb, m, n, k, layout, dtype):
        events.append('gemm')
        assert m > 0, 'zero-row vendor GEMM reached'
        return True, ((a.float() * sa) @ (b.float() * sb).t()).to(dtype)
    quant.per_token_quant_int8 = q
    gemm.hipblaslt_w8a8_gemm = g
    for mod in (lightop, quant, gemm):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return SimpleNamespace(apply=runtime.apply_int8_linear,
                           error=runtime.HcuInt8LinearError, events=events,
                           envs=envs, runtime=runtime)


@pytest.mark.parametrize('shape', [(0, 4), (2, 0, 4)])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('params', [torch.bfloat16, torch.float16, torch.float32])
def test_empty_local_contract(route, shape, dtype, fused, params):
    x = torch.empty(shape, dtype=dtype)
    pair = (torch.empty_like(x, dtype=torch.int8), torch.empty((*shape[:-1], 1)))
    out = route.apply(x, torch.ones((6, 4), dtype=torch.int8), torch.ones((6, 1)), params,
                      x_and_scale_quanted=pair if fused else None)
    assert out.shape == (*shape[:-1], 6)
    assert out.dtype == (params if params in (torch.float16, torch.bfloat16) else torch.bfloat16)
    assert out.device == x.device and out.numel() == 0
    assert route.events == []


@pytest.mark.parametrize('case', ['k', 'weight_dtype', 'scale_dtype', 'scale_shape', 'bias_dtype',
                                  'bias_shape', 'fused_shape', 'fused_dtype', 'fused_tuple', 'asymmetric', 'zero_k',
                                  'weight_device', 'scale_device', 'fused_device', 'bias_device'])
def test_empty_rejects_invalid_contract(route, case):
    args = dict(input=torch.empty((0, 4), dtype=torch.bfloat16),
                weight=torch.ones((6, 4), dtype=torch.int8),
                weight_scale=torch.ones((6, 1)), params_dtype=torch.bfloat16)
    if case == 'k': args['input'] = torch.empty((0, 5))
    if case == 'weight_dtype': args['weight'] = args['weight'].float()
    if case == 'scale_dtype': args['weight_scale'] = args['weight_scale'].half()
    if case == 'scale_shape': args['weight_scale'] = torch.ones((7, 1))
    if case == 'bias_dtype': args['bias'] = torch.ones(6)
    if case == 'bias_shape': args['bias'] = torch.ones(7, dtype=torch.bfloat16)
    if case == 'fused_shape': args['x_and_scale_quanted'] = (torch.empty((0, 5), dtype=torch.int8), torch.empty((0, 1)))
    if case == 'fused_dtype': args['x_and_scale_quanted'] = (torch.empty((0, 4)), torch.empty((0, 1)))
    if case == 'fused_tuple': args['x_and_scale_quanted'] = ('bad', 'bad')
    if case == 'asymmetric': args['input_zero_point'] = torch.ones(1, dtype=torch.int8)
    if case == 'zero_k': args['input'] = torch.empty((0, 0))
    if case == 'weight_device': args['weight'] = torch.empty((6, 4), dtype=torch.int8, device='meta')
    if case == 'scale_device': args['weight_scale'] = torch.empty((6, 1), device='meta')
    if case == 'fused_device': args['x_and_scale_quanted'] = (torch.empty((0, 4), dtype=torch.int8, device='meta'), torch.empty((0, 1)))
    if case == 'bias_device': args['bias'] = torch.empty(6, dtype=torch.bfloat16, device='meta')
    with pytest.raises(route.error): route.apply(**args)
    assert route.events == []


@pytest.mark.parametrize('fused', [False, True])
def test_positive_rows_keep_quant_gemm_and_values(route, fused):
    x = torch.ones((2, 4), dtype=torch.bfloat16)
    pair = (x.to(torch.int8), torch.ones((2, 1))) if fused else None
    out = route.apply(x, torch.ones((6, 4), dtype=torch.int8), torch.ones((6, 1)),
                      torch.bfloat16, bias=torch.ones(6, dtype=torch.bfloat16),
                      x_and_scale_quanted=pair)
    torch.testing.assert_close(out, torch.full((2, 6), 5, dtype=torch.bfloat16))
    assert route.events == (['gemm'] if fused else ['quant', 'gemm'])


def install_scheme(route):
    common = body(HCU / 'vllm_hcu/patch/worker/op_opt/_common.py',
                  {'inspect': inspect, 'importlib': importlib})
    p = HCU / 'vllm_hcu/patch/worker/op_opt/patch_compressed_tensors_w8a8_int8.py'
    ns = {k: common[k] for k in ('PatchCompatibilityError', 'already_applied',
          'load_exact_module', 'require_callable', 'require_class', 'require_exact_signature')}
    ns.update(functools=functools, ModuleType=ModuleType)
    body(p, ns)
    ns['_custom_quantization_enabled'] = lambda: route.envs.custom_quantization_gemm_enabled()
    scheme_path = CORE / 'vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py'
    sn = {'torch': torch}
    execute([cls(scheme_path, 'CompressedTensorsW8A8Int8',
                 {'apply_weights', 'process_weights_after_loading'})], sn, scheme_path)
    module = ModuleType(ns['TARGET_MODULE'])
    module.CompressedTensorsW8A8Int8 = sn['CompressedTensorsW8A8Int8']
    assert ns['apply_to_module'](module)
    assert ns['apply_to_module'](module) is False
    ct_path = CORE / 'vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py'
    execute([cls(ct_path, 'CompressedTensorsLinearMethod', {'apply'})], sn, ct_path)
    ct_patch = HCU / 'vllm_hcu/patch/worker/op_opt/patch_compressed_tensors.py'
    ct_ns = {k: common[k] for k in ('PatchCompatibilityError', 'already_applied',
             'load_exact_module', 'require_callable', 'require_class', 'require_exact_signature')}
    ct_ns.update(functools=functools, ModuleType=ModuleType)
    body(ct_patch, ct_ns)
    ct_module = ModuleType(ct_ns['TARGET_MODULE'])
    ct_module.CompressedTensorsLinearMethod = sn['CompressedTensorsLinearMethod']
    assert ct_ns['apply_to_module'](ct_module)
    assert ct_ns['apply_to_module'](ct_module) is False
    return sn


@pytest.mark.parametrize('tokens', [0, 2])
@pytest.mark.parametrize('tp', [1, 2])
@pytest.mark.parametrize('mode', ['ordinary', 'fused', 'invalid_fused', 'master_off'])
def test_actual_hcu_linear_binding_keeps_output_gather(route, tokens, tp, mode):
    sn = install_scheme(route)
    path = HCU / 'vllm_hcu/model_executor/layers/linear.py'
    def gather(x):
        route.events.append('tp_gather_output')
        return torch.cat((x, x), -1)
    sn['tensor_model_parallel_all_gather'] = gather
    execute([cls(path, 'ColumnParallelLinear', {'forward', '_forward_with_hcu_quanted'})], sn, path)
    layer = sn['ColumnParallelLinear']()
    layer.bias = None
    layer.skip_bias_add = layer.return_bias = False
    layer.gather_output = True
    layer.tp_size = tp
    layer.weight = torch.ones((6 // tp, 4), dtype=torch.int8)
    layer.weight_scale = torch.ones((6 // tp, 1))
    layer.params_dtype = torch.bfloat16
    layer.scheme = sn['CompressedTensorsW8A8Int8']()
    layer.quant_method = sn['CompressedTensorsLinearMethod']()
    x = torch.ones((tokens, 4), dtype=torch.bfloat16)
    pair = None
    if mode in ('fused', 'invalid_fused'):
        pair = (torch.ones((tokens, 5 if mode == 'invalid_fused' else 4), dtype=torch.int8),
                torch.ones((tokens, 1)))
    if mode == 'master_off':
        route.envs.custom_quantization_gemm_enabled = lambda: False
        def fallback(layer, value, bias):
            route.events.append('target_fallback')
            return value.new_empty((tokens, 6 // tp))
        layer.scheme.kernel = SimpleNamespace(apply_weights=fallback)
    if mode == 'invalid_fused':
        with pytest.raises(route.error, match='quantized input/scale shapes'):
            layer._forward_with_hcu_quanted(x, pair)
        assert route.events == []
        return
    out = layer._forward_with_hcu_quanted(x, pair)
    assert out.shape == (tokens, 6) and out.dtype == torch.bfloat16
    expected = ['target_fallback'] if mode == 'master_off' else (
        (['gemm'] if mode == 'fused' else ['quant', 'gemm']) if tokens else []
    )
    if tp > 1: expected.append('tp_gather_output')
    assert route.events == expected
    if tokens and mode != 'master_off':
        torch.testing.assert_close(out, torch.full_like(out, 4))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('scale_shape', [(), (1,), (1, 1), (6, 1)])
@pytest.mark.parametrize('bias_shape', [(), (1,), (6,), (1, 6), (0, 6)])
def test_empty_accepts_supported_scale_and_bias_shapes(route, scale_shape, bias_shape):
    out = route.apply(torch.empty((0, 4), dtype=torch.bfloat16),
                      torch.ones((6, 4), dtype=torch.int8), torch.ones(scale_shape),
                      torch.bfloat16, bias=torch.empty(bias_shape, dtype=torch.bfloat16))
    assert out.shape == (0, 6) and out.dtype == torch.bfloat16
    assert route.events == []


@pytest.mark.parametrize('tokens', [0, 2])
def test_disabled_fused_flags_ignore_unused_prequantized_argument(route, tokens):
    route.envs.VLLM_HCU_USE_FUSED_SILU_MUL_QUANT = False
    route.envs.VLLM_HCU_USE_FUSED_RMS_QUANT = False
    out = route.apply(torch.ones((tokens, 4), dtype=torch.bfloat16),
                      torch.ones((6, 4), dtype=torch.int8), torch.ones((6, 1)),
                      torch.bfloat16, x_and_scale_quanted=('unused', 'unused'))
    assert out.shape == (tokens, 6)
    assert route.events == (['quant', 'gemm'] if tokens else [])
