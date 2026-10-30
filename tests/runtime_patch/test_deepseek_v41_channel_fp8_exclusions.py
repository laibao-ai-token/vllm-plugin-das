# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts using exact target FP8 parser, mapper and skip function bodies.

The target core is read as source, never imported. Only the layer/method and CT
construction boundaries are stubs; no tensor, checkpoint or GPU claim is made.
Set VLLM_SOURCE_ROOT and VLLM_HCU_SOURCE_ROOT for an out-of-tree source run.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
import functools
import importlib
import inspect
import os
from pathlib import Path
import re
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace

import pytest

HCU_ROOT = Path(os.environ.get("VLLM_HCU_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
CORE_ROOT = Path(os.environ.get("VLLM_SOURCE_ROOT", HCU_ROOT.parent.parent / "vllm-community-main"))
PRODUCTION = Path(os.environ.get(
    "HCU_CHANNEL_PATCH_FILE",
    HCU_ROOT / "vllm_hcu/patch/worker/core_fix/patch_deepseek_v41_channel_fp8.py",
))


def _nodes(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [n for n in tree.body if getattr(n, "name", None) in names]
    assert {n.name for n in selected} == set(names), str(path)
    return selected


def _execute(nodes, namespace, filename):
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.Module(body=[future, *copy.deepcopy(nodes)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)
    return namespace


def _class_subset(path, class_name, names, base_names):
    cls = _nodes(path, {class_name})[0]
    cls = copy.deepcopy(cls)
    cls.bases = [ast.Name(id=name, ctx=ast.Load()) for name in base_names]
    cls.decorator_list = []
    cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
    assert {n.name for n in cls.body} == set(names)
    return cls


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def route(monkeypatch):
    if not (CORE_ROOT / "vllm/model_executor/layers/quantization/fp8.py").is_file():
        pytest.skip("fixed target vLLM source is required")

    class LinearBase:
        pass

    class RoutedExperts:
        def __init__(self):
            self.moe_config = object()

    class UnquantizedLinearMethod:
        pass

    class UnquantizedFusedMoEMethod:
        def __init__(self, config):
            self.moe_config = config

    def unexpected_quantized_method(*args):
        raise AssertionError("ignored layer reached the native quantized constructor")

    ns = {"MappingProxyType": MappingProxyType, "dataclass": dataclass,
          "field": field, "re": re, "LinearBase": LinearBase,
          "RoutedExperts": RoutedExperts, "UnquantizedLinearMethod": UnquantizedLinearMethod,
          "UnquantizedFusedMoEMethod": UnquantizedFusedMoEMethod,
          "Fp8LinearMethod": unexpected_quantized_method,
          "Fp8MoEMethod": unexpected_quantized_method,
          "ACTIVATION_SCHEMES": ["static", "dynamic"]}
    quant_root = CORE_ROOT / "vllm/model_executor/layers/quantization"
    skip_file = quant_root / "utils/quant_utils.py"
    _execute(_nodes(skip_file, {"is_layer_skipped"}), ns, skip_file)
    mapper_file = CORE_ROOT / "vllm/model_executor/models/utils.py"
    _execute(_nodes(mapper_file, {"WeightsMapper"}), ns, mapper_file)
    base_file = quant_root / "base_config.py"
    base = _class_subset(base_file, "QuantizationConfig",
                         {"__init__", "get_from_keys", "get_from_keys_or"}, [])
    _execute([base], ns, base_file)
    fp8_file = quant_root / "fp8.py"
    parent = _class_subset(fp8_file, "Fp8Config",
                           {"__init__", "from_config", "apply_vllm_mapper", "get_quant_method"},
                           ["QuantizationConfig"])
    _execute([parent], ns, fp8_file)
    model_file = CORE_ROOT / "vllm/models/deepseek_v41/quant_config.py"
    child = _class_subset(model_file, "DeepseekV4FP8Config", {"get_quant_method"}, ["Fp8Config"])
    _execute([child], ns, model_file)
    config_class = ns["DeepseekV4FP8Config"]
    official = config_class.get_quant_method

    _module(monkeypatch, "vllm.model_executor.layers.linear", LinearBase=LinearBase,
            UnquantizedLinearMethod=UnquantizedLinearMethod)
    _module(monkeypatch, "vllm.model_executor.layers.fused_moe", RoutedExperts=RoutedExperts)
    _module(monkeypatch, "vllm.model_executor.layers.quantization.utils.quant_utils",
            is_layer_skipped=ns["is_layer_skipped"])

    common_file = HCU_ROOT / "vllm_hcu/patch/worker/core_fix/_common.py"
    helpers = {"PatchCompatibilityError", "load_exact_module", "require_callable",
               "require_class", "require_exact_signature"}
    adapter_ns = {"inspect": inspect, "importlib": importlib, "functools": functools,
                  "ModuleType": ModuleType}
    _execute(_nodes(common_file, helpers), adapter_ns, common_file)
    tree = ast.parse(PRODUCTION.read_text(encoding="utf-8"))
    _execute([n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))],
             adapter_ns, PRODUCTION)
    target = ModuleType(adapter_ns["TARGET_MODULE"])
    target.DeepseekV4FP8Config = config_class
    assert adapter_ns["apply_to_module"](target) is True
    assert adapter_ns["apply_to_module"](target) is False
    assert config_class._vllm_hcu_original_get_quant_method is official
    adapter_ns["_hf_quantization_config"] = lambda self: {
        "quant_method": "fp8", "activation_scheme": "dynamic", "is_per_channel": True,
    }
    calls = []
    sentinel = object()

    class CompressedTensorsConfig:
        @classmethod
        def from_config(cls, schema):
            instance = cls()
            instance.schema = schema
            instance.packed_modules_mapping = {}
            return instance

        def get_quant_method(self, layer, prefix):
            calls.append((layer, prefix, self.packed_modules_mapping))
            return sentinel

    _module(monkeypatch,
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
            CompressedTensorsConfig=CompressedTensorsConfig)

    def config(ignored=(), mode="exact", mapping=None, key="modules_to_not_convert"):
        value = config_class.from_config({"quant_method": "fp8", "activation_scheme": "dynamic",
                                          key: list(ignored)})
        value.ignored_layers_match_mode = mode
        value.packed_modules_mapping = mapping or {}
        value.expert_dtype = "fp8"
        value.is_scale_e8m0 = False
        return value

    return SimpleNamespace(config=config, linear=LinearBase, experts=RoutedExperts,
                           unquantized_linear=UnquantizedLinearMethod,
                           unquantized_experts=UnquantizedFusedMoEMethod,
                           mapper=ns["WeightsMapper"], skip=ns["is_layer_skipped"],
                           adapter=adapter_ns, calls=calls, sentinel=sentinel)


@pytest.mark.parametrize("key", ["ignored_layers", "modules_to_not_convert"])
@pytest.mark.parametrize("experts", [False, True])
def test_explicit_checkpoint_exclusion_keeps_native_method(route, key, experts):
    prefix = "model.layers.0.ffn.experts" if experts else "model.layers.0.attn.wq_b"
    layer = route.experts() if experts else route.linear()
    config = route.config([prefix], key=key)
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, route.unquantized_experts if experts else route.unquantized_linear)
    if experts:
        assert method.moe_config is layer.moe_config
    assert route.calls == []
    assert not hasattr(config, route.adapter["_CHANNEL_CONFIG_ATTR"])


@pytest.mark.parametrize("mode,ignored,prefix,skipped", [
    ("exact", "attn.wq_b", "model.layers.0.attn.wq_b", False),
    ("substring", "attn.wq_b", "model.layers.0.attn.wq_b", True),
    ("suffix", "wq_b", "model.layers.0.attn.wq_b", True),
    ("suffix", "wq_b", "model.layers.0.attn.other_wq_b", False),
    ("suffix", "ffn.experts", "model.layers.0.ffn.experts", True),
    ("exact", "model.layers.0.ffn.experts.3.w1", "model.layers.0.ffn.experts", True),
])
def test_real_target_skip_modes_and_expert_descendants(route, mode, ignored, prefix, skipped):
    config = route.config([ignored], mode=mode)
    layer = route.experts() if "experts" in prefix else route.linear()
    assert route.skip(prefix, config.ignored_layers, config.packed_modules_mapping,
                      match_mode=mode) is skipped
    method = config.get_quant_method(layer, prefix)
    if skipped:
        assert isinstance(method, route.unquantized_experts if "experts" in prefix else route.unquantized_linear)
        assert route.calls == []
    else:
        assert method is route.sentinel
        assert len(route.calls) == 1


@pytest.mark.parametrize("experts", [False, True])
def test_mapper_adjusted_exclusion_is_used_without_second_mapping(route, experts):
    suffix = "ffn.experts" if experts else "attn.wq_b"
    config = route.config(["checkpoint.layers.0." + suffix])
    config.apply_vllm_mapper(route.mapper(orig_to_new_prefix={"checkpoint.": "model."}))
    prefix = "model.layers.0." + suffix
    assert config.ignored_layers == [prefix]
    method = config.get_quant_method(route.experts() if experts else route.linear(), prefix)
    assert isinstance(method, route.unquantized_experts if experts else route.unquantized_linear)
    assert route.calls == []


@pytest.mark.parametrize("kind", ["fused", "all_shards", "partial", "none"])
def test_packed_modules_keep_native_precision_consistency(route, kind):
    root = "model.layers.0.ffn."
    mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    ignored = {"fused": [root + "gate_up_proj"],
               "all_shards": [root + "gate_proj", root + "up_proj"],
               "partial": [root + "gate_proj"], "none": []}[kind]
    config = route.config(ignored, mapping=mapping)
    if kind == "partial":
        with pytest.raises(ValueError, match="some but not all shards"):
            config.get_quant_method(route.linear(), root + "gate_up_proj")
        assert route.calls == []
    else:
        method = config.get_quant_method(route.linear(), root + "gate_up_proj")
        if kind == "none":
            assert method is route.sentinel
            assert route.calls[0][2] is config.packed_modules_mapping
        else:
            assert isinstance(method, route.unquantized_linear)
            assert route.calls == []


def test_cached_channel_config_tracks_parent_mapping_and_later_exclusions(route):
    config = route.config()
    prefix = "model.layers.0.attn.wq_b"
    assert config.get_quant_method(route.linear(), prefix) is route.sentinel
    cached = getattr(config, route.adapter["_CHANNEL_CONFIG_ATTR"])
    config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    assert config.get_quant_method(route.linear(), prefix) is route.sentinel
    assert getattr(config, route.adapter["_CHANNEL_CONFIG_ATTR"]) is cached
    assert route.calls[-1][2] is config.packed_modules_mapping
    assert cached.schema["ignore"] == ["re:.*attn\\.wo_a.*"]
    config.ignored_layers = [prefix]
    assert isinstance(config.get_quant_method(route.linear(), prefix), route.unquantized_linear)
    assert len(route.calls) == 2


def test_native_v41_expert_exact_check_precedes_parent_match_mode(route):
    prefix = "model.layers.0.ffn.experts"
    config = route.config([prefix + ".3.w1"], mode="suffix")
    assert not route.skip(prefix, config.ignored_layers, match_mode="suffix")
    layer = route.experts()
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, route.unquantized_experts)
    assert method.moe_config is layer.moe_config
    assert route.calls == []


def test_wo_a_bf16_route_is_preserved(route):
    assert isinstance(route.config().get_quant_method(route.linear(), "model.layers.0.attn.wo_a"),
                      route.unquantized_linear)
    assert route.calls == []


def test_invalid_skip_mode_keeps_native_rejection(route):
    with pytest.raises(ValueError, match="Unsupported layer skip match mode"):
        route.config(mode="invalid").get_quant_method(route.linear(), "a.wq_b")
    assert route.calls == []
