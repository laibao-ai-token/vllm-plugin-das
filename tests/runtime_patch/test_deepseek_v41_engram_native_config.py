# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Native Engram config and callback contracts, without a target-core import.

Pydantic/config decorators are removed for isolated execution. Their validation
methods are called explicitly; these tests do not validate decorator integration,
actual distributed groups, host registration, GPU execution or full bootstrap.
"""
from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

HCU_ROOT = Path(os.environ.get("VLLM_HCU_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
CORE_ROOT = Path(os.environ.get("VLLM_SOURCE_ROOT", HCU_ROOT.parent.parent / "vllm-community-main"))
DISPATCHER_FILE = Path(os.environ.get(
    "HCU_ENGRAM_DISPATCHER_FILE", HCU_ROOT / "vllm_hcu/patch/platform/core_fix/__init__.py",
))
RETIRED_FILE = Path(os.environ.get(
    "HCU_ENGRAM_RETIRED_FILE", HCU_ROOT / "vllm_hcu/patch/platform/core_fix/patch_engram_config.py",
))


def _execute(nodes, namespace, filename):
    future = ast.parse("from __future__ import annotations").body[0]
    tree = ast.Module(body=[future, *copy.deepcopy(nodes)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(filename), "exec"), namespace)
    return namespace


@pytest.fixture
def native():
    path = CORE_ROOT / "vllm/config/engram.py"
    if not path.is_file():
        pytest.skip("fixed target vLLM source is required")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "model_has_engram_layers":
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "EngramConfig":
            node = copy.deepcopy(node)
            node.decorator_list = []
            for member in node.body:
                if isinstance(member, ast.FunctionDef):
                    member.decorator_list = []
            nodes.append(node)
    ns = _execute(nodes, {}, path)
    return SimpleNamespace(config=ns["EngramConfig"], has_layers=ns["model_has_engram_layers"])


def _model(architecture="DeepseekV41ForCausalLM", field="engram_layer_ids", ids=(1,)):
    return SimpleNamespace(architecture=architecture, hf_text_config=SimpleNamespace(**{field: list(ids)}))


def _parallel(dp=1, tp=1, elastic=False):
    return SimpleNamespace(data_parallel_size=dp, tensor_parallel_size=tp, enable_elastic_ep=elastic)


@pytest.mark.parametrize("architecture,field", [
    ("DeepseekV41ForCausalLM", "engram_layer_ids"),
    ("Qwen4ExpForCausalLM", "ple_layer_ids"),
    ("Qwen4ExpForConditionalGeneration", "ple_layer_ids"),
])
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_native_models_keep_validation_without_hcu_bypass(native, architecture, field, cpu_offload):
    config = native.config()
    config.cpu_offload = cpu_offload
    model = _model(architecture, field)
    config.verify_model_config(model)
    assert native.has_layers(model)


@pytest.mark.parametrize("model", [None, _model("UnsupportedArchitecture"),
                                   _model(ids=()), _model(field="ple_layer_ids")])
def test_native_rejections_are_preserved(native, model):
    with pytest.raises(ValueError, match="supported Engram embeddings"):
        native.config().verify_model_config(model)
    assert not native.has_layers(model)


@pytest.mark.parametrize("dp,tp,across,expected", [
    (1, 4, False, 4), (2, 4, False, 4), (2, 4, True, 8), (8, 1, True, 8),
])
def test_native_embedding_group_size_is_not_replaced(native, dp, tp, across, expected):
    config = native.config()
    config.embedding_across_dp = across
    assert config.get_parallel_size(_parallel(dp=dp, tp=tp)) == expected


@pytest.mark.parametrize("offload,thp,elastic,expected", [
    (True, False, False, True), (False, False, False, False),
    (True, True, False, False), (True, False, True, False),
])
def test_native_dp_shared_default_keeps_eligibility_checks(native, offload, thp, elastic, expected):
    config = native.config()
    config.cpu_offload = offload
    config.use_thp = thp
    config.dp_shared_memory = None
    config.resolve_dp_shared_memory(_parallel(dp=2, elastic=elastic))
    assert config.dp_shared_memory is expected


@pytest.mark.parametrize("offload,shared,thp,message", [
    (False, True, False, "dp_shared_memory requires cpu_offload"),
    (False, False, True, "use_thp requires"),
    (True, True, True, "use_thp requires"),
])
def test_native_memory_topology_rejections_are_not_bypassed(native, offload, shared, thp, message):
    config = native.config()
    config.cpu_offload = offload
    config.dp_shared_memory = shared
    config.use_thp = thp
    with pytest.raises(ValueError, match=message):
        config._validate_shared_memory()


@pytest.mark.parametrize("shared,across,dp,elastic,message", [
    (True, False, 1, False, "data_parallel_size > 1"),
    (True, False, 2, True, "not supported with elastic EP"),
    (False, True, 2, True, "not supported with elastic EP"),
])
def test_native_parallel_rejections_are_preserved(native, shared, across, dp, elastic, message):
    config = native.config()
    config.dp_shared_memory = shared
    config.embedding_across_dp = across
    with pytest.raises(ValueError, match=message):
        config.verify_parallel_config(_parallel(dp=dp, elastic=elastic))


def test_registered_callbacks_retire_only_the_obsolete_engram_adapter():
    tree = ast.parse(DISPATCHER_FILE.read_text(encoding="utf-8"))
    ordered = next(n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "_ORDERED_ADAPTERS" for t in n.targets))
    names = [node.id for node in ordered.value.elts]
    assert "patch_engram_config" not in names
    assert not any(isinstance(n, ast.ImportFrom) and any(alias.name == "patch_engram_config" for alias in n.names)
                   for n in tree.body)
    assert names[:3] == ["patch_deep_gemm_foundation", "patch_envs", "patch_import_utils"]
    arms = []
    ns = {name: SimpleNamespace(PATCH_ID=name, TARGET_MODULE="target." + name,
                                TARGETS=("symbol." + name,),
                                arm_partial_import_bridge=lambda: arms.append(True)) for name in names}
    registrations = []

    class Coordinator:
        def register_callback(self, patch_id, target, callback, *, targets):
            registrations.append((patch_id, target, callback, targets))
            return patch_id

    for adapter in ns.values():
        adapter.apply_to_module = object()
    coordinator = Coordinator()
    ns["IMPORT_COORDINATOR"] = coordinator
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"register_platform_core_callbacks", "platform_core_callback_names"}]
    _execute([ordered, *functions], ns, DISPATCHER_FILE)
    assert ns["register_platform_core_callbacks"](coordinator) == tuple(names)
    assert len(registrations) == len(names)
    assert [r[0] for r in registrations] == names
    assert all("engram" not in r[1] for r in registrations)
    assert len(arms) == 2
    assert ns["platform_core_callback_names"]() == tuple((name, "target." + name) for name in names)


def test_obsolete_engram_wrapper_is_removed_from_hcu():
    assert not RETIRED_FILE.exists()
