# SPDX-License-Identifier: Apache-2.0
"""Portable runner-selection regression; no vLLM/HCU/torch imports.

Exercise the actual HCU callback and PCP guards via AST. The property below is
only the portable native selection contract; the handoff-only probe executes the
complete pinned native property, collectors, and V1/V2 validators separately.
"""
from __future__ import annotations

import ast
import builtins
import functools
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

REL = "vllm_hcu/patch/platform/core_fix/patch_vllm_config.py"
ENV = "VLLM_USE_V2_MODEL_RUNNER"
HAS_TRITON = True


class Environment:
    @property
    def VLLM_USE_V2_MODEL_RUNNER(self):
        value = os.environ.get(ENV)
        return None if value is None else bool(int(value))


envs = Environment()


class PortableConfig:
    """An actual descriptor, preserving native auto/override/Triton ordering."""

    @property
    def use_v2_model_runner(self):
        use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        if use_v2_model_runner is not None:
            return use_v2_model_runner
        if not HAS_TRITON:
            return False
        unsupported = self._get_v2_model_runner_unsupported_features()
        if unsupported:
            return False
        return True


def source_path():
    override = os.environ.get("HCU_RUNNER_SELECTION_SOURCE_ROOT")
    root = Path(override) if override else Path(__file__).resolve().parents[2]
    return root / REL


def load_hcu(path, original, native_validate=lambda self: None):
    """Extract actual functions, with only external import boundaries replaced."""
    tree = ast.parse(Path(path).read_text())
    names = (
        "_require_hcu_pcp_attribute", "_hy4_pcp_topology",
        "_require_hy4_pcp8_contract", "is_glm53_pcp", "_is_glm53_mtp_pcp",
        "_require_glm53_pcp_contract", "_require_glm53_mtp_pcp_contract",
        "_require_mrv2_pcp_contract", "_validate_hcu_pcp_scope",
        "hcu_get_v2_model_runner_unsupported_features",
        "hcu_validate_v2_model_runner",
    )
    nodes = []
    for name in names:
        matches = [n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == name]
        assert len(matches) == 1, name
        nodes.extend(matches)
    real_import = builtins.__import__

    def boundary_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_hcu.v1.dsa_cache_layer_split":
            return NS(is_hcu_dsa_mla_model=lambda model: False)
        return real_import(name, globals, locals, fromlist, level)

    class PatchCompatibilityError(RuntimeError):
        pass

    namespace = dict(
        __builtins__=dict(vars(builtins), __import__=boundary_import),
        functools=functools,
        get_v2_unsupported_features=original,
        validate_v2_model_runner=native_validate,
        PatchCompatibilityError=PatchCompatibilityError,
        get_hcu_config=lambda config: config.hcu_features,
    )
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def config_for(cls=PortableConfig, pcp=8):
    config = cls()
    config.parallel_config = NS(
        prefill_context_parallel_size=pcp, tensor_parallel_size=1,
        pipeline_parallel_size=1, decode_context_parallel_size=1,
        data_parallel_size=1, enable_expert_parallel=True, enable_eplb=False,
    )
    config.model_config = NS(
        architectures=["GlmMoeDsaForCausalLM"], use_mla=True,
        is_hybrid=False, enforce_eager=True, is_multimodal_model=False,
    )
    config.cache_config = NS(kv_offloading_size=None)
    config.speculative_config = None
    config.lora_config = None
    config.kv_transfer_config = None
    config.hcu_features = NS(enable_lightly_cp=False, enable_multi_layers_mtp=False)
    config.native_reasons = []
    config.native_calls = 0
    return config


def native_collector(self):
    self.native_calls += 1
    return self.native_reasons


@pytest.fixture
def env_case(monkeypatch):
    def set_case(value, triton=True):
        if value is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, value)
        monkeypatch.setitem(globals(), "HAS_TRITON", triton)
    return set_case


def make_config(pcp=8, original=native_collector):
    namespace = load_hcu(source_path(), original)
    cls = type("PropertyConfig", (PortableConfig,), {
        "_get_v2_model_runner_unsupported_features":
            namespace["hcu_get_v2_model_runner_unsupported_features"],
        "_validate_v2_model_runner": namespace["hcu_validate_v2_model_runner"],
    })
    assert isinstance(PortableConfig.use_v2_model_runner, property)
    return config_for(cls, pcp), namespace


@pytest.mark.parametrize("value", [None, "1", "0"])
@pytest.mark.parametrize("pcp", [1, 8])
@pytest.mark.parametrize("reasons", [[], ["heterogeneous-vocabulary draft models"],
                                    ["speculative method 'medusa'"],
                                    ["dual batch overlap"]])
def test_property_selects_without_reentry_or_environment_write(env_case, value, pcp,
                                                              reasons):
    env_case(value)
    config, _ = make_config(pcp)
    config.native_reasons = list(reasons)
    before = dict(os.environ)
    expected = (not reasons) if value is None else value == "1"
    assert config.use_v2_model_runner is expected
    assert config.native_calls == (1 if value is None else 0)
    assert dict(os.environ) == before


@pytest.mark.parametrize("value", [None, "1", "0"])
@pytest.mark.parametrize("pcp", [1, 8])
def test_no_triton_keeps_native_fallback_and_override_order(env_case, value, pcp):
    env_case(value, triton=False)
    config, _ = make_config(pcp)
    assert config.use_v2_model_runner is (value == "1")
    assert config.native_calls == 0


@pytest.mark.parametrize("pcp", [1, 8])
@pytest.mark.parametrize("reasons", [[], ["prefill context parallelism"],
    ["pipeline parallelism with external_launcher", "speculative method 'suffix'",
     "parallel drafting for EAGLE speculative decoding",
     "heterogeneous-vocabulary draft models", "dual batch overlap", "future reason"]])
def test_callback_preserves_native_list_identity_order_and_unknown_reasons(
        env_case, pcp, reasons):
    env_case("1")
    config, _ = make_config(pcp)
    config.native_reasons = list(reasons)
    before = list(config.native_reasons)
    assert config._get_v2_model_runner_unsupported_features() is config.native_reasons
    assert config.native_reasons == before
    assert config.native_calls == 1


@pytest.mark.parametrize("failure", [ValueError("native failure"),
                                    RuntimeError("native failure"),
                                    AttributeError("native failure")])
def test_native_exceptions_propagate_once_then_next_access_succeeds(env_case, failure):
    env_case(None)
    state = {"calls": 0}

    def original(config):
        state["calls"] += 1
        if state["calls"] == 1:
            raise failure
        return []

    config, _ = make_config(original=original)
    with pytest.raises(type(failure)) as caught:
        _ = config.use_v2_model_runner
    assert caught.value is failure
    assert config.use_v2_model_runner is True
    assert state["calls"] == 2


def test_callback_never_enters_full_scope_validation(env_case):
    env_case(None)
    config, namespace = make_config()

    def forbidden(config):
        raise AssertionError("selector callback entered scope validation")

    namespace["_validate_hcu_pcp_scope"] = forbidden
    assert config.use_v2_model_runner is True
    assert config.native_calls == 1


def test_native_guarded_reentry_is_not_amplified(env_case):
    env_case(None)
    calls = []

    def native(config):
        calls.append(config)
        if len(calls) == 1:
            assert config.use_v2_model_runner is True
        return []

    config, _ = make_config(original=native)
    assert config.use_v2_model_runner is True
    assert calls == [config, config]


@pytest.mark.parametrize("value", [None, "1"])
def test_direct_supported_pcp_guard_uses_property_and_terminates(env_case, value):
    env_case(value)
    config, namespace = make_config()
    assert namespace["_validate_hcu_pcp_scope"](config) is True


def test_direct_explicit_v1_pcp_guard_still_rejects(env_case):
    env_case("0")
    config, namespace = make_config()
    with pytest.raises(ValueError, match="HCU PCP requires Model Runner V2"):
        namespace["_require_mrv2_pcp_contract"](config)
    assert config.native_calls == 0


@pytest.mark.parametrize("value", [None, "1", "0"])
def test_non_pcp_scope_does_not_read_selector(env_case, value):
    env_case(value)
    config, namespace = make_config(pcp=1)
    assert namespace["_validate_hcu_pcp_scope"](config) is False
    assert config.native_calls == 0
