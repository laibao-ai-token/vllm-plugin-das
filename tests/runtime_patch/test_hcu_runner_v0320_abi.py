# SPDX-License-Identifier: Apache-2.0
"""v0320 shared runner ABI at CPU-only fake superclass boundaries.

AST-extract actual HCU methods, without importing vllm, torch, HCU, or a core
checkout. HCU_RUNNER_ABI_SOURCE_ROOT optionally selects another source image.
These tests prove Python argument/return/control-flow behavior, not GPU kernels,
real model outputs, memory sizing, or distributed collectives.
"""

import ast
import copy
import inspect
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest


METHODS = (
    "prepare_inputs", "profile_run", "execute_model", "_dummy_run",
    "_dummy_run_with_phase", "prepare_dummy_attn",
)
ROOT = Path(os.environ.get("HCU_RUNNER_ABI_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
SOURCE = ROOT / "vllm_hcu/v1/hcu_model_runner_v2.py"


class BoundaryFailure(RuntimeError):
    pass


def load_runner(monkeypatch):
    """Compile unmodified method bodies against an observable strict base."""
    events = []
    token = object()
    batch = NS(is_prefilling_np=object(), seq_lens_cpu=token,
               num_computed_tokens_cpu=token)
    phase = NS(value=None)

    @contextmanager
    def phase_scope(value=None):
        old = phase.value
        phase.value = value
        events.append(("scope_enter", value))
        try:
            yield
        finally:
            phase.value = old
            events.append(("scope_exit", old))

    def set_phase(value):
        phase.value = value
        events.append(("phase", value))

    class Base:
        def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc,
                           num_active_loras: int):
            events.append(("prepare", scheduler_output, batch_req_state,
                           batch_desc, num_active_loras))
            if getattr(self, "fail_prepare", False):
                raise BoundaryFailure("prepare")
            return batch

        def profile_run(self, randomize_inputs: bool = False):
            call = ("profile", randomize_inputs, self.max_num_tokens,
                    self.speculator, getattr(self, "_hcu_profile_mtp_full_batch", False))
            events.append(call)
            self.profile_calls = getattr(self, "profile_calls", 0) + 1
            if self.profile_calls == getattr(self, "fail_profile_at", None):
                raise BoundaryFailure("profile")
            return token

        def execute_model(self, scheduler_output, intermediate_tensors=None,
                          dummy_run=False, skip_attn_for_dummy_run=False,
                          is_profile=False, context_len=0,
                          valid_dummy_state_slots=False, randomize_inputs=False):
            events.append(("execute", scheduler_output, intermediate_tensors,
                           dummy_run, skip_attn_for_dummy_run, is_profile,
                           context_len, valid_dummy_state_slots, randomize_inputs,
                           phase.value))
            if getattr(self, "fail_execute", False):
                raise BoundaryFailure("execute")
            return token

        def _dummy_run(self, num_tokens, *args, skip_attn=False,
                       uniform_decode=False, context_len=0, skip_eplb=False,
                       is_profile=False, valid_dummy_state_slots=False,
                       randomize_inputs=False, **kwargs):
            events.append(("dummy", num_tokens, args, skip_attn, uniform_decode,
                           context_len, skip_eplb, is_profile,
                           valid_dummy_state_slots, randomize_inputs, kwargs,
                           self.model, phase.value))
            if getattr(self, "_hcu_profile_mtp_full_batch", False):
                stub_output = self.model(input_ids=NS(shape=(num_tokens,)))
                events.append(("stub", stub_output, self.model.marker))
            if getattr(self, "fail_dummy", False):
                raise BoundaryFailure("dummy")
            return token

        def prepare_dummy_attn(self, input_batch, valid_state_slots=False):
            events.append(("dummy_attn", input_batch, valid_state_slots,
                           self.pcp_manager))
            if getattr(self, "fail_attn", False):
                raise BoundaryFailure("attn")
            return token

    fake_module = ModuleType("vllm_hcu.v1.glm53_pcp")
    fake_module.bind_glm53_pcp_batch = lambda manager, value: events.append(("bind", manager, value))
    fake_module.glm53_pcp_step_scope = nullcontext
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "HcuGPUModelRunnerV2")
    methods = [copy.deepcopy(n) for n in cls.body
               if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    assert len(methods) == len(METHODS)
    for method in methods:
        method.decorator_list = []
    cls = ast.ClassDef(name="HcuGPUModelRunnerV2", bases=[ast.Name(id="Base", ctx=ast.Load())],
                       keywords=[], body=methods, decorator_list=[])
    compiled = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {
        "Base": Base, "nullcontext": nullcontext,
        "is_deepseek_v4": lambda cfg: cfg.dsv4,
        "is_glm53_pcp": lambda cfg: cfg.glm53,
        "is_step5_config": lambda cfg: False,
        "_uses_global_pcp_batch": lambda cfg: cfg.global_batch,
        "get_hcu_config": lambda cfg: NS(deepep_auto=cfg.deepep_auto),
        "set_deepep_auto_request_phase": set_phase,
        "deepep_auto_request_phase_scope": phase_scope,
        "_deepep_auto_dummy_phase_scope": phase_scope,
        "replicated_mtp_batch_scope": nullcontext,
        "_DEEPEP_AUTO_DUMMY_PHASE": NS(get=lambda: phase.value),
        "torch": NS(bfloat16=object(), zeros=lambda *shape, **kw: NS(shape=shape, **kw)),
    }
    exec(compile(compiled, str(SOURCE), "exec"), namespace)
    runner = namespace["HcuGPUModelRunnerV2"]()
    runner.vllm_config = NS(parallel_config=NS(prefill_context_parallel_size=1),
                            dsv4=False, glm53=False, global_batch=False, deepep_auto=True)
    runner._hsl_enabled = False
    runner.pcp_manager = None
    runner.max_num_tokens = 128
    runner.max_num_reqs = 4
    runner.decode_query_len = 1
    runner.speculator = None
    runner.model = NS(marker=object())
    runner.model_config = NS(get_hidden_size=lambda: 16)
    runner.dtype = object()
    runner.device = "cpu"
    return runner, events, token, batch, phase


@pytest.mark.parametrize("num_active_loras", [0, 7])
@pytest.mark.parametrize("mode", ["pcp1", "host_mirror", "partitioned", "glm5_global", "v4_global", "v41_global"])
def test_prepare_forwards_exact_lora_and_preserves_batch_phase(monkeypatch, num_active_loras, mode):
    runner, events, _, batch, phase = load_runner(monkeypatch)
    config = runner.vllm_config
    config.dsv4 = mode in ("v4_global", "v41_global")
    config.global_batch = mode in ("glm5_global", "v4_global", "v41_global")
    runner._hsl_enabled = mode == "host_mirror"
    global_phase = object()
    if mode not in ("pcp1", "host_mirror"):
        runner.pcp_manager = NS(_global_batch=NS(is_prefilling_np=global_phase))
    args = (object(), object(), object())
    result = runner.prepare_inputs(*args, num_active_loras)
    assert result is batch
    call = events[0]
    assert call[0] == "prepare"
    assert all(a is b for a, b in zip(call[1:], (*args, num_active_loras)))
    expected_phase = global_phase if config.dsv4 else batch.is_prefilling_np
    assert phase.value is expected_phase
    assert events[1] == ("phase", expected_phase)
    if runner._hsl_enabled:
        assert batch.seq_lens_cpu is None and batch.num_computed_tokens_cpu is None
        assert runner._hsl_batch is batch
    else:
        assert batch.seq_lens_cpu is not None and batch.num_computed_tokens_cpu is not None
    if config.global_batch:
        assert events[2] == ("bind", runner.pcp_manager, batch)
    assert len(events) == (3 if config.global_batch else 2)


def test_prepare_exception_has_no_phase_or_bind_side_effect(monkeypatch):
    runner, events, _, _, phase = load_runner(monkeypatch)
    runner.fail_prepare = True
    with pytest.raises(BoundaryFailure, match="prepare"):
        runner.prepare_inputs(object(), object(), object(), 3)
    assert len(events) == 1 and phase.value is None


@pytest.mark.parametrize("randomize", [False, True, None], ids=["false", "true", "default"])
@pytest.mark.parametrize("mode", ["pcp1", "global", "partitioned", "mtp"])
def test_profile_all_super_sites_forward_and_preserve_returns(monkeypatch, randomize, mode):
    runner, events, token, _, _ = load_runner(monkeypatch)
    runner.vllm_config.parallel_config.prefill_context_parallel_size = 1 if mode == "pcp1" else 4
    runner.vllm_config.global_batch = mode == "global"
    speculator = object() if mode == "mtp" else None
    runner.speculator = speculator
    runner._hcu_profile_mtp_full_batch = False
    result = runner.profile_run() if randomize is None else runner.profile_run(randomize_inputs=randomize)
    value = False if randomize is None else randomize
    if mode in ("pcp1", "global"):
        assert result is token
        assert events == [("profile", value, 128, speculator, False)]
    else:
        assert result is None  # Existing partitioned branch deliberately discards super return.
        assert events[0] == ("profile", value, 44, None, False)
        if mode == "mtp":
            assert events[1] == ("profile", value, 128, speculator, True)
        assert len(events) == (2 if mode == "mtp" else 1)
    assert all(event[1] is value for event in events)
    assert runner.max_num_tokens == 128
    assert runner.speculator is speculator
    assert runner._hcu_profile_mtp_full_batch is False


@pytest.mark.parametrize("randomize", [False, True])
@pytest.mark.parametrize("mode,fail_at", [("pcp1", 1), ("global", 1), ("partitioned", 1), ("mtp", 1), ("mtp", 2)])
def test_profile_exception_restores_budget_speculator_and_fullbatch(monkeypatch, randomize, mode, fail_at):
    runner, events, _, _, _ = load_runner(monkeypatch)
    runner.vllm_config.parallel_config.prefill_context_parallel_size = 1 if mode == "pcp1" else 4
    runner.vllm_config.global_batch = mode == "global"
    speculator = object() if mode == "mtp" else None
    runner.speculator = speculator
    runner._hcu_profile_mtp_full_batch = False
    runner.fail_profile_at = fail_at
    with pytest.raises(BoundaryFailure, match="profile"):
        runner.profile_run(randomize_inputs=randomize)
    assert len(events) == fail_at
    assert all(event[1] is randomize for event in events)
    assert runner.max_num_tokens == 128
    assert runner.speculator is speculator
    assert runner._hcu_profile_mtp_full_batch is False


@pytest.mark.parametrize("randomize", [False, True])
@pytest.mark.parametrize("global_batch", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_execute_override_preserves_core_arguments_return_and_scope(monkeypatch, randomize, global_batch, fail):
    runner, events, token, _, phase = load_runner(monkeypatch)
    runner.vllm_config.global_batch = global_batch
    runner.vllm_config.glm53 = global_batch
    runner.fail_execute = fail
    scheduler, intermediate = object(), object()
    kwargs = dict(intermediate_tensors=intermediate, dummy_run=True,
                  skip_attn_for_dummy_run=True, is_profile=True, context_len=17,
                  valid_dummy_state_slots=True, randomize_inputs=randomize)
    if fail:
        with pytest.raises(BoundaryFailure, match="execute"):
            runner.execute_model(scheduler, **kwargs)
    else:
        assert runner.execute_model(scheduler, **kwargs) is token
    call = next(event for event in events if event[0] == "execute")
    assert call[1:10] == (scheduler, intermediate, True, True, True, 17, True, randomize, None)
    assert call[8] is randomize
    assert phase.value is None
    assert events[-1] == ("scope_exit", None)


@pytest.mark.parametrize("randomize", [False, True])
@pytest.mark.parametrize("full_batch", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_dummy_override_forwards_all_core_keywords_and_restores_model(monkeypatch, randomize, full_batch, fail):
    runner, events, token, _, phase = load_runner(monkeypatch)
    runner._hcu_profile_mtp_full_batch = full_batch
    runner.fail_dummy = fail
    real_model = runner.model
    kwargs = dict(skip_attn=True, uniform_decode=True, context_len=19,
                  skip_eplb=True, is_profile=True, valid_dummy_state_slots=True,
                  randomize_inputs=randomize, future_keyword=token)
    if fail:
        with pytest.raises(BoundaryFailure, match="dummy"):
            runner._dummy_run(23, token, **kwargs)
    else:
        assert runner._dummy_run(23, token, **kwargs) is token
    call = next(event for event in events if event[0] == "dummy")
    assert call[1:11] == (23, (token,), True, True, 19, True, True, True,
                           randomize, {"future_keyword": token})
    assert call[9] is randomize
    assert call[12] is False
    assert (call[11] is real_model) is (not full_batch)
    if full_batch:
        stub = next(event for event in events if event[0] == "stub")
        assert stub[1].shape == (23, 16) and stub[1].device == "cpu"
        assert stub[2] is real_model.marker
    assert runner.model is real_model and phase.value is None


@pytest.mark.parametrize("global_batch", [False, True])
@pytest.mark.parametrize("valid_slots", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_dummy_attn_global_manager_restoration(monkeypatch, global_batch, valid_slots, fail):
    runner, events, token, batch, _ = load_runner(monkeypatch)
    runner.vllm_config.global_batch = global_batch
    manager = NS() if global_batch else None
    runner.pcp_manager = manager
    runner.fail_attn = fail
    if fail:
        with pytest.raises(BoundaryFailure, match="attn"):
            runner.prepare_dummy_attn(batch, valid_slots)
    else:
        assert runner.prepare_dummy_attn(batch, valid_slots) is token
    assert events == [("dummy_attn", batch, valid_slots, None)]
    assert runner.pcp_manager is manager


def test_fixed_core_required_prepare_signature(monkeypatch):
    runner, _, _, _, _ = load_runner(monkeypatch)
    signature = inspect.signature(runner.prepare_inputs)
    assert list(signature.parameters) == ["scheduler_output", "batch_req_state", "batch_desc", "num_active_loras"]
    assert signature.parameters["num_active_loras"].default is inspect.Parameter.empty
    assert signature.parameters["num_active_loras"].annotation is int
    with pytest.raises(TypeError, match="num_active_loras"):
        runner.prepare_inputs(object(), object(), object())
