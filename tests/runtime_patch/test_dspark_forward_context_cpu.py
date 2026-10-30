# SPDX-License-Identifier: Apache-2.0
"""Native _run_model + HCU context body before real AGRS source dispatch, on CPU."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from test_dspark_producer_cpu import (
    route, source_nodes, load, module, ROOT, HCU, CORE, pytest, torch,
)


def context_runner(route, monkeypatch):
    original_context = SimpleNamespace(dp_metadata=SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [2] * 8))
    active = [original_context]
    route.agrs_ns['get_forward_context'] = lambda: active[0]
    @contextmanager
    def override(context):
        previous = active[0]
        active[0] = context
        try: yield
        finally: active[0] = previous
    modes = SimpleNamespace(NONE=0, FULL=1)
    from dataclasses import dataclass
    metadata_created = []
    forward_module = SimpleNamespace(
        CUDAGraphMode=modes, track_batchsize=False, torch=torch,
        current_platform=SimpleNamespace(set_additional_forward_context=lambda **kwargs: {}),
        create_forward_context=lambda attn, config, dp, *a, **kw: SimpleNamespace(dp_metadata=dp, config=config),
        override_forward_context=override,
    )
    core_ns = dict(contextmanager=contextmanager, dataclass=dataclass, torch=torch,
                   CUDAGraphMode=modes, track_batchsize=False,
                   current_platform=forward_module.current_platform,
                   create_forward_context=forward_module.create_forward_context,
                   override_forward_context=override)
    source_nodes(CORE / 'vllm/forward_context.py',
                 {'_compute_sp_num_tokens': None, 'DPMetadata': None, 'set_forward_context': None}, core_ns)
    real_metadata = core_ns['DPMetadata']
    original_core_context = core_ns['set_forward_context']
    forward_module.DPMetadata = real_metadata
    ns = {'contextmanager': contextmanager}
    source_nodes(HCU / 'vllm_hcu/forward_context_runtime.py', {'set_forward_context': None}, ns)
    def set_forward_context(attn, config, **kwargs):
        # Same selected route as baseline patch_forward_context: fixed LL uses
        # HCU runtime; AGRS+deepep_auto=False uses the real native original_set.
        if config.parallel_config.all2all_backend == 'deepep_low_latency':
            return ns['set_forward_context'](forward_module, attn, config, **kwargs)
        return original_core_context(attn, config, **kwargs)
    native_ns = dict(torch=torch, CUDAGraphMode=modes, BatchDescriptor=SimpleNamespace,
                     set_forward_context=set_forward_context)
    path = CORE / 'vllm/v1/worker/gpu/spec_decode/dflash/speculator.py'
    source_nodes(path, {'DFlashSpeculator': {'_run_model'}}, native_ns)
    target = module(monkeypatch, 'vllm.v1.worker.gpu.spec_decode.dflash.speculator',
                    DFlashSpeculator=native_ns['DFlashSpeculator'], CUDAGraphMode=modes)
    callback = load(monkeypatch, 'vllm_hcu.patch.worker.core_fix.patch_dspark_draft_context',
                    ROOT / 'vllm_hcu/patch/worker/core_fix/patch_dspark_draft_context.py')
    return target, callback, active, original_context


def test_native_ll_context_fails_without_early_scope_then_producer_repairs(route, monkeypatch):
    route.feature.set_hcu_config(route.config, deepep_auto=False)
    assert route.config.parallel_config._vllm_hcu_deepep_auto is False
    route.config.parallel_config.is_moe_model = True
    model = route.runtime.load_with_draft_scope(route.native.load_dspark_model, route.target_model, route.config)
    target, callback, active, original_context = context_runner(route, monkeypatch)
    runner = target.DFlashSpeculator()
    runner._speculator_name = 'DSpark'
    runner.vllm_config = route.config
    runner.model = model
    runner.input_buffers = SimpleNamespace(input_ids=torch.ones(64, dtype=torch.int64), positions=torch.arange(64))
    counts = torch.tensor([2] * 8)
    # The actual HCU LL context body leaves dp_metadata=None. Merely wrapping
    # model.forward enters too late; actual AGRS _get_sizes asserts here.
    with pytest.raises(AssertionError): runner._run_model(2, None, None, counts)
    assert active[0] is original_context
    assert callback.apply_to_module(target) is True
    assert callback.apply_to_module(target) is False
    output = runner._run_model(2, None, None, counts)
    assert output.shape == (2, 4)
    assert any(event[0] == 'padded_counts' and event[1] is counts for event in route.events)
    assert runner.vllm_config is route.config
    assert route.ps.get_ep_group() is route.ep
    assert active[0] is original_context
    with pytest.raises(ValueError, match='eager'):
        runner._run_model(2, None, None, counts, target.CUDAGraphMode.FULL)
    assert runner.vllm_config is route.config and active[0] is original_context


def test_late_loader_alias_is_repaired_and_unknown_alias_rejected(route, monkeypatch):
    original = route.native.load_dspark_model
    caller = module(monkeypatch, route.callback.CALLER_MODULE, load_dspark_model=original)
    assert route.callback.apply_to_module(route.native)
    assert caller.load_dspark_model is route.native.load_dspark_model
    caller.load_dspark_model = original
    assert route.callback.apply_to_module(route.native) is False
    assert caller.load_dspark_model is route.native.load_dspark_model
    caller.load_dspark_model = lambda *a: None
    with pytest.raises(route.common.PatchCompatibilityError, match='alias'):
        route.callback.apply_to_module(route.native)


def test_nonowning_group_and_communicator_lifecycle_and_host_thread(route):
    model = route.runtime.load_with_draft_scope(route.native.load_dspark_model, route.target_model, route.config)
    scope = model._hcu_dspark_draft_scope
    for owner, names in [(scope.draft_ep, ('destroy',)),
                         (scope.draft_ep.device_communicator, ('destroy', 'checkpoint_prepare', 'checkpoint_restore', 'suspend', 'resume'))]:
        for name in names:
            with pytest.raises(RuntimeError, match='borrow'):
                getattr(owner, name)()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(model, input_ids=torch.ones(2, dtype=torch.int64))
        with pytest.raises(RuntimeError, match='owning worker thread'):
            future.result()
    assert route.ps.get_ep_group() is route.ep
    assert route.ep.device_communicator.all2all_manager is route.manager
