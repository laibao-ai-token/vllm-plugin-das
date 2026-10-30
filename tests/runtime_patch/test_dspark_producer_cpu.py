# SPDX-License-Identifier: Apache-2.0
"""CPU producer contracts. PG collectives, model loading and quantization are stubs."""
from __future__ import annotations
import ast
import copy
from contextlib import contextmanager
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Callable, cast
from pydantic.fields import FieldInfo
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

HCU = Path(os.environ.get('VLLM_HCU_SOURCE_ROOT', Path(__file__).resolve().parents[2]))
_DEFAULT_CORE = next((path for path in (HCU.parent / 'vllm-community-main',
                                      HCU.parent.parent / 'vllm-community-main')
                      if (path / 'vllm/config/utils.py').is_file()),
                     HCU.parent / 'vllm-community-main')
CORE = Path(os.environ.get('VLLM_SOURCE_ROOT', _DEFAULT_CORE))
ROOT = Path(os.environ.get('HCU_DSPARK_CANDIDATE_ROOT', HCU))


def module(monkeypatch, name, **values):
    m = ModuleType(name)
    m.__path__ = []
    m.__dict__.update(values)
    monkeypatch.setitem(sys.modules, name, m)
    return m


def load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, m)
    spec.loader.exec_module(m)
    return m


def source_nodes(path, names, ns):
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(copy.deepcopy(node))
        if isinstance(node, ast.ClassDef) and node.name in names:
            node = copy.deepcopy(node)
            methods = names[node.name]
            node.bases = []
            if methods is not None:
                node.body = [m for m in node.body if getattr(m, 'name', '') in methods]
            nodes.append(node)
    future = ast.parse('from __future__ import annotations').body[0]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), 'exec'), ns)


@dataclass
class Parallel:
    tensor_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 8
    data_parallel_rank: int = 0
    is_moe_model: bool = True
    enable_expert_parallel: bool = True
    enable_elastic_ep: bool = False
    enable_eplb: bool = False
    all2all_backend: str = 'deepep_low_latency'
    eplb_config: object = field(default_factory=lambda: SimpleNamespace(num_redundant_experts=0))

    @property
    def use_sequence_parallel_moe(self):
        return False


@dataclass
class Config:
    parallel_config: object
    speculative_config: object
    model_config: object = field(default_factory=lambda: SimpleNamespace(enforce_eager=True, get_vocab_size=lambda: 16))
    scheduler_config: object = field(default_factory=lambda: SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=64))
    kernel_config: object = field(default_factory=lambda: SimpleNamespace(moe_backend='deep_gemm'))
    additional_config: dict = field(default_factory=lambda: {'hcu': {'deepep_auto': True, 'moe_backend': 'deep_gemm', 'dspark_draft_all2all_backend': 'allgather_reducescatter'}})
    compilation_config: object = field(default_factory=lambda: SimpleNamespace(mode='none'))
    attention_config: object = field(default_factory=lambda: SimpleNamespace(backend='hcu', use_non_causal=False))
    cache_config: object = None
    load_config: object = None
    quant_config: object = None


@pytest.fixture
def route(monkeypatch):
    assert not torch.cuda.is_initialized()
    for name in ('vllm_hcu', 'vllm_hcu.patch', 'vllm_hcu.patch.worker',
                 'vllm_hcu.patch.worker.core_fix', 'vllm_hcu.patch.platform',
                 'vllm_hcu.patch.platform.core_fix', 'vllm', 'vllm.distributed',
                 'vllm.distributed.device_communicators', 'vllm.model_executor',
                 'vllm.model_executor.layers', 'vllm.model_executor.models',
                 'vllm.v1', 'vllm.v1.worker', 'vllm.v1.worker.gpu',
                 'vllm.v1.worker.gpu.spec_decode', 'vllm.v1.worker.gpu.spec_decode.eagle',
                 'vllm.v1.worker.gpu.spec_decode.dspark', 'vllm.compilation'):
        module(monkeypatch, name)
    feature = load(monkeypatch, 'vllm_hcu.patch.config', ROOT / 'vllm_hcu/patch/config.py')
    runtime = load(monkeypatch, 'vllm_hcu.dspark_runtime', ROOT / 'vllm_hcu/dspark_runtime.py')
    common = load(monkeypatch, 'vllm_hcu.patch.worker.core_fix._common', HCU / 'vllm_hcu/patch/worker/core_fix/_common.py')
    callback = load(monkeypatch, 'vllm_hcu.patch.worker.core_fix.patch_dspark_draft_scope', ROOT / 'vllm_hcu/patch/worker/core_fix/patch_dspark_draft_scope.py')
    events = []
    current = [None]
    @contextmanager
    def config_context(config):
        previous = current[0]
        current[0] = config
        try: yield
        finally: current[0] = previous
    module(monkeypatch, 'vllm.config', set_current_vllm_config=config_context)
    @contextmanager
    def tag(name):
        events.append(('tag', name))
        yield
    module(monkeypatch, 'vllm.compilation.backends', set_model_tag=tag)
    ps = module(monkeypatch, 'vllm.distributed.parallel_state')
    sys.modules['vllm.distributed'].parallel_state = ps
    group_ns = {'torch': torch}
    source_nodes(CORE / 'vllm/distributed/parallel_state.py', {'GroupCoordinator': {
        'dispatch_router_logits', 'dispatch', 'combine', 'allocate_combine_input', 'combine_into_output'}}, group_ns)
    class CPUGroup:
        def rank(self): return 0
        def size(self): return 8
    class Group(group_ns['GroupCoordinator']):
        def __init__(self, size):
            self.world_size = size
            self.rank_in_group = 0
            self.cpu_group = CPUGroup()
            self.device_group = object()
            self.device_communicator = None
        def all_gatherv(self, tensors, dim, sizes):
            events.append(('all_gatherv', self.world_size, tuple(sizes)))
            return [torch.cat([t] * self.world_size, dim=dim) for t in tensors]
        def reduce_scatterv(self, x, dim, sizes):
            events.append(('reduce_scatterv', self.world_size, tuple(sizes)))
            return x.narrow(dim, 0, sizes[0]).clone()
    dp, tp, pcp, ep = Group(8), Group(1), Group(1), Group(8)
    ps._EP = ep
    ps.get_ep_group = lambda: ps._EP
    ps.get_dp_group = lambda: dp
    ps.get_tp_group = lambda: tp
    ps.get_pcp_group = lambda: pcp
    ps.in_the_same_node_as = lambda group, source_rank: [True] * group.size()
    module(monkeypatch, 'vllm.forward_context', get_forward_context=lambda: SimpleNamespace(
        dp_metadata=SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [2] * 8)))
    ns = {'get_current_vllm_config': lambda: current[0],
          'get_ep_group': ps.get_ep_group, 'get_dp_group': ps.get_dp_group,
          'get_pcp_group': ps.get_pcp_group,
          'get_forward_context': sys.modules['vllm.forward_context'].get_forward_context,
          'torch': torch}
    source_nodes(CORE / 'vllm/distributed/device_communicators/base_device_communicator.py',
                 {'All2AllManagerBase': {'__init__'}}, ns)
    source_nodes(CORE / 'vllm/distributed/device_communicators/all2all.py',
                 {'AgRsAll2AllManager': None}, ns)
    # Restore its actual base after source extraction, without importing core.
    ag_tree = ast.parse((CORE / 'vllm/distributed/device_communicators/all2all.py').read_text())
    ag = copy.deepcopy(next(n for n in ag_tree.body if isinstance(n, ast.ClassDef) and n.name == 'AgRsAll2AllManager'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.parse('from __future__ import annotations').body[0], ag], type_ignores=[])), 'actual_agrs_source', 'exec'), ns)
    module(monkeypatch, 'vllm.distributed.device_communicators.all2all', AgRsAll2AllManager=ns['AgRsAll2AllManager'])
    comm_ns = {}
    source_nodes(CORE / 'vllm/distributed/device_communicators/cuda_communicator.py',
                 {'CudaCommunicator': {'dispatch_router_logits', 'dispatch', 'combine'}}, comm_ns)
    comm = comm_ns['CudaCommunicator']()
    target_manager = SimpleNamespace(name='target_ll')
    comm.all2all_manager = target_manager
    comm.all2all_backend = 'deepep_low_latency'
    ep.device_communicator = comm
    parallel = Parallel()
    draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type='deepseek_v41', sample_from_anchor=True), enforce_eager=True, dtype=torch.bfloat16, get_vocab_size=lambda: 16)
    spec = SimpleNamespace(method='dspark', target_parallel_config=parallel,
                           draft_parallel_config=SimpleNamespace(tensor_parallel_size=1),
                           draft_model_config=draft_model_config, num_speculative_tokens=5,
                           moe_backend='deep_gemm', attention_backend=None, kv_cache_dtype=None)
    config = Config(parallel, spec)
    class TargetModule:
        def __deepcopy__(self, memo):
            raise AssertionError('target module/weights must not be deep-copied')
    target_layer = TargetModule()
    config.compilation_config.static_forward_context = {'target.attn': target_layer}
    config.compilation_config.static_all_moe_layers = ['target.moe']
    # dataclasses.replace also needs dataclass attention/eplb objects in native loader.
    @dataclass
    class Attention:
        backend: object = None
        use_non_causal: bool = False
    @dataclass
    class EPLB:
        num_redundant_experts: int = 0
    config.attention_config = Attention()
    parallel.eplb_config = EPLB()
    pf_ns = {'get_ep_group': ps.get_ep_group, 'torch': torch,
             '_quantize_and_setup_dispatch': lambda x, quant, defer: (x, None, None)}
    source_nodes(CORE / 'vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py',
                 {'MoEPrepareAndFinalizeNaiveDPEPModular': {'prepare'}}, pf_ns)
    pf = pf_ns['MoEPrepareAndFinalizeNaiveDPEPModular']()
    pf._lora_context = None
    pf.is_sequence_parallel = False
    models = []
    def get_model(*, vllm_config, model_config):
        events.append(('get_model', ps.get_ep_group(), vllm_config))
        model = torch.nn.Module()
        model.model = SimpleNamespace()
        model.moe = SimpleNamespace()
        model.explicit_manager = runtime.get_dspark_all2all_manager(model.moe)
        assert not hasattr(model.moe, '_hcu_dspark_all2all_manager')
        # CPU boundary for D-owned CT consumption, not producer behavior.
        model.moe._hcu_dspark_all2all_manager = model.explicit_manager
        vllm_config.compilation_config.static_forward_context['draft.attn'] = model
        vllm_config.compilation_config.static_all_moe_layers.append('draft.moe')
        def forward(input_ids, positions=None, inputs_embeds=None):
            events.append(('forward', ps.get_ep_group()))
            if getattr(model, 'fail', False): raise RuntimeError('model failed')
            x = torch.ones((input_ids.shape[0], 4))
            topk = torch.ones((input_ids.shape[0], 1))
            def dispatch_combine():
                gathered, _, _, _, _ = pf.prepare(x, topk, topk.long(), 128, None, False, None)
                return ps.get_ep_group().combine(gathered)
            metadata = ns['get_forward_context']().dp_metadata
            if metadata is not None and hasattr(metadata, 'sp_local_sizes'):
                events.append(('padded_counts', metadata.num_tokens_across_dp_cpu))
                # D-owned MoERunner boundary: execute the actual native metadata
                # context manager while leaving the MoE math itself a CPU stub.
                with metadata.sp_local_sizes(1, 1, True):
                    result = dispatch_combine()
                assert metadata.local_sizes is None
                events.append(('local_sizes_restored',))
                return result
            return dispatch_combine()
        model.forward = forward
        models.append(model)
        return model
    module(monkeypatch, 'vllm.model_executor.model_loader', get_model=get_model)
    module(monkeypatch, 'vllm.model_executor.models.qwen3_dflash', dflash_has_any_non_causal=lambda cfg: False)
    module(monkeypatch, 'vllm.model_executor.models.utils', get_draft_quant_config=lambda cfg: 'draft_quant')
    module(monkeypatch, 'vllm.v1.worker.gpu.spec_decode.eagle.utils',
           _should_share=lambda *a: False, get_target_lm_head=lambda *a: None,
           maybe_share_target_embed=lambda *a: None)
    # Integration boundary: execute the REAL fixed-core copy helpers, not
    # dataclasses.replace; install the independently reviewed companion.
    copy_ns = dict(MISSING=MISSING, field=field, fields=fields, is_dataclass=is_dataclass,
                   FieldInfo=FieldInfo, cast=cast, Callable=Callable, Any=Any,
                   logger=SimpleNamespace(warning_once=lambda *a: None))
    source_nodes(CORE / 'vllm/config/utils.py',
                 {'get_field': None, 'is_init_field': None, 'replace': None}, copy_ns)
    copy_utils = module(monkeypatch, 'vllm.config.utils', replace=copy_ns['replace'])
    sys.modules['vllm.config'].replace = copy_utils.replace
    module(monkeypatch, 'vllm.config.parallel', ParallelConfig=Parallel)
    load(monkeypatch, 'vllm_hcu.patch.runtime_state', HCU / 'vllm_hcu/patch/runtime_state.py')
    load(monkeypatch, 'vllm_hcu.patch.platform.core_fix._common', HCU / 'vllm_hcu/patch/platform/core_fix/_common.py')
    copy_patch = load(monkeypatch, 'vllm_hcu.patch.platform.core_fix.patch_config_replace',
                      ROOT / 'vllm_hcu/patch/platform/core_fix/patch_config_replace.py')
    copy_patch.apply_to_module(copy_utils)
    feature.bind_hcu_parallel_aliases(config)
    assert all(hasattr(config.parallel_config, name) for name in copy_patch._PRIVATE_FIELDS)
    native_ns = dict(replace=copy_utils.replace, logger=SimpleNamespace(info_once=lambda *a: None, warning_once=lambda *a: None),
                     dspark_draft_supports_eplb=lambda cfg: False,
                     get_pp_safe_draft_load_config=lambda cfg: cfg, get_draft_load_config=lambda cfg: cfg.load_config)
    native_path = CORE / 'vllm/v1/worker/gpu/spec_decode/dspark/utils.py'
    source_nodes(native_path, {'_resolve_dspark_attention_backend': None, '_get_dspark_parallel_config': None,
                              'load_dspark_model': None}, native_ns)
    native = module(monkeypatch, callback.TARGET_MODULE, load_dspark_model=native_ns['load_dspark_model'])
    return SimpleNamespace(runtime=runtime, feature=feature, callback=callback, common=common,
                           config=config, native=native, ps=ps, ep=ep, manager=target_manager, agrs_ns=ns,
                           current=current, events=events, models=models,
                           target_model=SimpleNamespace(model=SimpleNamespace()))


def test_real_callback_native_loader_and_runtime_agrs_group(route):
    assert route.callback.apply_to_module(route.native) is True
    assert route.callback.apply_to_module(route.native) is False
    model = route.native.load_dspark_model(route.target_model, route.config)
    scope = model._hcu_dspark_draft_scope
    assert route.ps.get_ep_group() is route.ep
    assert route.ep.device_communicator.all2all_manager is route.manager
    assert scope.manager is not route.manager
    assert model.explicit_manager is scope.manager
    assert model.moe._hcu_dspark_all2all_manager is scope.manager
    assert scope.draft_ep.device_group is route.ep.device_group  # Borrowed, not a new PG.
    assert scope.draft_ep.device_communicator is not route.ep.device_communicator
    assert scope.config.parallel_config.data_parallel_size == 8
    assert scope.config.parallel_config.all2all_backend == 'allgather_reducescatter'
    assert route.config.parallel_config.all2all_backend == 'deepep_low_latency'
    assert route.config.additional_config['hcu']['deepep_auto'] is True
    assert route.feature.get_hcu_config(scope.config).deepep_auto is False
    assert scope.config.compilation_config is not route.config.compilation_config
    assert scope.config.compilation_config.static_forward_context is route.config.compilation_config.static_forward_context
    assert route.config.compilation_config.static_forward_context['draft.attn'] is model
    assert route.config.compilation_config.static_all_moe_layers == ['target.moe', 'draft.moe']
    out = model(input_ids=torch.ones(2, dtype=torch.int64))
    assert out.shape == (2, 4)
    assert any(e[0] == 'all_gatherv' for e in route.events)
    assert any(e[0] == 'reduce_scatterv' for e in route.events)
    assert route.ps.get_ep_group() is route.ep and route.current[0] is None
    assert route.runtime.get_dspark_all2all_manager(SimpleNamespace()) is None
    assert not torch.cuda.is_initialized()


def test_failure_and_nested_scope_restore_exact_target(route):
    model = route.runtime.load_with_draft_scope(route.native.load_dspark_model, route.target_model, route.config)
    scope = model._hcu_dspark_draft_scope
    with scope.activate():
        assert route.runtime.get_dspark_token_capacity() is scope.capacity
        with scope.activate(): assert route.ps.get_ep_group() is scope.draft_ep
        assert route.ps.get_ep_group() is scope.draft_ep
    assert route.runtime.get_dspark_token_capacity() is None
    model.fail = True
    with pytest.raises(RuntimeError, match='model failed'):
        model(input_ids=torch.ones(2, dtype=torch.int64))
    assert route.ps.get_ep_group() is route.ep
    assert route.ep.device_communicator.all2all_manager is route.manager
    assert route.current[0] is None
    with pytest.raises(ValueError, match='exceeds checked'):
        model(input_ids=torch.ones(65, dtype=torch.int64))
    assert route.ps.get_ep_group() is route.ep


@pytest.mark.parametrize('anchor,query', [(True, 20), (False, 24)])
def test_dspark5_token_capacity_and_no_scheduler_mutation(route, anchor, query):
    route.config.speculative_config.draft_model_config.hf_config.sample_from_anchor = anchor
    result = route.runtime.token_capacity(route.config)
    assert result.query_tokens_per_rank == query
    assert result.verify_tokens_per_rank == 24
    assert result.capacity_per_rank == 64
    assert route.config.scheduler_config.max_num_batched_tokens == 64
    route.config.scheduler_config.max_num_batched_tokens = 23
    with pytest.raises(ValueError, match='insufficient'):
        route.runtime.token_capacity(route.config)
    assert route.config.scheduler_config.max_num_batched_tokens == 23


@pytest.mark.parametrize('change', ['pcp', 'tp', 'pp', 'graph', 'ep', 'elastic', 'backend'])
def test_unimplemented_topology_is_not_claimed_supported(route, change):
    if change == 'pcp': route.config.speculative_config.target_parallel_config.prefill_context_parallel_size = 8
    if change == 'tp': route.config.parallel_config.tensor_parallel_size = 2
    if change == 'pp': route.config.parallel_config.pipeline_parallel_size = 2
    if change == 'graph': route.config.model_config.enforce_eager = False
    if change == 'ep': route.config.parallel_config.enable_expert_parallel = False
    if change == 'elastic': route.config.parallel_config.enable_elastic_ep = True
    if change == 'backend': route.config.speculative_config.moe_backend = 'triton'
    with pytest.raises(ValueError): route.runtime.draft_config(route.config)
    assert route.ps.get_ep_group() is route.ep


def test_sidecar_roundtrip_and_explicit_opt_in(route):
    f = route.feature.HcuFeatureConfig(dspark_draft_all2all_backend='allgather_reducescatter')
    assert route.feature.HcuFeatureConfig.from_mapping(f.to_dict()) == f
    with pytest.raises(ValueError): route.feature.HcuFeatureConfig(dspark_draft_all2all_backend='deepep_low_latency')
    route.config.additional_config = {'hcu': {}}
    seen = []
    def original(target_model, vllm_config):
        seen.append(vllm_config)
        return 'unchanged'
    assert route.runtime.load_with_draft_scope(original, route.target_model, route.config) == 'unchanged'
    assert seen == [route.config] and route.ps.get_ep_group() is route.ep


def test_callback_signature_drift_fails_before_replacement(route):
    def drift(target_model, vllm_config, unexpected=None): pass
    route.native.load_dspark_model = drift
    with pytest.raises(route.common.PatchCompatibilityError): route.callback.apply_to_module(route.native)
    assert route.native.load_dspark_model is drift


def test_bound_manager_never_silently_replaced(route):
    model = route.runtime.load_with_draft_scope(route.native.load_dspark_model, route.target_model, route.config)
    with model._hcu_dspark_draft_scope.activate():
        with pytest.raises(RuntimeError, match='different draft scope'):
            route.runtime.get_dspark_all2all_manager(SimpleNamespace(_hcu_dspark_all2all_manager=object()))
    assert route.ps.get_ep_group() is route.ep


@pytest.mark.parametrize('phase', ['manager', 'loader'])
def test_constructor_failure_restores_global_state(route, monkeypatch, phase):
    def failed_manager(*args, **kwargs):
        raise RuntimeError('manager construction failed')
    def failed_loader(target_model, vllm_config):
        raise RuntimeError('loader failed')
    if phase == 'manager':
        monkeypatch.setattr(sys.modules['vllm.distributed.device_communicators.all2all'],
                            'AgRsAll2AllManager', failed_manager)
    original = failed_loader if phase == 'loader' else route.native.load_dspark_model
    with pytest.raises(RuntimeError, match='failed'):
        route.runtime.load_with_draft_scope(original, route.target_model, route.config)
    assert route.ps.get_ep_group() is route.ep
    assert route.current[0] is None
    assert route.runtime.get_dspark_token_capacity() is None
    assert route.ep.device_communicator.all2all_manager is route.manager
