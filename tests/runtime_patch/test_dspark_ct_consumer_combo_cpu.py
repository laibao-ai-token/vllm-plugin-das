"""CPU composition of common producer/copy/caller and landed D CT/factory.
Native source bodies are real; model/checkpoint/GEMM and PG collectives remain
explicit CPU boundaries. Quant-method construction deliberately bypasses the
still-closed deployment selector; no DP8/draft/graph admission is implied.
"""
from __future__ import annotations
import ast
import copy
from dataclasses import dataclass
import importlib.util
import inspect
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import pytest
import torch

D = Path(os.environ.get('VLLM_HCU_SOURCE_ROOT', Path(__file__).resolve().parents[2]))
CORE = Path(os.environ.get('VLLM_SOURCE_ROOT', D.parent.parent/'vllm-community-main'))
COMMON = Path(os.environ.get('HCU_DSPARK_CANDIDATE_ROOT', D))
OBSERVATIONS = []


def load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def nodes(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [n for n in tree.body if getattr(n, 'name', None) in names]
    tree.body.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), 'exec'), namespace)


class MethodBase:
    def __init__(self, moe):
        self.moe = moe
        self.moe_quant_config = self.moe_kernel = None


class QuantConfig:
    @staticmethod
    def make(dtype, **kw):
        return NS(quant_dtype=dtype, a1_scale=None, a2_scale=None,
                  use_nvfp4_w4a4=False, is_scale_swizzled=False, mx_alignment=0, **kw)


def replace_parameter(layer, name, value):
    old = getattr(layer, name)
    param = torch.nn.Parameter(value, requires_grad=False)
    param.__dict__.update(old.__dict__)
    setattr(layer, name, param)


def set_weight_attrs(param, values):
    param.__dict__.update(values)


@pytest.fixture
def combo(monkeypatch, request):
    if not (COMMON/'tests/runtime_patch/test_dspark_producer_cpu.py').is_file():
        pytest.skip('requires the separately reviewed common producer/config-copy test dependencies')
    producer = load(monkeypatch, 'test_dspark_producer_cpu', COMMON/'tests/runtime_patch/test_dspark_producer_cpu.py')
    context = load(monkeypatch, 'test_dspark_forward_context_cpu', COMMON/'tests/runtime_patch/test_dspark_forward_context_cpu.py')
    r = producer.route.__wrapped__(monkeypatch)
    assert r.models == []  # The fixture's artificial binding get_model is never called.
    r.feature.set_hcu_config(r.config, deepep_auto=False)
    r.config.model_config.dtype = torch.bfloat16
    h = NS(route=r, models=[], factory_calls=[], owner_writes=[], kernel_builds=[],
           prepare_events=[], fail=None, counts=None, last_metadata=None)
    mod = lambda name, **kw: producer.module(monkeypatch, name, **kw)
    for name in ('vllm_hcu.patch.worker.op_opt', 'vllm_hcu.patch.worker.op_opt.moe',
                 'vllm_hcu.model_executor', 'vllm_hcu.model_executor.layers',
                 'vllm_hcu.model_executor.layers.fused_moe',
                 'vllm_hcu.model_executor.layers.quantization',
                 'vllm.model_executor.layers.fused_moe',
                 'vllm.model_executor.layers.fused_moe.prepare_finalize'):
        mod(name)
    mk = mod('vllm.model_executor.layers.fused_moe.modular_kernel',
             FusedMoEPrepareAndFinalizeModular=object,
             FusedMoEPrepareAndFinalizeMonolithic=object,
             FusedMoEActivationFormat=NS(Standard='standard', BatchedExperts='batched'))
    # Preserve actual FusedMoEParallelConfig.make and its native rank arithmetic.
    parallel_ns = mod('combo_native_moe_parallel').__dict__
    parallel_ns.update(dataclass=dataclass, get_dp_group=r.ps.get_dp_group,
                       get_pcp_group=r.ps.get_pcp_group, get_tensor_model_parallel_rank=lambda: 0)
    nodes(CORE/'vllm/model_executor/layers/fused_moe/config.py', {'FusedMoEParallelConfig'}, parallel_ns)
    Parallel = parallel_ns['FusedMoEParallelConfig']

    # Restore inherited communicator methods omitted by the producer CPU fixture.
    base_ns = {}
    producer.source_nodes(CORE/'vllm/distributed/device_communicators/base_device_communicator.py',
        {'DeviceCommunicatorBase': {'allocate_combine_input', 'combine_into_output'}}, base_ns)
    comm_type = type(r.ep.device_communicator)
    for name in ('allocate_combine_input', 'combine_into_output'):
        setattr(comm_type, name, getattr(base_ns['DeviceCommunicatorBase'], name))

    reduce_ns = dict(torch=torch)
    producer.source_nodes(CORE/'vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py',
        {'TopKWeightAndReduceNoOP': None, 'TopKWeightAndReduceDelegate': None,
         'TopKWeightAndReduceContiguous': None}, reduce_ns)
    noop = reduce_ns['TopKWeightAndReduceNoOP']
    def quant_input(x, scale, **kwargs):
        # CPU substitute for the external quantization kernel only.
        s = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-8)/127
        return (x.float()/s).round().clamp(-127, 127).to(torch.int8), s
    pf_ns = dict(torch=torch, mk=mk, get_ep_group=r.ps.get_ep_group,
                 moe_kernel_quantize_input=quant_input, **{k:v for k,v in reduce_ns.items() if k.startswith('TopK')})
    producer.source_nodes(CORE/'vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py',
        {'_quantize_and_setup_dispatch': None, '_unwrap_scale_and_prepare_for_moe': None,
         'MoEPrepareAndFinalizeNaiveDPEPModular': None,
         'make_moe_prepare_and_finalize_naive_dp_ep': None}, pf_ns)
    Naive = pf_ns['MoEPrepareAndFinalizeNaiveDPEPModular']

    ll = mod('vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll', torch=torch, mk=mk)
    producer.source_nodes(CORE/'vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py',
                          {'DeepEPLLPrepareAndFinalize': None}, ll.__dict__)
    ll_runtime = mod('vllm_hcu.model_executor.layers.fused_moe.deepep_runtime', inspect=inspect)
    producer.source_nodes(D/'vllm_hcu/model_executor/layers/fused_moe/deepep_runtime.py',
        {'_has_hcu_low_latency_dispatch_abi': None, 'll_init': None, 'll_do_quant': None,
         'll_receiver': None, 'll_prepare_async': None}, ll_runtime.__dict__)
    load(monkeypatch, 'vllm_hcu.patch.worker.op_opt._common', D/'vllm_hcu/patch/worker/op_opt/_common.py')
    load(monkeypatch, 'vllm_hcu.patch.worker.op_opt.moe._common', D/'vllm_hcu/patch/worker/op_opt/moe/_common.py')
    ll_patch = load(monkeypatch, 'vllm_hcu.patch.worker.op_opt.moe.patch_deepep_ll',
                    D/'vllm_hcu/patch/worker/op_opt/moe/patch_deepep_ll.py')
    assert ll_patch.apply_to_module(ll)
    factory = mod('vllm.model_executor.layers.fused_moe.all2all_utils',
                  torch=torch, get_ep_group=r.ps.get_ep_group,
                  current_platform=NS(is_xpu=lambda: False, fp8_dtype=lambda: torch.float8_e4m3fn),
                  DEEPEP_QUANT_BLOCK_SHAPE=[128,128],
                  DeepEPLLPrepareAndFinalize=ll.DeepEPLLPrepareAndFinalize,
                  make_moe_prepare_and_finalize_naive_dp_ep=pf_ns['make_moe_prepare_and_finalize_naive_dp_ep'])
    nodes(CORE/'vllm/model_executor/layers/fused_moe/all2all_utils.py',
          {'get_ep_all2all_manager','maybe_make_prepare_finalize','maybe_roundup_layer_hidden_size'}, factory.__dict__)
    native_factory = factory.maybe_make_prepare_finalize
    def audited_factory(moe, quant_config, routing_tables=None, allow_new_interface=False,
                        use_monolithic=False, all2all_manager=None, input_dtype=None):
        h.factory_calls.append((moe.moe_parallel_config.all2all_backend, all2all_manager, input_dtype))
        if h.fail == 'factory' and all2all_manager is not None:
            raise RuntimeError('injected factory failure')
        return native_factory(moe, quant_config, routing_tables, allow_new_interface,
                              use_monolithic, all2all_manager, input_dtype)
    factory.maybe_make_prepare_finalize = audited_factory
    factory_patch = load(monkeypatch, 'vllm_hcu.patch.worker.op_opt.moe.patch_all2all_utils',
                         D/'vllm_hcu/patch/worker/op_opt/moe/patch_all2all_utils.py')
    assert factory_patch.apply_to_module(factory)
    assert not factory_patch.apply_to_module(factory)

    class LLManager:
        world_size = dp_world_size = 8
        rank = 0
        def __init__(self): self.requests = []
        def get_handle(self, kwargs):
            self.requests.append(kwargs.copy())
            return NS(request=kwargs.copy())  # No hardware buffer is created.
    h.target_manager = LLManager()
    r.ep.device_communicator.all2all_manager = h.target_manager

    class CPUKernel:
        def __init__(self, pf, experts):
            if h.fail == 'kernel' and isinstance(pf, Naive):
                raise RuntimeError('injected kernel construction failure')
            self.prepare_finalize, self.experts = pf, experts
            h.kernel_builds.append(self)
        def apply(self, **kw):
            if not isinstance(self.prepare_finalize, Naive):
                raise AssertionError('LL device execution is outside this CPU composition')
            meta = h.active[0].dp_metadata
            h.last_metadata = meta
            h.prepare_events.append((r.ps.get_ep_group(), meta,
                                     r.ps.get_ep_group().device_communicator.all2all_manager))
            aq, scales, _, ids, weights = self.prepare_finalize.prepare(
                kw['hidden_states'], kw['topk_weights'], kw['topk_ids'],
                kw['global_num_experts'], kw['expert_map'], False, self.experts.quant)
            if h.fail == 'expert': raise RuntimeError('injected expert boundary failure')
            # Identity expert is the only math boundary; actual native PF
            # dispatch/finalize, GroupCoordinator, communicator and AGRS run.
            fused = (aq.float()*scales).to(kw['hidden_states'].dtype)
            output = torch.empty_like(kw['hidden_states'])
            self.prepare_finalize.finalize(output, fused, weights, ids, False, noop())
            return output
    mk.FusedMoEKernel = CPUKernel
    rt = mod('vllm_hcu.model_executor.layers.quantization.slimquant_w4a8_deepgemm_runtime',
             pack_w4a8_moe_hipc_weight=lambda w:w,
             view_w4a8_moe_hipc_weight_n32_layout=lambda w:w.view(w.size(0),w.size(2)//32,w.size(1)//32,4,32,8),
             DeepEPDeepGemmW4A8ContiguousExperts=lambda cfg,q:NS(quant=q),
             DeepEPDeepGemmW4A8BatchedExperts=lambda cfg,q,*a:NS(quant=q))
    ct_ns = dict(torch=torch, Parameter=torch.nn.Parameter, functools=__import__('functools'),
                 CompressedTensorsMoEMethod=MethodBase, FusedMoEQuantConfig=QuantConfig,
                 replace_parameter=replace_parameter, set_weight_attrs=set_weight_attrs)
    ct_path = D/'vllm_hcu/model_executor/layers/quantization/compressed_tensors_w4a8_deepgemm.py'
    tree = ast.parse(ct_path.read_text(), filename=str(ct_path))
    tree.body = [n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    tree.body.insert(0,ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0))
    exec(compile(ast.fix_missing_locations(tree),str(ct_path),'exec'),ct_ns)
    CT = ct_ns['HCUCompressedTensorsW4A8Int8MoEMethod']

    class MoE(NS):
        def __setattr__(self, name, value):
            if name == '_hcu_dspark_all2all_manager':
                caller = sys._getframe(1)
                h.owner_writes.append((value, caller.f_code.co_filename, caller.f_code.co_name))
            super().__setattr__(name, value)
        def __getattr__(self, name):
            return getattr(self.moe_parallel_config, name)
    def make_model(config, model_config):
        if h.fail == 'loader': raise RuntimeError('injected loader failure')
        par = config.parallel_config
        parallel = Parallel.make(par.tensor_parallel_size, par.prefill_context_parallel_size,
                                 par.data_parallel_size, 1, par)
        moe = MoE(moe_parallel_config=parallel, num_experts=8, num_local_experts=1,
                  hidden_dim=5120, max_num_tokens=config.scheduler_config.max_num_batched_tokens,
                  in_dtype=model_config.dtype, moe_backend=config.kernel_config.moe_backend,
                  activation='silu', is_act_and_mul=True, has_bias=False, swiglu_limit=10.0)
        model = torch.nn.Module()
        model.model, model.moe = NS(), moe
        w = NS(num_bits=4,type='int',strategy='channel',symmetric=True,dynamic=False,group_size=None)
        a = NS(num_bits=8,type='int',strategy='token',symmetric=True,dynamic=True)
        model.method = CT(w,a,moe)
        model.layer = torch.nn.Module()
        model.method.create_weights(model.layer,1,5120,256,torch.bfloat16)
        model.layer._expert_routing_tables = lambda: None
        for name in ('w13_weight_packed','w2_weight_packed'):
            getattr(model.layer,name).data.zero_()
        for name in ('w13_weight_scale','w2_weight_scale'):
            getattr(model.layer,name).data.fill_(1)
        model.layer.activation, model.layer.global_num_experts = 'silu',8
        model.layer.expert_map, model.layer.apply_router_weight_on_input = None,False
        h.models.append(model)
        assert not hasattr(moe,'_hcu_dspark_all2all_manager')
        model.method.process_weights_after_loading(model.layer)  # Sole final binding writer.
        def forward(input_ids,positions=None,inputs_embeds=None):
            rows = input_ids.shape[0]
            x = torch.ones((rows,5120),dtype=torch.bfloat16)
            weights = torch.ones((rows,1)); ids = torch.zeros((rows,1),dtype=torch.int64)
            metadata = h.active[0].dp_metadata
            def apply(): return model.method.apply(model.layer,x,weights,ids,None,None)
            if metadata is None: return apply()
            with metadata.sp_local_sizes(1,1,True): return apply()
        model.forward = forward
        config.compilation_config.static_forward_context['draft.attn'] = model
        config.compilation_config.static_all_moe_layers.append('draft.moe')
        return model
    h.make_model = make_model
    monkeypatch.setattr(sys.modules['vllm.model_executor.model_loader'],'get_model',
                        lambda *,vllm_config,model_config: make_model(vllm_config,model_config))
    assert r.models == []
    target, early, active, original = context.context_runner(r,monkeypatch)
    h.context_target, h.early, h.active, h.original_context = target,early,active,original
    caller = mod(r.callback.CALLER_MODULE,load_dspark_model=r.native.load_dspark_model,
                 DFlashSpeculator=target.DFlashSpeculator, torch=torch)
    path = CORE/'vllm/v1/worker/gpu/spec_decode/dspark/speculator.py'
    cls = copy.deepcopy(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DSparkSpeculator'))
    cls.bases=[ast.Name(id='DFlashSpeculator',ctx=ast.Load())]
    cls.body=[n for n in cls.body if getattr(n,'name',None)=='load_draft_model']
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,cls],type_ignores=[])),str(path),'exec'),caller.__dict__)
    runner = caller.DSparkSpeculator()
    runner._speculator_name='DSpark'; runner.vllm_config=r.config
    runner.draft_logits=None; runner.enable_adaptive_verification=False
    runner.input_buffers=NS(input_ids=torch.ones(65,dtype=torch.int64),positions=torch.arange(65))
    h.runner=runner
    assert r.callback.apply_to_module(r.native)
    assert not r.callback.apply_to_module(r.native)
    assert caller.load_dspark_model is r.native.load_dspark_model
    def load_draft():
        before=copy.deepcopy(r.config.parallel_config.__dict__)
        try:
            model=runner.load_draft_model(r.target_model,set())
            runner.model=model
            return model
        finally:
            h.target_parallel_preserved=(r.config.parallel_config.__dict__==before)
            assert h.target_parallel_preserved
    h.load=load_draft
    h.target=make_model(r.config,r.config.model_config)
    h.target_models=len(h.models)
    h.target_writes=len(h.owner_writes)
    assert h.target.method._construction_manager is None
    h.target_config_values=copy.deepcopy(r.config.additional_config)
    def restored():
        assert r.ps.get_ep_group() is r.ep and r.current[0] is None
        assert r.ep.device_communicator.all2all_manager is h.target_manager
        assert active[0] is original and runner.vllm_config is r.config
        assert r.runtime.get_dspark_token_capacity() is None
        assert r.config.additional_config==h.target_config_values
        assert h.target.method._construction_manager is None
        assert not hasattr(h.target.moe,'_hcu_dspark_all2all_manager')
        assert r.models==[]
    h.restored=restored
    yield h
    assert not torch.cuda.is_initialized()
    scope=getattr(getattr(runner,'model',None),'_hcu_dspark_draft_scope',None)
    OBSERVATIONS.append(dict(
        case=request.node.nodeid,
        native_loader_source=runner.load_draft_model.__func__.__code__.co_filename,
        native_forward_source=inspect.unwrap(runner._run_model).__code__.co_filename,
        target_default_owner_none=h.target.method._construction_manager is None,
        target_handle_request_capacity=h.target_manager.requests[0]['max_num_tokens_per_dp_rank'],
        target_pf_capacity=h.target.method.moe_kernel.prepare_finalize.max_num_tokens_per_rank(),
        target_dispatchers=h.target.method.moe_kernel.prepare_finalize.num_dispatchers(),
        draft_capacity=None if scope is None else scope.capacity.__dict__.copy(),
        ownership_writes=[dict(manager_class=type(value).__name__,source_file=file,source_function=fn)
                          for value,file,fn in h.owner_writes],
        forbidden_original_model_stub_calls=len(r.models),
        factory_calls=[dict(backend=backend,manager_class=None if manager is None else type(manager).__name__,
                            input_dtype=None if dtype is None else str(dtype))
                       for backend,manager,dtype in h.factory_calls],
        prepares=[dict(dp_metadata_present=meta is not None,
                       padded_rows=None if meta is None else meta.num_tokens_across_dp_cpu.tolist(),
                       caller_count_identity=None if meta is None or h.counts is None else meta.num_tokens_across_dp_cpu is h.counts,
                       manager_class=type(manager).__name__)
                  for ep,meta,manager in h.prepare_events],
        count_semantics='caller-provided uniform padded CPU rows; not effective per-rank requests',
        restored=dict(ep=r.ps.get_ep_group() is r.ep,current_config=r.current[0] is None,
                      forward_context=active[0] is original,speculator_config=runner.vllm_config is r.config,
                      target_manager=r.ep.device_communicator.all2all_manager is h.target_manager,
                      active_capacity=r.runtime.get_dspark_token_capacity() is None),
        target_parallel_preserved=getattr(h,'target_parallel_preserved',None),
        copy_dependency_failures=getattr(h,'copy_dependency_failures',None),
        early_context_failure_function=getattr(h,'context_failure_function',None),
    ))


@pytest.mark.parametrize('anchor,rows',[(True,20),(False,24)])
def test_native_producer_ct_postload_factory_and_early_context(combo,anchor,rows):
    h=combo; r=h.route
    r.config.speculative_config.draft_model_config.hf_config.sample_from_anchor=anchor
    model=h.load(); scope=model._hcu_dspark_draft_scope
    assert model.moe._hcu_dspark_all2all_manager is scope.manager
    assert model.method._construction_manager is scope.manager
    assert model.method._dspark_token_capacity is scope.capacity
    assert (scope.capacity.query_tokens_per_rank,scope.capacity.verify_tokens_per_rank,
            scope.capacity.capacity_per_rank)==(rows,24,64)
    assert scope.manager is not h.target_manager
    assert h.factory_calls[-1]==('allgather_reducescatter',scope.manager,None)
    assert h.owner_writes[-1][0] is scope.manager
    assert Path(h.owner_writes[-1][1])==D/'vllm_hcu/model_executor/layers/quantization/compressed_tensors_w4a8_deepgemm.py'
    assert h.owner_writes[-1][2]=='process_weights_after_loading'
    assert h.target_manager.requests[0]['max_num_tokens_per_dp_rank']==64
    assert h.target.method.moe_kernel.prepare_finalize.max_num_tokens_per_rank()==64
    assert h.target.method.moe_kernel.prepare_finalize.num_dispatchers()==8
    assert h.early.apply_to_module(h.context_target)
    assert not h.early.apply_to_module(h.context_target)
    counts=torch.tensor([rows]*8); h.counts=counts
    output=h.runner._run_model(rows,None,None,counts)
    assert output.shape==(rows,5120)
    ep,metadata,manager=h.prepare_events[-1]
    assert ep is scope.draft_ep and manager is scope.manager
    assert metadata.num_tokens_across_dp_cpu is counts and metadata.local_sizes is None
    assert any(e[0]=='all_gatherv' for e in r.events)
    assert any(e[0]=='reduce_scatterv' for e in r.events)
    h.restored()


def test_context_must_switch_before_native_run_model(combo):
    h=combo; model=h.load(); counts=torch.tensor([20]*8); h.counts=counts
    with pytest.raises(AssertionError) as missing_context:
        h.runner._run_model(20,None,None,counts)
    assert missing_context.traceback[-1].name=='_get_sizes'
    h.context_failure_function=missing_context.traceback[-1].name
    assert h.prepare_events[-1][1] is None
    h.restored()
    h.early.apply_to_module(h.context_target)
    assert h.runner._run_model(20,None,None,counts).shape==(20,5120)
    assert h.prepare_events[-1][1].num_tokens_across_dp_cpu is counts
    assert h.prepare_events[-1][2] is model._hcu_dspark_draft_scope.manager
    h.restored()


@pytest.mark.parametrize('phase',['manager','loader','factory','kernel'])
def test_load_failure_restores_scope_without_artificial_binding(combo,monkeypatch,phase):
    h=combo; h.fail=phase
    if phase=='manager':
        def failed_manager(*args,**kwargs):
            raise RuntimeError('injected manager construction failure')
        monkeypatch.setattr(sys.modules['vllm.distributed.device_communicators.all2all'],
                            'AgRsAll2AllManager',failed_manager)
    with pytest.raises(RuntimeError,match='injected'): h.load()
    h.restored()
    assert h.owner_writes==[]
    if len(h.models)>h.target_models:
        method=h.models[-1].method
        assert not hasattr(h.models[-1].moe,'_hcu_dspark_all2all_manager')
        assert not hasattr(method,'_construction_manager')


@pytest.mark.parametrize('phase',['expert','dispatch','combine'])
def test_forward_failure_restores_context_groups_and_metadata(combo,monkeypatch,phase):
    h=combo; model=h.load(); h.early.apply_to_module(h.context_target)
    dp=h.route.ps.get_dp_group()
    if phase=='expert': h.fail='expert'
    else:
        name='all_gatherv' if phase=='dispatch' else 'reduce_scatterv'
        def failed(*a,**kw): raise RuntimeError('injected collective boundary failure')
        monkeypatch.setattr(dp,name,failed)
    h.counts=torch.tensor([20]*8)
    with pytest.raises(RuntimeError,match='injected'):
        h.runner._run_model(20,None,None,h.counts)
    h.restored()
    assert model.method._construction_manager is model._hcu_dspark_draft_scope.manager
    assert model.method._dspark_token_capacity is model._hcu_dspark_draft_scope.capacity
    assert h.last_metadata.local_sizes is None


def test_capacity_overflow_rejected_before_ct_or_collective(combo):
    h=combo; model=h.load(); before=len(h.prepare_events)
    with pytest.raises(ValueError,match='exceeds checked'):
        model(input_ids=torch.ones(65,dtype=torch.int64))
    assert len(h.prepare_events)==before
    assert h.route.config.scheduler_config.max_num_batched_tokens==64
    h.restored()


@pytest.mark.parametrize('policy',['pcp8','target_graph','draft_graph','draft_backend'])
def test_unsupported_policies_stay_closed_before_ct_creation(combo,policy):
    h=combo; cfg=h.route.config
    if policy=='pcp8': cfg.speculative_config.target_parallel_config.prefill_context_parallel_size=8
    if policy=='target_graph': cfg.model_config.enforce_eager=False
    if policy=='draft_graph': cfg.speculative_config.draft_model_config.enforce_eager=False
    if policy=='draft_backend': cfg.speculative_config.moe_backend='triton'
    before=len(h.models)
    with pytest.raises(ValueError): h.load()
    assert len(h.models)==before and h.owner_writes==[]
    h.restored()


def test_actual_copy_companion_is_a_required_dependency(combo,monkeypatch):
    h=combo; cfg=h.route.config
    utils=sys.modules['vllm.config.utils']
    raw=utils._vllm_hcu_original_parallel_replace
    with pytest.raises(ValueError) as raw_copy:
        raw(cfg.parallel_config,prefill_context_parallel_size=1)
    assert '_vllm_hcu_' in str(raw_copy.value)
    original_loader=h.route.native._vllm_hcu_original_load_dspark_model
    before=len(h.models)
    with monkeypatch.context() as missing_companion:
        missing_companion.setitem(original_loader.__globals__,'replace',raw)
        with pytest.raises(ValueError) as loader_copy:
            h.load()
    assert '_vllm_hcu_' in str(loader_copy.value)
    h.copy_dependency_failures=dict(raw=str(raw_copy.value),native_loader=str(loader_copy.value))
    assert len(h.models)==before and h.owner_writes==[]
    h.restored()
    result=utils.replace(cfg.parallel_config,prefill_context_parallel_size=1)
    assert result is not cfg.parallel_config
    assert result.all2all_backend==cfg.parallel_config.all2all_backend
    for name in ('_vllm_hcu_deepep_auto','_vllm_hcu_expert_map_path',
                 '_vllm_hcu_expert_map_record_path','_vllm_hcu_eplb_disable_rearrange',
                 '_vllm_hcu_eplb_static_dispatch_policy'):
        assert getattr(result,name)==getattr(cfg.parallel_config,name)
    model=h.load()
    assert model.method._construction_manager is model._hcu_dspark_draft_scope.manager
    assert h.owner_writes[-1][2]=='process_weights_after_loading'
    h.restored()
