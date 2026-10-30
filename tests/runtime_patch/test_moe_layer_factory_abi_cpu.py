"""Portable CPU contracts for the fixed-core FusedMoEFactory adapter ABI.
HCU/core source roots are optional. GPU imports, expert-map/model constructors
and numerical kernels are boundaries; the native factory body is unmodified.
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

HCU=Path(os.environ.get('VLLM_HCU_SOURCE_ROOT',Path(__file__).resolve().parents[2]))
_DEFAULT_CORE=next((p for p in (HCU.parent/'vllm-community-main',HCU.parent.parent/'vllm-community-main')
                    if (p/'vllm/model_executor/layers/fused_moe/layer.py').is_file()),HCU.parent/'vllm-community-main')
CORE=Path(os.environ.get('VLLM_SOURCE_ROOT',_DEFAULT_CORE))
OBSERVATIONS=[]


def module(monkeypatch,name,**values):
    m=ModuleType(name);m.__path__=[];m.__dict__.update(values)
    monkeypatch.setitem(sys.modules,name,m)
    if '.' in name and name.rsplit('.',1)[0] in sys.modules:
        monkeypatch.setattr(sys.modules[name.rsplit('.',1)[0]],name.rsplit('.',1)[1],m,raising=False)
    return m


def load(monkeypatch,name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,name,m)
    spec.loader.exec_module(m)
    return m


def source_defs(path,names,ns,*,strip_bases=False,strip_decorators=False,methods=None):
    nodes=[]
    for n in ast.parse(path.read_text()).body:
        if getattr(n,'name',None) not in names:continue
        n=copy.deepcopy(n)
        if isinstance(n,ast.ClassDef):
            if strip_bases:n.bases=[]
            if strip_decorators:n.decorator_list=[]
            if methods is not None:n.body=[x for x in n.body if getattr(x,'name',None) in methods]
        nodes.append(n)
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),str(path),'exec'),ns)


def build_route(monkeypatch):
    if not (CORE/'vllm/model_executor/layers/fused_moe/layer.py').is_file():
        pytest.skip('paired core source unavailable; optional VLLM_SOURCE_ROOT selects it')
    for name in ('vllm_hcu','vllm_hcu.patch','vllm_hcu.patch.worker','vllm_hcu.patch.worker.op_opt',
                 'vllm_hcu.patch.worker.op_opt.moe','vllm_hcu.model_executor',
                 'vllm_hcu.model_executor.layers','vllm_hcu.model_executor.layers.fused_moe',
                 'vllm','vllm.model_executor','vllm.model_executor.layers',
                 'vllm.model_executor.layers.fused_moe'):
        module(monkeypatch,name)
    common=load(monkeypatch,'vllm_hcu.patch.worker.op_opt._common',HCU/'vllm_hcu/patch/worker/op_opt/_common.py')
    load(monkeypatch,'vllm_hcu.patch.worker.op_opt.moe._common',HCU/'vllm_hcu/patch/worker/op_opt/moe/_common.py')
    state=load(monkeypatch,'vllm_hcu.patch.runtime_state',HCU/'vllm_hcu/patch/runtime_state.py')
    coordinator=load(monkeypatch,'vllm_hcu.patch.import_coordinator',HCU/'vllm_hcu/patch/import_coordinator.py')
    adapter=load(monkeypatch,'vllm_hcu.patch.worker.op_opt.moe.patch_layer',HCU/'vllm_hcu/patch/worker/op_opt/moe/patch_layer.py')
    load(monkeypatch,'vllm_hcu.deepseek_v4_runtime',HCU/'vllm_hcu/deepseek_v4_runtime.py')
    cfg=NS(parallel_config=NS(enable_expert_parallel=True,enable_eplb=False,enable_dbo=False,
               expert_placement_strategy='linear',elastic_ep_max_dp_size=None,all2all_backend='deepep_low_latency'),
           scheduler_config=NS(max_num_batched_tokens=64),
           model_config=NS(dtype=torch.bfloat16,architectures=['DeepseekV41ForCausalLM'],hf_config=NS(model_type='deepseek_v41')),
           kernel_config=NS(moe_backend='deep_gemm'),lora_config=None,device_config=NS(device='cpu'),
           compilation_config=NS(max_cudagraph_capture_size=0))
    module(monkeypatch,'vllm.config',get_current_vllm_config=lambda:cfg,get_current_vllm_config_or_none=lambda:cfg)
    parent=sys.modules[adapter.TARGET_MODULE]
    layer=module(monkeypatch,adapter.LAYER_MODULE,torch=torch)
    routed=module(monkeypatch,adapter.ROUTED_EXPERTS_MODULE,torch=torch)
    official=module(monkeypatch,'vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method',torch=torch)
    source_defs(CORE/'vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py',
                {'UnquantizedFusedMoEMethod'},official.__dict__,strip_bases=True,strip_decorators=True)
    base=official.UnquantizedFusedMoEMethod
    hcu=module(monkeypatch,'vllm_hcu.model_executor.layers.fused_moe.unquantized_fused_moe_method',torch=torch,_Original=base)
    source_defs(HCU/'vllm_hcu/model_executor/layers/fused_moe/unquantized_fused_moe_method.py',
                {'HcuUnquantizedFusedMoEMethod'},hcu.__dict__)
    source_defs(CORE/'vllm/model_executor/layers/fused_moe/routed_experts.py',{'RoutedExperts'},routed.__dict__,
                strip_bases=True,strip_decorators=True,methods={'expert_map','get_expert_weights','load_weights'})
    source_defs(CORE/'vllm/model_executor/layers/fused_moe/layer.py',{'FusedMoEFactory'},layer.__dict__)
    for owner in (parent,layer):
        owner.FusedMoEFactory=layer.FusedMoEFactory;owner.RoutedExperts=routed.RoutedExperts
    parent.UnquantizedFusedMoEMethod=routed.UnquantizedFusedMoEMethod=base
    parallel=module(monkeypatch,'layer_abi_native_parallel',dataclass=dataclass,
                    get_dp_group=lambda:NS(rank_in_group=0),get_pcp_group=lambda:NS(rank_in_group=0),
                    get_tensor_model_parallel_rank=lambda:0)
    source_defs(CORE/'vllm/model_executor/layers/fused_moe/config.py',{'FusedMoEParallelConfig'},parallel.__dict__)
    captures={}
    def make_parallel_config(*,tp_size,dp_size,pcp_size,is_sequence_parallel,parallel_config):
        return parallel.FusedMoEParallelConfig.make(tp_size or 1,pcp_size or 1,dp_size or 8,1,parallel_config)
    def make_config(**kw):
        captures['moe_config']=kw.copy()
        return NS(**kw,pcp_size=kw['moe_parallel_config'].pcp_size)
    def create_router(**kw):
        captures['router']=kw.copy()
        return NS(routing_method_type='CPU-contract')
    class ExpertMap:
        def __init__(self,**kw):
            self.local_num_experts=kw['global_num_experts']//kw['moe_parallel_config'].ep_size
    class RecordingExperts:
        def __init__(self,layer_name,params_dtype,moe_config,quant_config,**kw):
            self.moe_config=moe_config
            self.quant_method=NS(moe=moe_config)
            self.apply_router_weight_on_input=False
            captures['experts']=(layer_name,params_dtype,moe_config,quant_config,kw)
    class RecordingRunner:
        def __init__(self,**kw):
            self.routed_experts=kw['routed_experts'];captures['runner']=kw.copy()
    layer.__dict__.update(get_current_vllm_config=lambda:cfg,
        MoEActivation=NS(from_str=lambda text:NS(value=text,is_gated=True)),make_parallel_config=make_parallel_config,
        determine_expert_counts=lambda e,r,n,f:(e+r,e,0),ExpertMapManager=ExpertMap,
        rocm_aiter_ops=NS(is_fused_moe_enabled=lambda:False),create_fused_moe_router=create_router,
        FusedMoEConfig=make_config,logger=NS(debug=lambda *a:None))
    r=NS(adapter=adapter,common=common,state=state,coordinator=coordinator,parent=parent,layer=layer,
         routed=routed,hcu=hcu,official=base,cfg=cfg,captures=captures,
         RecordingExperts=RecordingExperts,RecordingRunner=RecordingRunner,native_factory=layer.FusedMoEFactory)
    return r


@pytest.fixture
def route(monkeypatch):
    r=build_route(monkeypatch)
    yield r
    assert not torch.cuda.is_initialized()


def snapshot(r):
    cls=r.routed.RoutedExperts
    return (r.parent.FusedMoEFactory,r.layer.FusedMoEFactory,vars(cls).get('expert_map'),
            cls.get_expert_weights,cls.load_weights,r.parent.UnquantizedFusedMoEMethod,
            r.routed.UnquantizedFusedMoEMethod,getattr(r.parent,r.adapter._MARKER,False))


def install(r,adapter=None):
    adapter=adapter or r.adapter
    coord=r.coordinator.ExactImportCoordinator(registry=r.state.PATCH_REGISTRY)
    try:
        coord.register_callback(adapter.PATCH_ID,adapter.TARGET_MODULE,adapter.apply_to_module,targets=adapter.TARGETS)
        assert r.state.PATCH_REGISTRY.get(adapter.PATCH_ID).status is r.state.PatchStatus.APPLIED
        first=r.parent.FusedMoEFactory
        coord.register_callback(adapter.PATCH_ID,adapter.TARGET_MODULE,adapter.apply_to_module,targets=adapter.TARGETS)
        assert r.parent.FusedMoEFactory is first
        assert adapter.apply_to_module(r.parent) is False
    finally:
        coord.uninstall()


def test_real_registration_checks_all_gates_aliases_and_markers(route):
    r=route;install(r)
    assert r.parent.FusedMoEFactory is r.layer.FusedMoEFactory
    assert r.parent.FusedMoE is r.parent.FusedMoEFactory
    assert r.routed.UnquantizedFusedMoEMethod is r.hcu.HcuUnquantizedFusedMoEMethod
    assert getattr(r.routed.RoutedExperts.expert_map.fget,r.adapter._EXPERT_MAP_MARKER)
    assert getattr(r.routed.RoutedExperts.get_expert_weights,r.adapter._GET_WEIGHTS_MARKER)
    assert getattr(r.routed.RoutedExperts.load_weights,r.adapter._LOAD_WEIGHTS_MARKER)
    assert r.state.PATCH_REGISTRY.get(r.adapter.PATCH_ID).status.name=='APPLIED'


@pytest.mark.parametrize('style,backend,requested',[
    ('default','deepep_low_latency',False),('default','deepep_v2',False),
    ('keyword','deepep_low_latency',True),('keyword','deepep_v2',True),
    ('positional','deepep_low_latency',True),('positional','deepep_v2',True),
    ('keyword','deepep_v2',False),('positional','deepep_v2',False),
])
def test_native_factory_consumes_shared_prefix_and_conditional_skip(route,style,backend,requested):
    r=route;r.cfg.parallel_config.all2all_backend=backend;install(r)
    shared=None if style=='default' else ''.join(['model.','shared_experts'])
    kwargs=dict(runner_cls=r.RecordingRunner,routed_experts_cls=r.RecordingExperts,tp_size=1,dp_size=8,pcp_size=1)
    if style!='default':kwargs.update(shared_expert_prefix=shared,skip_padding=requested)
    if style=='positional':
        bound=inspect.signature(r.native_factory).bind(8,1,512,256,**kwargs)
        bound.apply_defaults()
        runner=r.parent.FusedMoEFactory(*bound.args,**bound.kwargs)
    else:runner=r.parent.FusedMoEFactory(8,1,512,256,**kwargs)
    assert r.captures['moe_config']['shared_expert_prefix'] is shared
    expected=requested and backend=='deepep_v2'
    assert r.captures['router']['skip_padding'] is expected
    assert runner is not None and r.captures['runner']['moe_config'].shared_expert_prefix is shared
    OBSERVATIONS.append(dict(style=style,backend=backend,requested_skip=requested,router_skip=expected,
                             shared_prefix=shared,registry='APPLIED',native_body=str(CORE/'vllm/model_executor/layers/fused_moe/layer.py')))


@pytest.mark.parametrize('kind',['unknown_arg','reordered_args','skip_default','skip_keyword_only',
    'factory_alias','routed_alias','unquantized_alias','hcu_subclass','expert_map','get_weights','load_weights'])
def test_signature_and_alias_drift_fail_before_partial_mutation(route,kind):
    r=route
    if kind in ('unknown_arg','reordered_args','skip_default','skip_keyword_only'):
        tree=ast.parse((CORE/'vllm/model_executor/layers/fused_moe/layer.py').read_text())
        fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='FusedMoEFactory'))
        if kind=='unknown_arg':fn.args.args.append(ast.arg(arg='unknown_core_argument'));fn.args.defaults.append(ast.Constant(None))
        if kind=='reordered_args':
            names=[a.arg for a in fn.args.args];i=names.index('shared_expert_prefix');j=i+1
            fn.args.args[i],fn.args.args[j]=fn.args.args[j],fn.args.args[i]
        if kind=='skip_default':fn.args.defaults[-1]=ast.Constant(True)
        if kind=='skip_keyword_only':
            arg=fn.args.args.pop();default=fn.args.defaults.pop()
            fn.args.kwonlyargs.append(arg);fn.args.kw_defaults.append(default)
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,fn],type_ignores=[])),'native_factory_drift','exec'),r.layer.__dict__)
        r.parent.FusedMoEFactory=r.layer.FusedMoEFactory
    if kind=='factory_alias':r.layer.FusedMoEFactory=lambda *a,**kw:None
    if kind=='routed_alias':r.layer.RoutedExperts=type('WrongRouted',(),{})
    if kind=='unquantized_alias':r.routed.UnquantizedFusedMoEMethod=type('WrongUnquantized',(),{})
    if kind=='hcu_subclass':r.hcu.HcuUnquantizedFusedMoEMethod=type('WrongHcu',(),{})
    if kind=='expert_map':r.routed.RoutedExperts.expert_map=None
    if kind=='get_weights':
        def bad(self,new_arg):pass
        r.routed.RoutedExperts.get_expert_weights=bad
    if kind=='load_weights':
        def bad(self,weights,new_arg):pass
        r.routed.RoutedExperts.load_weights=bad
    before=snapshot(r)
    coord=r.coordinator.ExactImportCoordinator(registry=r.state.PATCH_REGISTRY)
    try:
        with pytest.raises(r.common.PatchCompatibilityError):
            coord.register_callback(r.adapter.PATCH_ID,r.adapter.TARGET_MODULE,r.adapter.apply_to_module,targets=r.adapter.TARGETS)
        assert r.state.PATCH_REGISTRY.get(r.adapter.PATCH_ID).status.name=='FAILED'
        assert snapshot(r)==before
        assert not getattr(r.parent,r.adapter._MARKER,False)
    finally:coord.uninstall()


@pytest.mark.parametrize('kind',['factory_alias','captured_unquantized','weight_marker'])
def test_registered_marker_or_alias_drift_is_rejected_without_repair(route,kind):
    r=route;install(r)
    if kind=='factory_alias':r.layer.FusedMoEFactory=r.native_factory
    if kind=='captured_unquantized':r.routed.UnquantizedFusedMoEMethod=r.official
    if kind=='weight_marker':
        def plain(self):pass
        r.routed.RoutedExperts.get_expert_weights=plain
    before=snapshot(r)
    with pytest.raises(r.common.PatchCompatibilityError,match='stale'):
        r.adapter.apply_to_module(r.parent)
    assert snapshot(r)==before
