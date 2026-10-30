"""Fixed-core unquantized oracle ABI, real worker/cold-import CPU contracts.
Roots are optional paired-checkout overrides; no handoff resources are required.
Native oracle and postload caller modules execute whole source files. GPU/PG,
compiler registration and final expert-kernel setup are explicit CPU boundaries.
"""
from __future__ import annotations
import ast
import copy
from contextlib import contextmanager
import enum
import importlib
import importlib.util
import inspect
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import typing
import pytest
import torch

HCU=Path(os.environ.get('VLLM_HCU_SOURCE_ROOT',Path(__file__).resolve().parents[2]))
CORE=Path(os.environ.get('VLLM_SOURCE_ROOT',next((p for p in (HCU.parent/'vllm-community-main',HCU.parent.parent/'vllm-community-main') if (p/'vllm/model_executor/layers/fused_moe/oracle/unquantized.py').is_file()),HCU.parent/'vllm-community-main')))
TARGET='vllm.model_executor.layers.fused_moe.oracle.unquantized'
ADAPTER='vllm_hcu.patch.worker.op_opt.moe.patch_unquantized_oracle'
OBSERVATIONS=[]

def module(mp,name,**attrs):
    m=ModuleType(name);m.__path__=[];m.__dict__.update(attrs);mp.setitem(sys.modules,name,m)
    parent,_,leaf=name.rpartition('.')
    if parent and parent in sys.modules:mp.setattr(sys.modules[parent],leaf,m,raising=False)
    return m

def load(mp,name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    mp.setitem(sys.modules,name,m)
    parent,_,leaf=name.rpartition('.')
    if parent and parent in sys.modules:mp.setattr(sys.modules[parent],leaf,m,raising=False)
    spec.loader.exec_module(m);return m

def defs(path,names,ns):
    nodes=[copy.deepcopy(n) for n in ast.parse(path.read_text()).body if getattr(n,'name',None) in names]
    assert set(names)=={n.name for n in nodes}
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+nodes,type_ignores=[])),str(path),'exec'),ns)

def reserve(mp,name):
    # Record restoration before importlib inserts a real source module.
    mp.setitem(sys.modules,name,None);del sys.modules[name]

class FalseyLayer(torch.nn.Module):
    def __bool__(self):return False

class BaseMethod:
    def process_weights_after_loading(self,layer):self.base_postload_layer=layer

class CustomOpBoundary:
    @classmethod
    def register(cls,name):return lambda typ:typ

class ExpertBoundary:
    @staticmethod
    def is_supported_config(*args):return True,None

T=typing.TypeVar('T')
class OracleBoundary(typing.Generic[T]):pass


def build_route(mp,adapter_path=None):
    if not (CORE/'vllm/model_executor/layers/fused_moe/oracle/unquantized.py').is_file():
        pytest.skip('paired native source unavailable; optional VLLM_SOURCE_ROOT selects it')
    packages=('vllm','vllm.config','vllm.utils','vllm.model_executor','vllm.model_executor.layers',
      'vllm.model_executor.layers.fused_moe','vllm.model_executor.layers.fused_moe.oracle',
      'vllm.model_executor.layers.fused_moe.prepare_finalize','vllm.model_executor.layers.quantization',
      'vllm.model_executor.layers.quantization.utils','vllm_hcu','vllm_hcu.platforms','vllm_hcu.patch',
      'vllm_hcu.patch.worker','vllm_hcu.patch.worker.op_opt','vllm_hcu.patch.worker.op_opt.moe',
      'vllm_hcu.model_executor','vllm_hcu.model_executor.layers','vllm_hcu.model_executor.layers.fused_moe',
      'vllm_hcu.model_executor.layers.fused_moe.experts')
    for name in packages:
        m=module(mp,name)
        if name.startswith('vllm_hcu'):m.__path__=[str(HCU.joinpath(*name.split('.')))]
    sys.modules['vllm.model_executor.layers.fused_moe.oracle'].__path__=[str(CORE/'vllm/model_executor/layers/fused_moe/oracle')]
    logger=logging.getLogger('unquantized-abi-cpu');logger.info_once=logger.info;logger.warning_once=logger.warning
    platform=NS(is_rocm=lambda:False,is_cuda=lambda:False,is_cuda_alike=lambda:False,is_cpu=lambda:True,is_xpu=lambda:False)
    module(mp,'vllm.platforms',current_platform=platform)
    module(mp,'vllm.envs',VLLM_ROCM_MOE_PADDING=False)
    module(mp,'vllm.logger',init_logger=lambda name:logger)
    module(mp,'vllm.config.kernel',MoEBackend=str)
    activation=module(mp,'vllm.model_executor.layers.fused_moe.activation',MoEActivation=enum.Enum('MoEActivation',{'SILU':'silu'}))
    cfg=module(mp,'vllm.model_executor.layers.fused_moe.config',FusedMoEConfig=type('FusedMoEConfig',(),{}),FusedMoEParallelConfig=type('FusedMoEParallelConfig',(),{}),FusedMoEQuantConfig=type('FusedMoEQuantConfig',(),{}),RoutingMethodType=NS(Default='default'),biased_moe_quant_config=lambda *a,**kw:None)
    mk=module(mp,'vllm.model_executor.layers.fused_moe.modular_kernel',FusedMoEExperts=type('FusedMoEExperts',(),{}),FusedMoEKernel=type('FusedMoEKernel',(),{}),FusedMoEExpertsModular=ExpertBoundary,FusedMoEActivationFormat=enum.Enum('FusedMoEActivationFormat',{'Standard':'standard','BatchedExperts':'batched'}))
    module(mp,'vllm.model_executor.layers.fused_moe.oracle.base',MoEKernelOracle=OracleBoundary)
    module(mp,'vllm.model_executor.layers.fused_moe.all2all_utils',maybe_make_prepare_finalize=lambda *a,**kw:None)
    module(mp,'vllm._aiter_ops',rocm_aiter_ops=NS())
    module(mp,'vllm.utils.math_utils',round_up=lambda a,b:((a+b-1)//b)*b)
    module(mp,'vllm.model_executor.layers.quantization.utils.flashinfer_utils',align_moe_weights_for_fi=lambda *a:None,convert_moe_weights_to_flashinfer_trtllm_block_layout=lambda *a:None,swap_w13_to_w31=lambda x:x)
    module(mp,'vllm.model_executor.layers.fused_moe.fused_moe_method_base',FusedMoEMethodBase=BaseMethod)
    module(mp,'vllm.model_executor.layers.fused_moe.moe_output',UnfinalizedMoEOutput=type('UnfinalizedMoEOutput',(),{}))
    module(mp,'vllm.model_executor.layers.fused_moe.runner')
    module(mp,'vllm.model_executor.layers.fused_moe.runner.shared_experts',SharedExperts=type('SharedExperts',(),{}))
    module(mp,'vllm.model_executor.layers.fused_moe.topk_weight_and_reduce',TopKWeightAndReduceNoOP=type('TopKWeightAndReduceNoOP',(),{}))
    module(mp,'vllm.model_executor.layers.quantization.utils.quant_utils',QuantKey=type('QuantKey',(),{}))
    module(mp,'vllm.model_executor.custom_op',CustomOp=CustomOpBoundary)
    def replace_parameter(layer,name,value,prefer_copy=False):
        if prefer_copy:getattr(layer,name).data.copy_(value)
        else:setattr(layer,name,torch.nn.Parameter(value,requires_grad=False))
    module(mp,'vllm.model_executor.utils',is_weights_pre_processed=lambda:False,replace_parameter=replace_parameter,set_weight_attrs=lambda *a,**kw:None)
    henvs=load(mp,'vllm_hcu.platforms.envs',HCU/'vllm_hcu/platforms/envs.py')
    common=load(mp,'vllm_hcu.patch.worker.op_opt._common',HCU/'vllm_hcu/patch/worker/op_opt/_common.py')
    load(mp,'vllm_hcu.patch.worker.op_opt.moe._common',HCU/'vllm_hcu/patch/worker/op_opt/moe/_common.py')
    state=load(mp,'vllm_hcu.patch.runtime_state',HCU/'vllm_hcu/patch/runtime_state.py')
    coordinator=load(mp,'vllm_hcu.patch.import_coordinator',HCU/'vllm_hcu/patch/import_coordinator.py')
    adapter=load(mp,ADAPTER,adapter_path or HCU/'vllm_hcu/patch/worker/op_opt/moe/patch_unquantized_oracle.py')
    worker=dict(importlib=importlib,lru_cache=__import__('functools').lru_cache)
    defs(HCU/'vllm_hcu/patch/worker/__init__.py',{'_load_adapter','_require_bool_result','_apply_and_verify','_callback_for'},worker)
    for name in (TARGET,'vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method',
                 'vllm_hcu.model_executor.layers.fused_moe.unquantized_fused_moe_method',
                 'vllm_hcu.model_executor.layers.fused_moe.lightop_w16a16_runtime',
                 'vllm_hcu.model_executor.layers.fused_moe.experts.lightop_w16a16_moe'):
        reserve(mp,name)
    registry=state.PatchRegistry();coord=coordinator.ExactImportCoordinator(registry=registry)
    return NS(mp=mp,adapter=adapter,common=common,state=state,coordinator=coordinator,coord=coord,registry=registry,worker=worker,platform=platform,henvs=henvs,mk=mk,activation=activation,cfg=cfg,target=None,capture={})


def native(r):
    if r.target is None:r.target=importlib.import_module(TARGET)
    return r.target


def snap(r):
    target=native(r)
    return (dict(target.__dict__),{name:dict(sys.modules[name].__dict__) for name in r.adapter._CONSUMER_MODULES if name in sys.modules})


def install(r):
    r.coord.install()
    worker_cb=r.worker['_callback_for'](ADAPTER,False,r.coord,r.adapter.PATCH_ID)
    def callback(target):
        r.target=target;r.capture['before']=snap(r);worker_cb(target)
    r.coord.register_callback(r.adapter.PATCH_ID,TARGET,callback,targets=r.adapter.TARGETS)
    target=native(r)
    assert r.registry.get(r.adapter.PATCH_ID).status.name=='APPLIED'
    after=snap(r)
    r.coord.register_callback(r.adapter.PATCH_ID,TARGET,callback,targets=r.adapter.TARGETS)
    assert snap(r)==after
    assert r.adapter.apply_to_module(target) is False
    assert snap(r)==after
    return target


@pytest.fixture
def route(monkeypatch):
    r=build_route(monkeypatch)
    try:yield r
    finally:
        r.coord.uninstall()
        assert not torch.cuda.is_initialized()


def test_real_worker_cold_import_registration_and_exact_five_arguments(route):
    r=route;t=install(r)
    assert t.__file__==str(CORE/'vllm/model_executor/layers/fused_moe/oracle/unquantized.py')
    for wrapped in (True,False):
        s=inspect.signature(t.convert_to_unquantized_kernel_format,follow_wrapped=wrapped)
        assert tuple(s.parameters)==('unquantized_backend','moe_config','w13_weight','w2_weight','layer')
        assert s.parameters['layer'].default is None
        assert s.parameters['layer'].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    OBSERVATIONS.append(dict(case='cold_registration',status='APPLIED',repeat=False))


@pytest.mark.parametrize('mutation',['unknown','missing','reordered','kwonly','posonly','required','default_false','default_zero','default_object'])
def test_native_signature_drift_rejected_before_target_or_alias_changes(route,mutation):
    r=route;t=native(r)
    path=CORE/'vllm/model_executor/layers/fused_moe/oracle/unquantized.py'
    f=copy.deepcopy(next(n for n in ast.parse(path.read_text()).body if getattr(n,'name',None)=='convert_to_unquantized_kernel_format'))
    if mutation=='unknown':f.args.args.append(ast.arg(arg='unknown'));f.args.defaults.append(ast.Constant(value=None))
    elif mutation=='missing':f.args.args.pop();f.args.defaults=[]
    elif mutation=='reordered':f.args.args[0],f.args.args[1]=f.args.args[1],f.args.args[0]
    elif mutation=='kwonly':f.args.kwonlyargs=[f.args.args.pop()];f.args.kw_defaults=f.args.defaults;f.args.defaults=[]
    elif mutation=='posonly':f.args.posonlyargs=f.args.args;f.args.args=[]
    elif mutation=='required':f.args.defaults=[]
    else:f.args.defaults=[ast.Constant(value=False if mutation=='default_false' else 0)] if mutation!='default_object' else [ast.Call(func=ast.Name(id='object',ctx=ast.Load()),args=[],keywords=[])]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[])),str(path),'exec'),t.__dict__)
    consumer=module(r.mp,r.adapter._CONSUMER_MODULES[0],convert_to_unquantized_kernel_format=t.convert_to_unquantized_kernel_format)
    before=snap(r)
    with pytest.raises(r.common.PatchCompatibilityError):install(r)
    assert r.registry.get(r.adapter.PATCH_ID).status.name=='FAILED' and snap(r)==before
    assert not getattr(t,r.adapter._MARKER,False)


@pytest.mark.parametrize('kind',['UnquantizedMoeBackend','backend_to_kernel_cls','map_unquantized_backend','select_unquantized_moe_backend','convert_to_unquantized_kernel_format','make_unquantized_moe_kernel'])
def test_stale_applied_bindings_rejected_without_repair(route,kind):
    r=route;t=install(r);setattr(t,kind,lambda:None);before=snap(r)
    with pytest.raises(r.common.PatchCompatibilityError,match='stale'):r.adapter.apply_to_module(t)
    assert snap(r)==before


@contextmanager
def native_layer_trace(target):
    original=target.convert_to_unquantized_kernel_format.__wrapped__
    seen=[];previous=sys.getprofile()
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is original.__code__:seen.append(frame.f_locals['layer'])
    sys.setprofile(profile)
    try:yield seen
    finally:sys.setprofile(previous)


@pytest.mark.parametrize('caller',['wrapper','native_oracle'])
@pytest.mark.parametrize('mode',['default','positional','keyword','falsey'])
def test_native_layer_identity_and_defaults(route,caller,mode):
    r=route;t=install(r);cfg=NS();w13=torch.ones(2,4);w2=torch.ones(4,2)
    layer=FalseyLayer() if mode=='falsey' else NS(tag='layer')
    args=(t.UnquantizedMoeBackend.CPU,cfg,w13,w2)
    fn=t.convert_to_unquantized_kernel_format if caller=='wrapper' else t.UnquantizedMoEKernelOracle().convert_to_kernel_format
    with native_layer_trace(t) as seen:
        if mode=='default':answer=fn(*args);expected=None
        elif mode=='positional':answer=fn(*args,layer);expected=layer
        else:answer=fn(*args,layer=layer);expected=layer
    assert seen==[expected] and (not seen or seen[0] is expected)
    assert answer[0] is w13 and answer[1] is w2
    OBSERVATIONS.append(dict(case=caller,mode=mode,native_layer_identity=True))


def moon_boundary(r):
    calls=[];layout=NS(full_gate_weight=torch.full((2,4),7.),full_down_weight=torch.full((4,2),9.))
    def gather(w13,w2,**kw):calls.append((w13,w2,kw));return layout
    module(r.mp,'vllm.model_executor.layers.fused_moe.prepare_finalize.moonep',MOONEP_DEFAULT_NUM_PREFETCH_SLOTS=3,gather_moonep_weight_layout=gather)
    return calls,layout


@pytest.mark.parametrize('falsey',[False,True])
@pytest.mark.parametrize('caller',['wrapper','native_oracle'])
def test_native_moonep_saves_layout_on_exact_layer(route,falsey,caller):
    r=route;t=install(r);calls,layout=moon_boundary(r);layer=FalseyLayer() if falsey else torch.nn.Module()
    args=(t.UnquantizedMoeBackend.MOONEP,NS(num_experts=8),torch.zeros(2,4),torch.zeros(4,2))
    fn=t.convert_to_unquantized_kernel_format if caller=='wrapper' else t.UnquantizedMoEKernelOracle().convert_to_kernel_format
    with native_layer_trace(t) as seen:answer=fn(*args,layer=layer)
    assert seen[0] is layer and layer._moonep_weight_layout is layout
    assert answer[0] is layout.full_gate_weight and answer[1] is layout.full_down_weight
    assert calls[0][2]==dict(num_global_experts=8,num_prefetch_slots=3)
    OBSERVATIONS.append(dict(case='native_moonep',falsey=falsey,caller=caller,identity=True))


@pytest.mark.parametrize('error',['missing','reload'])
def test_native_moonep_missing_layer_and_existing_layout_errors_preserved(route,error):
    r=route;t=install(r);calls,layout=moon_boundary(r)
    args=(t.UnquantizedMoeBackend.MOONEP,NS(num_experts=8),torch.zeros(2,4),torch.zeros(4,2))
    if error=='missing':
        with pytest.raises(ValueError,match='requires the layer module'):t.convert_to_unquantized_kernel_format(*args)
    else:
        layer=FalseyLayer();layer._moonep_weight_layout=layout;before=dict(layer.__dict__)
        with pytest.raises(NotImplementedError,match='in-place weight reloads'):t.UnquantizedMoEKernelOracle().convert_to_kernel_format(*args,layer=layer)
        assert layer.__dict__==before
    assert not calls


@pytest.mark.parametrize('falsey',[False,True])
def test_actual_native_postload_caller_passes_layer_to_moonep(route,falsey):
    r=route;t=install(r);calls,layout=moon_boundary(r)
    name=r.adapter._CONSUMER_MODULES[0]
    method_module=load(r.mp,name,CORE/'vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py')
    method=object.__new__(method_module.UnquantizedFusedMoEMethod)
    method.unquantized_backend=t.UnquantizedMoeBackend.MOONEP;method.moe_kernel=None
    method._maybe_pad_weight=lambda w:w
    setups=[]
    def setup(layer):
        setups.append(layer);method.moe_kernel=NS(fused_experts=NS(process_weights_after_loading=lambda layer:setups.append(layer)))
    method._init_moe_kernel=setup
    layer=FalseyLayer() if falsey else torch.nn.Module();layer.moe_config=NS(num_experts=8)
    layer.w13_weight=torch.nn.Parameter(torch.zeros(2,4));layer.w2_weight=torch.nn.Parameter(torch.zeros(4,2))
    with native_layer_trace(t) as seen:method.process_weights_after_loading(layer)
    assert method.base_postload_layer is layer and setups==[layer,layer]
    assert seen[0] is layer and layer._moonep_weight_layout is layout
    assert torch.equal(layer.w13_weight,layout.full_gate_weight) and torch.equal(layer.w2_weight,layout.full_down_weight)
    assert len(calls)==1
    OBSERVATIONS.append(dict(case='native_postload',falsey=falsey,layer_identity=True))


def test_preimport_consumer_aliases_and_foreign_binding_preserved(route):
    r=route;t=native(r);foreign=lambda *a:None
    native_consumer=module(r.mp,r.adapter._CONSUMER_MODULES[0],select_unquantized_moe_backend=t.select_unquantized_moe_backend,convert_to_unquantized_kernel_format=t.convert_to_unquantized_kernel_format,make_unquantized_moe_kernel=t.make_unquantized_moe_kernel)
    hcu_consumer=module(r.mp,r.adapter._CONSUMER_MODULES[1],select_unquantized_moe_backend=t.select_unquantized_moe_backend,convert_to_unquantized_kernel_format=foreign,make_unquantized_moe_kernel=t.make_unquantized_moe_kernel)
    old=t.UnquantizedMoeBackend;install(r)
    assert native_consumer.convert_to_unquantized_kernel_format is t.convert_to_unquantized_kernel_format
    assert hcu_consumer.convert_to_unquantized_kernel_format is foreign
    assert native_consumer.select_unquantized_moe_backend is hcu_consumer.select_unquantized_moe_backend is t.select_unquantized_moe_backend
    assert t._vllm_hcu_original_unquantized_moe_backend is old
    w13=torch.ones(2,4);w2=torch.ones(4,2)
    assert t.convert_to_unquantized_kernel_format(old.CPU,NS(),w13,w2)==(w13,w2)


LEGACY_CASES=[('test_w16a16_auto_selects_lightop_only_when_master_and_leaf_enabled',()),('test_w16a16_backend_to_kernel_cls_preserves_official_list_contract',()),('test_w16a16_no_config_preserves_original_selector_and_weights',()),('test_w16a16_selection_checks_decode_and_budget_boundary_configs',()),('test_w16a16_converter_packs_only_owned_backend',())]
LEGACY_CASES += [('test_w16a16_disabled_or_explicit_backend_preserves_official_selection',args) for args in ((False,True,'auto'),(True,False,'auto'),(True,True,'aiter'),(True,True,'triton'))]
LEGACY_CASES += [('test_w16a16_oracle_rejects_stale_applied_binding',(name,)) for name in ('UnquantizedMoeBackend','backend_to_kernel_cls','map_unquantized_backend','select_unquantized_moe_backend','convert_to_unquantized_kernel_format','make_unquantized_moe_kernel')]


@pytest.mark.parametrize('name,args',LEGACY_CASES)
def test_updated_actual_legacy_fixture_and_lightop_paths(route,name,args):
    r=route
    # Execute the actual selected legacy definitions, including their updated
    # shared fixture and all original assertions. No full giant-file suite claim.
    ns=dict(enum=enum,torch=torch,pytest=pytest,ModuleType=ModuleType,SimpleNamespace=NS,henvs=r.henvs,MoEActivation=r.activation.MoEActivation,RoutingMethodType=r.cfg.RoutingMethodType,FusedMoEActivationFormat=r.mk.FusedMoEActivationFormat)
    defs(HCU/'tests/runtime_patch/test_lightop_w16a16_moe.py',{'_module','_config','_target_module',name},ns)
    fn=ns[name]
    if 'monkeypatch' in inspect.signature(fn).parameters:fn(r.mp,*args)
    else:fn(*args)


def test_updated_legacy_fixture_preserves_falsey_layer_identity(route):
    r=route;ns=dict(enum=enum,ModuleType=ModuleType,SimpleNamespace=NS,FusedMoEActivationFormat=r.mk.FusedMoEActivationFormat)
    defs(HCU/'tests/runtime_patch/test_lightop_w16a16_moe.py',{'_module','_target_module'},ns)
    t,calls=ns['_target_module']();assert r.adapter.apply_to_module(t)
    layer=FalseyLayer();w13=torch.ones(1);w2=torch.ones(1)
    t.convert_to_unquantized_kernel_format(t.UnquantizedMoeBackend.AITER,NS(),w13,w2,layer=layer)
    assert next(value for kind,value in calls if kind=='convert_layer') is layer


def test_owned_lightop_packing_accepts_falsey_layer_without_native_conversion(route):
    r=route;t=install(r);layer=FalseyLayer();before=dict(layer.__dict__)
    w13=torch.arange(2*64*32,dtype=torch.float32).reshape(2,64,32)
    w2=torch.arange(2*32*32,dtype=torch.float32).reshape(2,32,32)
    a=w13.clone();b=w2.clone()
    with native_layer_trace(t) as seen:
        packed=t.convert_to_unquantized_kernel_format(t.UnquantizedMoeBackend.HCU_LIGHTOP_W16A16,NS(),w13,w2,layer=layer)
    assert not seen and layer.__dict__==before
    assert packed[0].shape==(2,2,1024) and packed[1].shape==(2,2,512)
    assert torch.equal(w13,a) and torch.equal(w2,b)


def test_actual_native_aiter_shuffle_and_weight_markers_preserved(route):
    r=route;t=install(r);layer=FalseyLayer();calls=[]
    ops=sys.modules['vllm._aiter_ops'].rocm_aiter_ops
    ops.shuffle_weights=lambda a,b:(calls.append((a,b)) or (a,b))
    cfg=NS(intermediate_size_per_partition_unpadded=4,intermediate_size_per_partition=4)
    w13=torch.ones(2,8,4);w2=torch.ones(2,4,4)
    with native_layer_trace(t) as seen:
        answer=t.convert_to_unquantized_kernel_format(t._vllm_hcu_original_unquantized_moe_backend.AITER,cfg,w13,w2,layer)
    assert seen[0] is layer and calls==[(w13,w2)]
    assert answer[0] is w13 and answer[1] is w2 and w13.is_shuffled and w2.is_shuffled


def test_foreign_hcu_enum_member_rejected_before_mutation(route):
    r=route;t=native(r)
    values={member.name:member.value for member in t.UnquantizedMoeBackend}
    values['HCU_LIGHTOP_W16A16']='already-registered'
    t.UnquantizedMoeBackend=enum.Enum('UnquantizedMoeBackend',values)
    before=snap(r)
    with pytest.raises(r.common.PatchCompatibilityError,match='already present'):install(r)
    assert snap(r)==before and r.registry.get(r.adapter.PATCH_ID).status.name=='FAILED'
