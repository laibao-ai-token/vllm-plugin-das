# SPDX-License-Identifier: Apache-2.0
"""Real fixed-core replace/get_field + real Pydantic/HCU bind, reduced CPU schemas.

This does not import the full >=3.11 core or construct real ParallelConfig's
platform/distributed validators. Native DFlash stops at a deliberate base-init
boundary before GPU allocation; model loading in the native loader is a stub.
"""
from __future__ import annotations
import ast
import copy
from contextlib import contextmanager
from dataclasses import MISSING, field, fields, is_dataclass
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, cast

import pydantic
from pydantic import ConfigDict, Field, ValidationError
from pydantic.dataclasses import dataclass
from pydantic.fields import FieldInfo
from typing_extensions import dataclass_transform
import pytest
import torch

HCU = Path(os.environ.get('VLLM_HCU_SOURCE_ROOT', Path(__file__).resolve().parents[2]))
_DEFAULT_CORE = next((path for path in (HCU.parent / 'vllm-community-main',
                                      HCU.parent.parent / 'vllm-community-main')
                      if (path / 'vllm/config/utils.py').is_file()),
                     HCU.parent / 'vllm-community-main')
CORE = Path(os.environ.get('VLLM_SOURCE_ROOT', _DEFAULT_CORE))
ROOT = Path(os.environ.get('HCU_REPLACE_CANDIDATE_ROOT', HCU))


def module(monkeypatch, name, **attrs):
    value = ModuleType(name)
    value.__path__ = []
    value.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, value)
    return value


def load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, value)
    spec.loader.exec_module(value)
    return value


def execute(nodes, ns, path):
    tree = ast.Module(body=[ast.parse('from __future__ import annotations').body[0], *copy.deepcopy(nodes)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)


@pytest.fixture
def route(monkeypatch):
    assert not torch.cuda.is_initialized()
    for name in ['vllm', 'vllm.config', 'vllm.compilation', 'vllm.model_executor',
                 'vllm.model_executor.models', 'vllm.v1', 'vllm.v1.worker',
                 'vllm.v1.worker.gpu', 'vllm.v1.worker.gpu.spec_decode',
                 'vllm.v1.worker.gpu.spec_decode.eagle', 'vllm.v1.worker.gpu.spec_decode.dspark',
                 'vllm_hcu', 'vllm_hcu.patch', 'vllm_hcu.patch.platform',
                 'vllm_hcu.patch.platform.core_fix']:
        module(monkeypatch, name)
    state = load(monkeypatch, 'vllm_hcu.patch.runtime_state', HCU / 'vllm_hcu/patch/runtime_state.py')
    common = load(monkeypatch, 'vllm_hcu.patch.platform.core_fix._common', HCU / 'vllm_hcu/patch/platform/core_fix/_common.py')
    hcu = load(monkeypatch, 'vllm_hcu.patch.config', ROOT / 'vllm_hcu/patch/config.py')
    path = CORE / 'vllm/config/utils.py'
    ns = dict(MISSING=MISSING, field=field, fields=fields, is_dataclass=is_dataclass,
              ConfigDict=ConfigDict, FieldInfo=FieldInfo, PydanticField=Field, dataclass=dataclass,
              dataclass_transform=dataclass_transform, cast=cast, Callable=Callable, Any=Any,
              logger=SimpleNamespace(warning_once=lambda *a: None))
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
             and n.name in ('config', 'get_field', 'is_init_field', 'replace')
             and not any(isinstance(d, ast.Name) and d.id == 'overload' for d in n.decorator_list)]
    execute(nodes, ns, path)
    utils = module(monkeypatch, 'vllm.config.utils', **{k: ns[k] for k in ('config', 'get_field', 'is_init_field', 'replace')})
    original = utils.replace
    sys.modules['vllm.config'].replace = original
    mutations_enabled = [False]
    parallel_path = CORE / 'vllm/config/parallel.py'
    parallel_tree = ast.parse(parallel_path.read_text())
    port_pop = next(n for n in ast.walk(parallel_tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Attribute) and t.attr == 'data_parallel_master_port' for t in n.targets)
                    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                    and n.value.func.attr == 'pop')
    communicator = next(n for n in ast.walk(parallel_tree) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Attribute) and t.attr == 'communicator' for t in n.targets)
                        and isinstance(n.value, ast.Constant) and n.value.value == 'torch_gloo')
    def native_constructor_mutations(instance):
        if mutations_enabled[0]:
            execute([port_pop, communicator], {'self': instance}, parallel_path)
    schema = dict(config=ns['config'], Field=Field, SimpleNamespace=SimpleNamespace, torch=torch,
                  _native_constructor_mutations=native_constructor_mutations)
    exec(compile('''from __future__ import annotations
@config
class EPLBConfig:
    num_redundant_experts: int = Field(default=2, ge=0)
    communicator: str | None = None
@config
class ParallelConfig:
    tensor_parallel_size: int = Field(default=1, gt=0)
    prefill_context_parallel_size: int = Field(default=8, gt=0)
    pipeline_parallel_size: int = Field(default=1, gt=0)
    data_parallel_size: int = Field(default=1, gt=0)
    data_parallel_rank: int = 0
    decode_context_parallel_size: int = 1
    enable_expert_parallel: bool = True
    enable_eplb: bool = True
    enable_elastic_ep: bool = False
    all2all_backend: str = 'deepep_low_latency'
    eplb_config: object = Field(default_factory=EPLBConfig)
    _data_parallel_master_port_list: list[int] = Field(default_factory=lambda: [29001, 29002, 29003])
    data_parallel_master_port: int = 0
    derived: int = Field(default=17, init=False)
    def __post_init__(self): _native_constructor_mutations(self)
@config
class ModelConfig:
    hf_config: object = Field(default_factory=lambda: SimpleNamespace(model_type='deepseek_v41',sample_from_anchor=True))
    use_mla: bool = True
    enforce_eager: bool = True
    dtype: object = torch.bfloat16
    def get_vocab_size(self): return 16
@config
class SpeculativeConfig:
    method: str = 'dspark'
    num_speculative_tokens: int = 5
    target_parallel_config: object = None
    draft_parallel_config: object = None
    draft_model_config: object = Field(default_factory=ModelConfig)
    attention_backend: object = None
    kv_cache_dtype: object = None
    moe_backend: str = 'deep_gemm'
@config
class VllmConfig:
    parallel_config: object
    speculative_config: object
    model_config: object = Field(default_factory=ModelConfig)
    kernel_config: object = Field(default_factory=lambda: SimpleNamespace(moe_backend='deep_gemm'))
    scheduler_config: object = Field(default_factory=lambda: SimpleNamespace(max_num_seqs=4,max_num_batched_tokens=64))
    compilation_config: object = Field(default_factory=lambda: SimpleNamespace(static_forward_context={},static_all_moe_layers=[]))
    additional_config: dict = Field(default_factory=dict)
    attention_config: object = None
    cache_config: object = None
    load_config: object = None
    quant_config: object = None
@config
class AttentionConfig:
    backend: object = None
    use_non_causal: bool = False
''', '<reduced_real_pydantic_config_schemas>', 'exec'), schema)
    Parallel = schema['ParallelConfig']
    Parallel.__module__ = 'vllm.config.parallel'
    module(monkeypatch, 'vllm.config.parallel', ParallelConfig=Parallel, EPLBConfig=schema['EPLBConfig'])
    parallel = Parallel()
    spec = schema['SpeculativeConfig'](target_parallel_config=parallel, draft_parallel_config=Parallel(tensor_parallel_size=1))
    feature = hcu.HcuFeatureConfig(deepep_auto=True, moe_backend='deep_gemm',
                                  expert_map_path='/target/expert-map.json',
                                  eplb_disable_rearrange=True, eplb_static_dispatch_policy='locality_fair')
    cfg = schema['VllmConfig'](parallel_config=parallel,speculative_config=spec,
                               additional_config={'hcu': feature.to_dict()},attention_config=schema['AttentionConfig']())
    # Execute the existing actual HCU EPLB binder, then the exact deepep-auto
    # setattr statement from patch_vllm_config. Do not invent fixture-only keys.
    hcu.bind_hcu_eplb_config(cfg)
    binding_path = HCU / 'vllm_hcu/patch/platform/core_fix/patch_vllm_config.py'
    binding_tree = ast.parse(binding_path.read_text())
    setter = next(n for n in ast.walk(binding_tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == 'setattr'
                  and len(n.args)>1 and isinstance(n.args[1], ast.Constant)
                  and n.args[1].value == '_vllm_hcu_deepep_auto')
    execute([ast.Expr(value=setter)], dict(parallel_config=parallel, feature_config=feature), binding_path)
    adapter = load(monkeypatch, 'vllm_hcu.patch.platform.core_fix.patch_config_replace',
                   ROOT / 'vllm_hcu/patch/platform/core_fix/patch_config_replace.py')
    native_path = CORE / 'vllm/v1/worker/gpu/spec_decode/dspark/utils.py'
    native_ns = dict(replace=original, logger=SimpleNamespace(warning_once=lambda *a: None, info_once=lambda *a: None),
                     dspark_draft_supports_eplb=lambda config: False)
    execute([n for n in ast.parse(native_path.read_text()).body if isinstance(n, ast.FunctionDef)], native_ns, native_path)
    native = module(monkeypatch, 'vllm.v1.worker.gpu.spec_decode.dspark.utils',
                    replace=original, _get_dspark_parallel_config=native_ns['_get_dspark_parallel_config'])
    # Give actual functions the actual caller module dictionary, so alias
    # replacement affects their global replace just as a normal import does.
    native.__dict__.update(native_ns)
    execute([n for n in ast.parse(native_path.read_text()).body if isinstance(n, ast.FunctionDef)], native.__dict__, native_path)
    return SimpleNamespace(utils=utils, original=original, hcu=hcu, adapter=adapter,
                           common=common, state=state, schema=schema, parallel=parallel,
                           cfg=cfg, spec=spec, native=native, monkeypatch=monkeypatch,
                           mutations_enabled=mutations_enabled)


def test_baseline_real_replace_rejects_real_hcu_bound_private_attributes(route):
    extra = set(vars(route.parallel)) - {f.name for f in fields(type(route.parallel))}
    assert extra == set(route.adapter._PRIVATE_FIELDS)
    with pytest.raises(ValueError, match='Field.*_vllm_hcu_'):
        route.original(route.parallel, prefill_context_parallel_size=1)


def test_limited_wrapper_preserves_native_fields_validation_and_target(route):
    assert route.adapter.apply_to_module(route.utils)
    assert route.adapter.apply_to_module(route.utils) is False
    before = dict(vars(route.parallel))
    route.parallel.derived = 99
    result = route.utils.replace(route.parallel,prefill_context_parallel_size=1,tensor_parallel_size=2)
    assert result is not route.parallel
    assert result.prefill_context_parallel_size == 1 and result.tensor_parallel_size == 2
    assert result.derived == 17  # native is_init_field behavior remains intact
    assert route.parallel.prefill_context_parallel_size == 8 and route.parallel.derived == 99
    for name in route.adapter._PRIVATE_FIELDS: assert getattr(result,name) == before[name]
    with pytest.raises(ValidationError): route.utils.replace(route.parallel,tensor_parallel_size=0)
    with pytest.raises(ValidationError): route.utils.replace(route.parallel,unknown_update=3)
    route.parallel.unrecognized_runtime_attribute = 7
    with pytest.raises(ValueError, match='unrecognized_runtime_attribute'):
        route.utils.replace(route.parallel,prefill_context_parallel_size=1)
    assert route.parallel.unrecognized_runtime_attribute == 7


@pytest.mark.parametrize('explicit_alias', [False, True])
def test_native_constructor_mutations_do_not_change_source_owned_values(route, explicit_alias):
    route.adapter.apply_to_module(route.utils)
    route.parallel.data_parallel_size = 8
    source_ports = list(route.parallel._data_parallel_master_port_list)
    source_eplb = route.parallel.eplb_config
    assert source_eplb.communicator is None
    route.mutations_enabled[0] = True
    updates = ({'eplb_config': source_eplb,
                '_data_parallel_master_port_list': route.parallel._data_parallel_master_port_list}
               if explicit_alias else {})
    result = route.utils.replace(route.parallel, prefill_context_parallel_size=1, **updates)
    assert route.parallel._data_parallel_master_port_list == source_ports
    assert source_eplb.communicator is None
    assert result._data_parallel_master_port_list == source_ports[:-1]
    assert result.data_parallel_master_port == source_ports[-1]
    assert result.eplb_config is not source_eplb
    assert result.eplb_config.communicator == 'torch_gloo'


def make_dflash(route):
    class StopAfterConfig(Exception): pass
    captured = []
    class DraftModelSpeculator:
        def __init__(self, vllm_config, device):
            captured.append(vllm_config)
            raise StopAfterConfig()
    ns = dict(copy=copy, torch=torch, DraftModelSpeculator=DraftModelSpeculator)
    path = CORE / 'vllm/v1/worker/gpu/spec_decode/dflash/speculator.py'
    cls = copy.deepcopy(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DFlashSpeculator'))
    cls.body = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__']
    value = module(route.monkeypatch,'vllm.v1.worker.gpu.spec_decode.dflash.speculator',replace=route.original,**ns)
    execute([cls],value.__dict__,path)
    return value,StopAfterConfig,captured


def test_actual_dflash_front_constructor_copy_before_any_producer_opt_in(route):
    dflash,stop,captured=make_dflash(route)
    with pytest.raises(ValueError,match='_vllm_hcu_'): dflash.DFlashSpeculator(route.cfg,'cpu')
    assert not captured
    route.adapter.apply_to_module(route.utils)
    with pytest.raises(stop): dflash.DFlashSpeculator(route.cfg,'cpu')
    copied=captured[0]
    assert copied is not route.cfg and copied.parallel_config is not route.parallel
    assert copied.parallel_config.prefill_context_parallel_size==1
    assert route.parallel.prefill_context_parallel_size==8
    for name in route.adapter._PRIVATE_FIELDS:
        assert getattr(copied.parallel_config,name)==getattr(route.parallel,name)
    assert not torch.cuda.is_initialized()


def test_actual_dspark_parallel_copy_and_explicit_eplb_updates(route):
    with pytest.raises(ValueError,match='_vllm_hcu_'):
        route.native._get_dspark_parallel_config(route.parallel,1,route.spec.draft_model_config)
    route.adapter.apply_to_module(route.utils)
    result=route.native._get_dspark_parallel_config(route.parallel,1,route.spec.draft_model_config)
    assert result.pipeline_parallel_size==1 and result.tensor_parallel_size==1
    assert result.enable_eplb is False and result.eplb_config.num_redundant_experts==0
    assert route.parallel.enable_eplb is True and route.parallel.eplb_config.num_redundant_experts==2
    for name in route.adapter._PRIVATE_FIELDS: assert getattr(result,name)==getattr(route.parallel,name)


def test_native_loader_uses_real_replace_for_root_and_parallel_objects(route):
    route.adapter.apply_to_module(route.utils)
    seen=[]
    @contextmanager
    def tag(name): yield
    module(route.monkeypatch,'vllm.compilation.backends',set_model_tag=tag)
    def get_model(*,vllm_config,model_config):
        seen.append(vllm_config)
        return SimpleNamespace(model=SimpleNamespace(),lm_head=None)
    module(route.monkeypatch,'vllm.model_executor.model_loader',get_model=get_model)
    module(route.monkeypatch,'vllm.model_executor.models.qwen3_dflash',dflash_has_any_non_causal=lambda cfg: False)
    module(route.monkeypatch,'vllm.model_executor.models.utils',get_draft_quant_config=lambda cfg: 'draft_quant')
    module(route.monkeypatch,'vllm.v1.worker.gpu.spec_decode.eagle.utils',
           _should_share=lambda *a: False,get_target_lm_head=lambda *a: None,maybe_share_target_embed=lambda *a: None)
    route.native.get_draft_load_config=lambda cfg: cfg.load_config
    route.native.get_pp_safe_draft_load_config=lambda cfg: cfg
    model=route.native.load_dspark_model(SimpleNamespace(model=SimpleNamespace()),route.cfg)
    assert model is not None and len(seen)==1
    copied=seen[0]
    assert copied is not route.cfg and copied.parallel_config is not route.parallel
    assert copied.quant_config=='draft_quant'
    assert copied.parallel_config.enable_eplb is False
    assert route.cfg.quant_config is None and route.parallel.enable_eplb is True


@pytest.mark.parametrize('mapping',[False,True])
def test_sidecar_update_keeps_all_five_parallel_aliases_authoritative(route,mapping):
    obj = {'additional_config':copy.deepcopy(route.cfg.additional_config),'parallel_config':{}}
    value=obj if mapping else route.cfg
    route.hcu.set_hcu_config(value,deepep_auto=False,expert_map_path=None,
                             expert_map_record_path=None,eplb_disable_rearrange=False)
    p=value['parallel_config'] if mapping else value.parallel_config
    read=(lambda name:p[name]) if mapping else (lambda name:getattr(p,name))
    assert read('_vllm_hcu_deepep_auto') is False
    assert read('_vllm_hcu_expert_map_path') is None
    assert read('_vllm_hcu_expert_map_record_path') is None
    assert read('_vllm_hcu_eplb_disable_rearrange') is False
    assert read('_vllm_hcu_eplb_static_dispatch_policy')=='locality_fair'


def test_producer_draft_cleanup_and_native_copy_keep_aliases_consistent(route):
    runtime_path = ROOT / 'vllm_hcu/dspark_runtime.py'
    if not runtime_path.is_file():
        pytest.skip('standalone replace companion has no DSpark producer; copy contracts still run')
    route.adapter.apply_to_module(route.utils)
    route.parallel.prefill_context_parallel_size=1
    route.parallel.data_parallel_size=8
    runtime=load(route.monkeypatch,'vllm_hcu.dspark_runtime',runtime_path)
    draft=runtime.draft_config(route.cfg)
    copied=route.native._get_dspark_parallel_config(draft.parallel_config,1,route.spec.draft_model_config)
    assert route.hcu.get_hcu_config(draft).deepep_auto is False
    for p in (draft.parallel_config,copied):
        assert p._vllm_hcu_deepep_auto is False
        assert p._vllm_hcu_expert_map_path is None
        assert p._vllm_hcu_expert_map_record_path is None
        assert p._vllm_hcu_eplb_disable_rearrange is False
    assert route.parallel._vllm_hcu_deepep_auto is True
    assert route.parallel._vllm_hcu_expert_map_path=='/target/expert-map.json'
    assert route.hcu.get_hcu_config(route.cfg).expert_map_path=='/target/expert-map.json'


def test_nonparallel_unknown_aliases_remain_native_errors(route):
    route.adapter.apply_to_module(route.utils)
    other=route.schema['ModelConfig']()
    other._vllm_hcu_deepep_auto=True
    with pytest.raises(ValueError,match='_vllm_hcu_deepep_auto'): route.utils.replace(other)


def test_late_alias_repair_and_unknown_alias_rejection(route):
    route.adapter.apply_to_module(route.utils)
    dflash,_,_=make_dflash(route)
    assert route.adapter.apply_to_module(route.utils) is False
    assert dflash.replace is route.utils.replace
    dflash.replace=lambda *args,**kwargs: None
    current=route.utils.replace
    with pytest.raises(route.common.PatchCompatibilityError,match='alias'):
        route.adapter.apply_to_module(route.utils)
    assert route.utils.replace is current


def test_actual_coordinator_and_registry_register_loaded_utils_without_reentry(route):
    coordinator_module=load(route.monkeypatch,'vllm_hcu.patch.import_coordinator',HCU/'vllm_hcu/patch/import_coordinator.py')
    coordinator=coordinator_module.ExactImportCoordinator(registry=route.state.PATCH_REGISTRY)
    registration=coordinator.register_callback(route.adapter.PATCH_ID,route.adapter.TARGET_MODULE,
                                                route.adapter.apply_to_module,targets=route.adapter.TARGETS)
    record=route.state.PATCH_REGISTRY.get(route.adapter.PATCH_ID)
    assert record.status is route.state.PatchStatus.APPLIED
    wrapped=route.utils.replace
    coordinator.register_callback(route.adapter.PATCH_ID,route.adapter.TARGET_MODULE,
                                  route.adapter.apply_to_module,targets=route.adapter.TARGETS)
    assert route.utils.replace is wrapped
    copied=route.utils.replace(route.parallel,prefill_context_parallel_size=1)
    assert copied.prefill_context_parallel_size==1
    assert copied._vllm_hcu_expert_map_path=='/target/expert-map.json'


def test_direct_apply_convenience_owns_registry_only_outside_callback(route):
    assert route.adapter.apply(route.utils) is True
    assert route.state.PATCH_REGISTRY.get(route.adapter.PATCH_ID).status is route.state.PatchStatus.APPLIED
    assert route.adapter.apply(route.utils) is False


def test_updated_existing_strict_platform_inventory_from_actual_adapter_metadata(route):
    path=ROOT/'vllm_hcu/patch/platform/core_fix/__init__.py'
    tree=ast.parse(path.read_text())
    order=next(n.value for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='_ORDERED_ADAPTERS' for t in n.targets))
    adapters=[]
    for item in order.elts:
        rel=Path('vllm_hcu/patch/platform/core_fix')/(item.id+'.py')
        source=ROOT/rel if (ROOT/rel).exists() else HCU/rel
        metadata={}
        for node in ast.parse(source.read_text()).body:
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name) and target.id in ('PATCH_ID','TARGET_MODULE'):
                        metadata[target.id]=ast.literal_eval(node.value)
        adapters.append(SimpleNamespace(**metadata))
    ns={'_ORDERED_ADAPTERS':tuple(adapters)}
    execute([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='platform_core_callback_names'],ns,path)
    test_path=ROOT/'tests/patch/test_platform_dispatcher.py'
    execute([n for n in ast.parse(test_path.read_text()).body if isinstance(n,ast.FunctionDef)
             and n.name=='test_platform_core_inventory_is_explicit_and_ordered'],ns,test_path)
    ns['test_platform_core_inventory_is_explicit_and_ordered']()
