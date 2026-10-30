# SPDX-License-Identifier: Apache-2.0
"""Portable CPU metadata tests; no vLLM/Torch/PG/model import or execution."""
from __future__ import annotations

import builtins
from dataclasses import FrozenInstanceError, replace
import importlib.util
from pathlib import Path
from types import MappingProxyType
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'vllm_hcu/distributed/pcp8_group_descriptor.py'


class ReadOnly:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)

    def __setattr__(self, name, value):
        raise AssertionError(f'source mutation: {name}')

    def __delattr__(self, name):
        raise AssertionError(f'source deletion: {name}')


def changed(source, **updates):
    values = dict(vars(source))
    values.update(updates)
    return ReadOnly(**values)


def missing(source, name):
    values = dict(vars(source))
    del values[name]
    return ReadOnly(**values)


class Opaque:
    def __init__(self, label, calls):
        self.label = label
        self.calls = calls

    def __getattr__(self, name):
        def forbidden(*args, **kwargs):
            self.calls.append((self.label, name))
            raise AssertionError(f'PG/manager/collective/lifecycle call: {self.label}.{name}')
        return forbidden

    def __deepcopy__(self, memo):
        raise AssertionError('borrowed resource was copied')


class TargetClass:
    def __new__(cls, *args, **kwargs):
        raise AssertionError('target model constructed')


class DraftClass:
    def __new__(cls, *args, **kwargs):
        raise AssertionError('draft model constructed')


@pytest.fixture
def api(monkeypatch):
    name = '_pcp8_a1_descriptor_cpu'
    spec = importlib.util.spec_from_file_location(name, HELPER)
    result = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, result)
    spec.loader.exec_module(result)
    return result


def make_case(api, world_size=8, rank=0, anchor=True):
    calls = []
    device = Opaque('device-reference', calls)
    members = list(range((rank // 8) * 8, (rank // 8 + 1) * 8))
    def group(role, ranks):
        multi = role in ('pcp', 'ep')
        return ReadOnly(ranks=list(ranks), rank=rank, rank_in_group=list(ranks).index(rank),
                        world_size=len(ranks), torch_distributed_backend='nccl',
                        unique_name=f'{role}:provided', local_rank=1, device_index=1,
                        cpu_group=Opaque(role+'-cpu-pg', calls),
                        device_group=Opaque(role+'-device-pg', calls),
                        use_device_communicator=(role != 'world'),
                        device_communicator=Opaque(role+'-communicator', calls) if multi else None,
                        device=device)
    groups = api.GroupReferences(world=group('world', range(world_size)),
                                 tp=group('tp', [rank]), dp=group('dp', [rank]),
                                 pcp=group('pcp', members), ep=group('ep', members),
                                 pp=group('pp', [rank]), dcp=group('dcp', [rank]))
    def parallel(pcp, world):
        return ReadOnly(tensor_parallel_size=1, data_parallel_size=1,
                        prefill_context_parallel_size=pcp, pipeline_parallel_size=1,
                        decode_context_parallel_size=1, world_size=world,
                        enable_expert_parallel=True, enable_elastic_ep=False,
                        enable_eplb=False, use_sequence_parallel_moe=False,
                        _vllm_hcu_deepep_auto=False,
                        all2all_backend='deepep_high_throughput', data_parallel_rank=0,
                        _data_parallel_master_port_list=[29001,29002],
                        unrelated=Opaque('unrelated-config-reference', calls))
    target, draft = parallel(8, 8), parallel(1, 1)
    dtype = Opaque('actual-dtype-reference', calls)
    target_model = ReadOnly(enforce_eager=True, dtype=dtype)
    draft_model = ReadOnly(enforce_eager=True, dtype=dtype)
    spec = ReadOnly(method='dspark', num_speculative_tokens=5,
                    target_parallel_config=target, target_model_config=target_model,
                    draft_model_config=draft_model,
                    draft_parallel_config=ReadOnly(tensor_parallel_size=1,data_parallel_size=1,
                                                   enable_expert_parallel=False))
    kwargs = dict(mode=api.MODE, target_parallel_config=target,
                  native_draft_parallel_config=draft, target_model_config=target_model,
                  draft_model_config=draft_model, target_model_class=TargetClass,
                  draft_model_class=DraftClass, scheduler_config=ReadOnly(max_num_seqs=4,max_num_batched_tokens=64),
                  speculative_config=spec, groups=groups, sample_from_anchor=anchor,
                  factory_kwargs=MappingProxyType({'is_sequence_parallel': False}))
    return kwargs, calls


def replace_group(api, kwargs, role, **updates):
    groups = kwargs['groups']
    kwargs['groups'] = replace(groups, **{role: changed(getattr(groups,role), **updates)})


def source_fingerprint(kwargs):
    refs = [kwargs[key] for key in ('target_parallel_config','native_draft_parallel_config',
                                  'target_model_config','draft_model_config','scheduler_config','speculative_config')]
    refs += [getattr(kwargs['groups'], role) for role in ('world','tp','dp','pcp','ep','pp','dcp')]
    def value(v):
        if isinstance(v,list): return (id(v),tuple(v))
        if type(v) in (int,str,bool,type(None)): return (id(v),v)
        return (id(v),None)
    return tuple((id(ref),id(ref.__dict__),tuple((key,value(v)) for key,v in vars(ref).items())) for ref in refs)


@pytest.mark.parametrize('world_size,rank', [(8,0),(8,7),(16,9),(16,15)])
@pytest.mark.parametrize('anchor', [True,False])
def test_ordered_native_layout_and_conditional_bounds(api,world_size,rank,anchor):
    kwargs,calls=make_case(api,world_size,rank,anchor)
    before=source_fingerprint(kwargs)
    result=api.describe_pcp8_native_draft(**kwargs)
    assert source_fingerprint(kwargs)==before and calls==[]
    expected=tuple(range((rank//8)*8,(rank//8+1)*8))
    assert result.groups.world.global_rank==rank
    assert result.groups.world.rank_in_group==rank
    assert result.groups.pcp.members==result.groups.ep.members==expected
    assert result.groups.pcp.rank_in_group==result.groups.ep.rank_in_group==rank%8
    assert result.external_dp_index==rank//8
    assert result.target.pcp==8 and result.target.world_size==8
    assert result.native_draft.pcp==1 and result.native_draft.world_size==1
    assert result.groups.world.world_size==world_size
    assert (result.conditional_moe.tp,result.conditional_moe.dp,result.conditional_moe.pcp,result.conditional_moe.ep)==(1,1,8,8)
    assert result.conditional_moe.ep_rank==rank%8
    assert kwargs['speculative_config'].draft_parallel_config.enable_expert_parallel is False
    assert result.conditional_moe.ep==8  # Native loader does not copy that whole spec config.
    assert result.native_draft_backend=='deepep_high_throughput'
    assert result.future_expected_backend=='allgather_reducescatter'
    assert result.native_draft_backend!=result.future_expected_backend
    assert result.capacity.query_per_rank==(20 if anchor else 24)
    assert result.capacity.verify_per_rank==24 and result.capacity.local_token_buffer==64
    assert result.capacity.pcp_gather_rows_bound==8*result.capacity.query_per_rank
    assert result.future_agrs_transport is result.groups.pcp
    assert result.target.source is kwargs['target_parallel_config']
    assert result.native_draft.source is kwargs['native_draft_parallel_config']
    assert result.target_model_class is TargetClass and result.draft_model_class is DraftClass
    assert result.target_dtype is kwargs['target_model_config'].dtype
    assert result.scheduler_config is kwargs['scheduler_config']
    for role in ('world','tp','dp','pcp','ep','pp','dcp'):
        snapshot=getattr(result.groups,role);source=getattr(kwargs['groups'],role)
        assert snapshot.source is source
        assert snapshot.cpu_group is source.cpu_group and snapshot.device_group is source.device_group
        assert snapshot.device_communicator is source.device_communicator
        assert snapshot.device is source.device
    assert not hasattr(result,'validated') and not hasattr(result,'admitted')
    with pytest.raises(FrozenInstanceError): result.external_dp_index=99
    with pytest.raises(FrozenInstanceError): result.groups.ep.members=(0,)


def test_native_legal_world_and_singleton_none_communicators(api):
    kwargs,_=make_case(api,16,9)
    result=api.describe_pcp8_native_draft(**kwargs)
    for role in ('world','tp','dp','pp','dcp'):
        assert getattr(result.groups,role).device_communicator is None
    assert result.groups.pcp.device_communicator is not None
    assert result.groups.ep.device_communicator is not None


@pytest.mark.parametrize('role', ['world','tp','dp','pcp','ep','pp','dcp'])
def test_missing_group_has_no_singleton_fallback(api,role):
    kwargs,calls=make_case(api)
    kwargs['groups']=replace(kwargs['groups'],**{role:None})
    with pytest.raises(api.DescriptorNotReady,match='absent'):
        api.describe_pcp8_native_draft(**kwargs)
    assert calls==[]


@pytest.mark.parametrize('role', ['world','tp','dp','pcp','ep','pp','dcp'])
@pytest.mark.parametrize('resource', ['cpu_group','device_group'])
def test_missing_pg_reference_fails_even_for_singletons(api,role,resource):
    kwargs,calls=make_case(api)
    replace_group(api,kwargs,role,**{resource:None})
    with pytest.raises(api.DescriptorNotReady,match=resource): api.describe_pcp8_native_draft(**kwargs)
    assert calls==[]


@pytest.mark.parametrize('role', ['pcp','ep'])
def test_multi_rank_transport_communicator_is_required(api,role):
    kwargs,_=make_case(api)
    replace_group(api,kwargs,role,device_communicator=None)
    with pytest.raises(api.DescriptorNotReady,match='communicator'): api.describe_pcp8_native_draft(**kwargs)


@pytest.mark.parametrize('both', [False,True])
def test_same_set_and_same_size_wrong_order_is_not_native_layout(api,both):
    kwargs,_=make_case(api,16,9)
    reverse=list(reversed(range(8,16)))
    replace_group(api,kwargs,'ep',ranks=reverse,rank_in_group=reverse.index(9))
    if both: replace_group(api,kwargs,'pcp',ranks=reverse,rank_in_group=reverse.index(9))
    with pytest.raises(api.DescriptorMismatch,match='ordered native'):
        api.describe_pcp8_native_draft(**kwargs)


@pytest.mark.parametrize('role,changes', [
    ('ep',{'rank_in_group':9}), ('pcp',{'rank_in_group':0}),
    ('ep',{'world_size':7}), ('dp',{'rank':8}),
    ('pcp',{'ranks':[8,9,10,11,12,13,14,14]}),
    ('tp',{'ranks':[8],'rank':8,'rank_in_group':0}),
    ('ep',{'ranks':list(range(8)),'rank':1,'rank_in_group':1}),
])
def test_rank_size_and_native_group_mismatches_reject(api,role,changes):
    kwargs,calls=make_case(api,16,9)
    replace_group(api,kwargs,role,**changes)
    with pytest.raises(api.DescriptorMismatch): api.describe_pcp8_native_draft(**kwargs)
    assert calls==[]


@pytest.mark.parametrize('scenario', ['nonmultiple_world','world_order','world_rank'])
def test_world_native_layout_is_not_inferred_from_clone_world_one(api,scenario):
    kwargs,calls=make_case(api,16,9)
    if scenario=='nonmultiple_world':
        replace_group(api,kwargs,'world',ranks=list(range(12)),world_size=12,rank_in_group=9)
    elif scenario=='world_order':
        reverse=list(reversed(range(16)))
        replace_group(api,kwargs,'world',ranks=reverse,rank_in_group=reverse.index(9))
    else:
        replace_group(api,kwargs,'world',rank=8,rank_in_group=8)
    with pytest.raises((api.UnsupportedConstruction,api.DescriptorMismatch)):
        api.describe_pcp8_native_draft(**kwargs)
    assert calls==[]


@pytest.mark.parametrize('role,field', [('world','device_communicator'),('pcp','ranks'),
                                       ('tp','rank_in_group'),('ep','device_group')])
def test_absent_native_fields_are_not_synthesized(api,role,field):
    kwargs,_=make_case(api)
    kwargs['groups']=replace(kwargs['groups'],**{role:missing(getattr(kwargs['groups'],role),field)})
    with pytest.raises(api.DescriptorNotReady,match=field): api.describe_pcp8_native_draft(**kwargs)


def test_missing_effective_config_policy_is_not_defaulted(api):
    kwargs,_=make_case(api)
    kwargs['native_draft_parallel_config']=missing(kwargs['native_draft_parallel_config'],'_vllm_hcu_deepep_auto')
    with pytest.raises(api.DescriptorNotReady,match='_vllm_hcu_deepep_auto'):
        api.describe_pcp8_native_draft(**kwargs)


def test_real_native_backend_attribute_is_required_not_backend_alias(api):
    kwargs,_=make_case(api)
    group=missing(kwargs['groups'].ep,'torch_distributed_backend')
    group=changed(group,backend='nccl')
    kwargs['groups']=replace(kwargs['groups'],ep=group)
    with pytest.raises(api.DescriptorNotReady,match='torch_distributed_backend'):
        api.describe_pcp8_native_draft(**kwargs)


@pytest.mark.parametrize('role', ['world','pcp','ep'])
def test_wrong_group_backend_rejects(api,role):
    kwargs,_=make_case(api)
    replace_group(api,kwargs,role,torch_distributed_backend='gloo')
    with pytest.raises(api.UnsupportedConstruction,match='backend'): api.describe_pcp8_native_draft(**kwargs)


@pytest.mark.parametrize('which,field,value', [
    ('target_parallel_config','tensor_parallel_size',2),
    ('target_parallel_config','data_parallel_size',8),
    ('target_parallel_config','prefill_context_parallel_size',4),
    ('target_parallel_config','pipeline_parallel_size',2),
    ('target_parallel_config','decode_context_parallel_size',8),
    ('target_parallel_config','world_size',16),
    ('native_draft_parallel_config','prefill_context_parallel_size',8),
    ('native_draft_parallel_config','world_size',8),
    ('native_draft_parallel_config','data_parallel_size',8),
    ('target_parallel_config','enable_expert_parallel',False),
    ('target_parallel_config','enable_elastic_ep',True),
    ('target_parallel_config','enable_eplb',True),
    ('native_draft_parallel_config','use_sequence_parallel_moe',True),
    ('native_draft_parallel_config','_vllm_hcu_deepep_auto',True),
    ('target_parallel_config','all2all_backend','deepep_low_latency'),
    ('native_draft_parallel_config','all2all_backend','allgather_reducescatter'),
])
def test_unsupported_config_does_not_get_normalized_or_mutated(api,which,field,value):
    kwargs,calls=make_case(api)
    kwargs[which]=changed(kwargs[which],**{field:value})
    before=source_fingerprint(kwargs)
    with pytest.raises(api.UnsupportedConstruction): api.describe_pcp8_native_draft(**kwargs)
    assert source_fingerprint(kwargs)==before and calls==[]


@pytest.mark.parametrize('which', ['target_parallel_config','native_draft_parallel_config'])
def test_logical_dp_rank_is_not_global_rank(api,which):
    kwargs,_=make_case(api,16,9)
    kwargs[which]=changed(kwargs[which],data_parallel_rank=9)
    with pytest.raises(api.DescriptorMismatch,match='DP singleton'): api.describe_pcp8_native_draft(**kwargs)


@pytest.mark.parametrize('which', ['target_model_config','draft_model_config'])
def test_non_eager_model_rejects(api,which):
    kwargs,calls=make_case(api)
    kwargs[which]=changed(kwargs[which],enforce_eager=False)
    with pytest.raises(api.UnsupportedConstruction,match='non-eager'): api.describe_pcp8_native_draft(**kwargs)
    assert calls==[]


@pytest.mark.parametrize('case', ['mode','spec_method','factory_override','sp','eplb_override','redundant_override','missing_config_field','class_none','class_not_type','spec_target_ref','spec_target_model_ref','spec_model_ref'])
def test_missing_or_unsupported_construction_inputs(api,case):
    kwargs,calls=make_case(api)
    if case=='mode': kwargs['mode']='dp8'
    if case=='spec_method': kwargs['speculative_config']=changed(kwargs['speculative_config'],method='mtp')
    if case=='factory_override': kwargs['factory_kwargs']={'is_sequence_parallel':False,'pcp_size':1}
    if case=='sp': kwargs['factory_kwargs']={'is_sequence_parallel':True}
    if case=='eplb_override': kwargs['factory_kwargs']={'enable_eplb':True}
    if case=='redundant_override': kwargs['factory_kwargs']={'num_redundant_experts':8}
    if case=='missing_config_field': kwargs['native_draft_parallel_config']=missing(kwargs['native_draft_parallel_config'],'world_size')
    if case=='class_none': kwargs['draft_model_class']=None
    if case=='class_not_type': kwargs['target_model_class']=object()
    if case=='spec_target_ref': kwargs['speculative_config']=changed(kwargs['speculative_config'],target_parallel_config=object())
    if case=='spec_target_model_ref':
        actual_target=ReadOnly(enforce_eager=False,dtype=Opaque('different-target-dtype',calls))
        kwargs['speculative_config']=changed(kwargs['speculative_config'],target_model_config=actual_target)
    if case=='spec_model_ref': kwargs['speculative_config']=changed(kwargs['speculative_config'],draft_model_config=object())
    with pytest.raises((api.UnsupportedConstruction,api.DescriptorNotReady,api.DescriptorMismatch)):
        api.describe_pcp8_native_draft(**kwargs)
    assert calls==[]


@pytest.mark.parametrize('raw', [{}, {'is_sequence_parallel':False,'enable_eplb':False,'num_redundant_experts':0},
                                  {'tp_size':None,'dp_size':None,'pcp_size':None}])
def test_raw_factory_native_defaults_are_not_group_size_fallbacks(api,raw):
    kwargs,calls=make_case(api,16,9)
    kwargs['factory_kwargs']=MappingProxyType(raw)
    before=dict(raw)
    result=api.describe_pcp8_native_draft(**kwargs)
    assert raw==before and calls==[]
    assert result.conditional_moe.sp==1 and result.conditional_moe.ep==8
    assert result.groups.dp.members==(9,) and result.groups.ep.members==tuple(range(8,16))


@pytest.mark.parametrize('value', [None, 8])
def test_native_loader_spec_draft_tp_cannot_contradict_logical_input(api,value):
    kwargs,calls=make_case(api)
    spec_parallel=None if value is None else ReadOnly(tensor_parallel_size=value)
    kwargs['speculative_config']=changed(kwargs['speculative_config'],draft_parallel_config=spec_parallel)
    with pytest.raises((api.DescriptorNotReady,api.DescriptorMismatch)):
        api.describe_pcp8_native_draft(**kwargs)
    assert calls==[]


@pytest.mark.parametrize('field,value', [('max_num_seqs',0),('max_num_batched_tokens',0),('max_num_batched_tokens',23)])
def test_conditional_capacity_never_enlarges_scheduler(api,field,value):
    kwargs,calls=make_case(api)
    kwargs['scheduler_config']=changed(kwargs['scheduler_config'],**{field:value})
    before=source_fingerprint(kwargs)
    with pytest.raises((api.DescriptorMismatch,api.UnsupportedConstruction)):
        api.describe_pcp8_native_draft(**kwargs)
    assert source_fingerprint(kwargs)==before and calls==[]


def test_import_and_call_have_no_production_import_or_registry_side_effect(monkeypatch):
    name='_pcp8_a1_no_production_import'
    before=set(sys.modules)
    original=builtins.__import__
    def guard(module,*args,**kwargs):
        if module.split('.')[0] in ('vllm','vllm_hcu','torch'):
            raise AssertionError('production import: '+module)
        return original(module,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    spec=importlib.util.spec_from_file_location(name,HELPER)
    module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,module)
    spec.loader.exec_module(module)
    kwargs,calls=make_case(module,16,9)
    before_values=source_fingerprint(kwargs)
    descriptor=module.describe_pcp8_native_draft(**kwargs)
    assert descriptor.groups.ep.rank_in_group==1
    assert source_fingerprint(kwargs)==before_values and calls==[]
    added=set(sys.modules)-before
    assert not any(key.split('.')[0] in ('vllm','vllm_hcu','torch') for key in added)
