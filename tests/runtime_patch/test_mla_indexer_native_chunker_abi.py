# SPDX-License-Identifier: Apache-2.0
"""Portable class-static indexer callback contracts; no native/core imports.

The adjacent callback and complete common helper execute unchanged. Small exact
API fixtures stand in for native algorithms here; handoff-only source probes
separately exercise the fixed native chunker and both actual self-call sites.
"""
from __future__ import annotations
import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

ROOT=Path(__file__).resolve().parents[2]


def _load(monkeypatch,name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adapter(monkeypatch):
    package=ModuleType('_indexer_static_contract');package.__path__=[]
    monkeypatch.setitem(sys.modules,package.__name__,package)
    common=ROOT/'vllm_hcu/patch/worker/op_opt/_common.py'
    _load(monkeypatch,package.__name__+'._common',common)
    return _load(monkeypatch,package.__name__+'.callback',ROOT/'vllm_hcu/patch/worker/op_opt/patch_mla_indexer.py')


def fixture(adapter,chunker=None):
    calls=[]
    def original(compressed_seq_lens_cpu,prefill_query_lens_cpu,workspace_size,max_logits_bytes,request_offset=0):
        calls.append((compressed_seq_lens_cpu,prefill_query_lens_cpu,workspace_size,max_logits_bytes,request_offset))
        return [(slice(request_offset,request_offset+1),slice(0,0)),
                (slice(request_offset,request_offset+1),slice(0,2)),
                (slice(request_offset+1,request_offset+2),slice(3,2)),
                (slice(request_offset+1,request_offset+2),slice(2,5))]
    def split_decodes_and_prefills(common_attn_metadata,decode_threshold=1,require_uniform=False,treat_short_extends_as_decodes=True):
        return (1,0,1,0)
    class DeepseekV32IndexerMetadataBuilder:
        _split_indexer_prefill_chunks=staticmethod(chunker or original)
        def build(self,common_prefix_len,common_attn_metadata,fast_build=False):
            return NS()
    module=ModuleType(adapter.TARGET_MODULE)
    module.DeepseekV32IndexerMetadataBuilder=DeepseekV32IndexerMetadataBuilder
    module.split_decodes_and_prefills=split_decodes_and_prefills
    return module,DeepseekV32IndexerMetadataBuilder,calls


def snapshot(module,cls):
    return dict(vars(module)),dict(vars(cls))


def assert_unchanged(module,cls,before):
    assert dict(vars(module))==before[0]
    assert dict(vars(cls))==before[1]


def as_source(source):
    ns={};exec(source,ns);return ns['splitter']


def test_native_class_and_instance_descriptor_signature_filter_and_forward(adapter):
    module,cls,calls=fixture(adapter);original=vars(cls)['_split_indexer_prefill_chunks'].__func__
    assert adapter.apply_to_module(module) is True
    descriptor=vars(cls)['_split_indexer_prefill_chunks']
    assert isinstance(descriptor,staticmethod)
    assert descriptor.__func__.__wrapped__ is original
    assert cls._split_indexer_prefill_chunks is cls()._split_indexer_prefill_chunks
    signature=inspect.signature(cls._split_indexer_prefill_chunks)
    assert list(signature.parameters)==['compressed_seq_lens_cpu','prefill_query_lens_cpu','workspace_size','max_logits_bytes','request_offset']
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in signature.parameters.values())
    assert type(signature.parameters['request_offset'].default) is int
    assert signature.parameters['request_offset'].default==0
    a,b=object(),object()
    expected=[(slice(7,8),slice(0,2)),(slice(8,9),slice(2,5))]
    assert cls._split_indexer_prefill_chunks(a,b,11,13,7)==expected
    assert cls()._split_indexer_prefill_chunks(compressed_seq_lens_cpu=a,prefill_query_lens_cpu=b,workspace_size=11,max_logits_bytes=13,request_offset=7)==expected
    assert calls==[(a,b,11,13,7)]*2
    assert not hasattr(module,'split_indexer_prefill_chunks')
    assert vars(cls)['_vllm_hcu_original_split_indexer_prefill_chunks'].__func__ is original
    after=snapshot(module,cls)
    assert adapter.apply_to_module(module) is False
    assert_unchanged(module,cls,after)
    assert adapter.TARGETS[0]==adapter.TARGET_MODULE+'.DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks'
    assert adapter.TARGETS[1:]==(adapter.TARGET_MODULE+'.split_decodes_and_prefills',adapter.TARGET_MODULE+'.DeepseekV32IndexerMetadataBuilder.build')


def test_default_offset_and_native_errors_propagate(adapter):
    module,cls,calls=fixture(adapter);adapter.apply_to_module(module)
    cls()._split_indexer_prefill_chunks(None,None,1,2)
    assert calls[-1][-1]==0
    def raising(compressed_seq_lens_cpu,prefill_query_lens_cpu,workspace_size,max_logits_bytes,request_offset=0):
        raise ArithmeticError('native chunker failure')
    module,cls,_=fixture(adapter,raising);adapter.apply_to_module(module)
    with pytest.raises(ArithmeticError,match='native chunker failure'):
        cls()._split_indexer_prefill_chunks(None,None,1,2)


@pytest.mark.parametrize('source',[
    'def splitter(seq_lens_cpu,query_lens_cpu,workspace_size,max_logits_bytes,request_offset=0): return []',
    'def splitter(compressed_seq_lens_cpu,prefill_query_lens_cpu,workspace_size,max_logits_bytes,*,request_offset=0): return []',
    'def splitter(compressed_seq_lens_cpu,/,prefill_query_lens_cpu,workspace_size,max_logits_bytes,request_offset=0): return []',
    'def splitter(self,compressed_seq_lens_cpu,prefill_query_lens_cpu,workspace_size,max_logits_bytes,request_offset=0): return []',
    'def splitter(compressed_seq_lens_cpu,prefill_query_lens_cpu,*args,request_offset=0): return []',
    'def splitter(compressed_seq_lens_cpu,prefill_query_lens_cpu,workspace_size,max_logits_bytes,request_offset): return []',
    'def splitter(compressed_seq_lens_cpu,prefill_query_lens_cpu,workspace_size,max_logits_bytes,request_offset=None): return []',
    'def splitter(compressed_seq_lens_cpu,prefill_query_lens_cpu,workspace_size,max_logits_bytes,request_offset=False): return []',
    'def splitter(compressed_seq_lens_cpu,prefill_query_lens_cpu,workspace_size,max_logits_bytes,request_offset=1): return []',
    'def splitter(compressed_seq_lens_cpu,prefill_query_lens_cpu,workspace_size,max_logits_bytes): return []',
    'def splitter(compressed_seq_lens_cpu,prefill_query_lens_cpu,workspace_size,max_logits_bytes,request_offset=0,extra=None): return []',
])
def test_five_parameter_kind_name_and_default_drift_rejects_atomically(adapter,source):
    module,cls,_=fixture(adapter,as_source(source));before=snapshot(module,cls)
    with pytest.raises(adapter.PatchCompatibilityError):adapter.apply_to_module(module)
    assert_unchanged(module,cls,before)


@pytest.mark.parametrize('kind',['missing','function','classmethod','inherited','noncallable'])
def test_own_static_descriptor_required_without_module_fallback(adapter,kind):
    module,cls,_=fixture(adapter);function=vars(cls)['_split_indexer_prefill_chunks'].__func__
    if kind=='missing':delattr(cls,'_split_indexer_prefill_chunks')
    elif kind=='function':cls._split_indexer_prefill_chunks=function
    elif kind=='classmethod':cls._split_indexer_prefill_chunks=classmethod(function)
    elif kind=='noncallable':cls._split_indexer_prefill_chunks=staticmethod(None)
    else:
        original_cls=cls
        class Inherited(original_cls):pass
        cls=module.DeepseekV32IndexerMetadataBuilder=Inherited
    module.split_indexer_prefill_chunks=function  # Legacy export must not satisfy this contract.
    before=snapshot(module,cls)
    with pytest.raises(adapter.PatchCompatibilityError):adapter.apply_to_module(module)
    assert_unchanged(module,cls,before)


@pytest.mark.parametrize('gate',['split_batch','build'])
def test_later_gates_reject_before_static_wrapper_mutation(adapter,gate):
    module,cls,_=fixture(adapter)
    if gate=='split_batch':
        def drift(common_attn_metadata,decode_threshold=2,require_uniform=False,treat_short_extends_as_decodes=True):return (1,0,1,0)
        module.split_decodes_and_prefills=drift
    else:
        def drift(self,common_prefix_len,common_attn_metadata,fast_build=False,extra=None):return NS()
        cls.build=drift
    before=snapshot(module,cls)
    with pytest.raises(adapter.PatchCompatibilityError):adapter.apply_to_module(module)
    assert_unchanged(module,cls,before)


@pytest.mark.parametrize('target',['chunker','split_batch','build'])
def test_partial_wrapper_without_owner_marker_rejects(adapter,target):
    module,cls,_=fixture(adapter)
    fn=(cls._split_indexer_prefill_chunks if target=='chunker' else module.split_decodes_and_prefills if target=='split_batch' else cls.build)
    setattr(fn,adapter._WRAPPER,True);before=snapshot(module,cls)
    with pytest.raises(adapter.PatchCompatibilityError,match='without its module marker'):
        adapter.apply_to_module(module)
    assert_unchanged(module,cls,before)


@pytest.mark.parametrize('target',['descriptor','chunker','split_batch','build'])
def test_stale_repeated_application_rejects(adapter,target):
    module,cls,_=fixture(adapter);adapter.apply_to_module(module)
    if target=='descriptor':cls._split_indexer_prefill_chunks=cls._split_indexer_prefill_chunks
    elif target=='chunker':delattr(cls._split_indexer_prefill_chunks,adapter._WRAPPER)
    elif target=='split_batch':delattr(module.split_decodes_and_prefills,adapter._WRAPPER)
    else:delattr(cls.build,adapter._WRAPPER)
    before=snapshot(module,cls)
    with pytest.raises(adapter.PatchCompatibilityError):adapter.apply_to_module(module)
    assert_unchanged(module,cls,before)
