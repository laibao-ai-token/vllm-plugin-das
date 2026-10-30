# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""PCP lookback source contracts with real CPU history/position tensors."""
from __future__ import annotations
import ast
import copy
from dataclasses import dataclass
import functools
import hashlib
import importlib
import inspect
import linecache
import os
from pathlib import Path
import sys
import textwrap
from types import ModuleType, SimpleNamespace
import numpy as np
import pytest
import torch

HCU=Path(os.environ.get('VLLM_HCU_SOURCE_ROOT',Path(__file__).resolve().parents[2]))
CORE=Path(os.environ.get('VLLM_SOURCE_ROOT',HCU.parent.parent/'vllm-community-main'))
PATCH=Path(os.environ.get('HCU_LOOKBACK_PATCH_ROOT',HCU))
REL='vllm_hcu/patch/worker/core_fix/patch_deepseek_v41_model_state.py'


def execute(nodes,ns,path):
    module=ModuleType('lookback_source_'+str(len(sys.modules)));module.__dict__.update(ns)
    sys.modules[module.__name__]=module
    tree=ast.Module(body=[ast.parse('from __future__ import annotations').body[0],*copy.deepcopy(nodes)],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),module.__dict__)
    return module


def select(path,names,ns):
    return execute([n for n in ast.parse(path.read_text()).body if getattr(n,'name',None) in names],ns,path)


def model_class(path,name,methods,bases):
    node=copy.deepcopy(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name))
    node.bases=[ast.Name(id=b,ctx=ast.Load()) for b in bases]
    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in methods]
    return node


@pytest.fixture
def runtime():
    assert not torch.cuda.is_initialized()
    helper=select(HCU/'vllm_hcu/patch/worker/core_fix/_common.py',
        {'PatchCompatibilityError','load_exact_module','require_class','require_callable','require_exact_signature'},
        {'inspect':inspect,'importlib':importlib})
    patch_path=PATCH/REL
    ns={key:getattr(helper,key) for key in ('PatchCompatibilityError','load_exact_module','require_class','require_callable','require_exact_signature')}
    ns.update({'torch':torch,'functools':functools,'hashlib':hashlib,'inspect':inspect,'textwrap':textwrap,'ModuleType':ModuleType})
    patch=execute([n for n in ast.parse(patch_path.read_text()).body if not isinstance(n,(ast.Import,ast.ImportFrom))],ns,patch_path)
    default_path=CORE/'vllm/v1/worker/gpu/model_states/default.py'
    default_node=model_class(default_path,'DefaultModelState',{'prepare_inputs','prepare_dummy_inputs'},[])
    default=execute([default_node],{'torch':torch},default_path).DefaultModelState
    def base_init(self,vllm_config,model,encoder_cache,device):
        self.vllm_config=vllm_config
        self.max_num_reqs=vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens
        self.rope_state=None
        self.supports_mm_inputs=False
        self.prompt_embeds_state=None
    default.__init__=base_init  # constructor/environment boundary, not native runtime acceptance
    calls=[]
    class Gather:
        def __getitem__(self,grid):
            def run(window,indices,cursors,history,stride,count,**kwargs):
                calls.append('native_gather')
                window.fill_(-1)
                for row in range(count):
                    slot=int(indices[row]);start=int(cursors[slot])
                    for depth in range(window.shape[1]):
                        if start-1-depth>=0:window[row,depth]=history[slot,start-1-depth]
            return run
    native_path=CORE/'vllm/models/deepseek_v41/nvidia/model_state.py'
    native=model_class(native_path,'DeepseekV41ModelState',
        {'__init__','prepare_inputs','prepare_dummy_inputs','prepare_attn','add_request'},['DefaultModelState'])
    module=execute([native],{'torch':torch,'np':np,'DefaultModelState':default,
        'UvaBufferPool':lambda *a:object(),'_gather_lookback_kernel':Gather(),
        'triton':SimpleNamespace(next_power_of_2=lambda n:1 << (n-1).bit_length())},native_path)
    target=ModuleType(patch.TARGET_MODULE)
    target.DeepseekV41ModelState=module.DeepseekV41ModelState
    target.DefaultModelState=default
    originals={name:getattr(target.DeepseekV41ModelState,name) for name in ('__init__','prepare_inputs','prepare_dummy_inputs','prepare_attn','add_request')}
    def state(pcp=8,max_reqs=4,depth=3):
        config=SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
            scheduler_config=SimpleNamespace(max_num_seqs=max_reqs,max_num_batched_tokens=64),compilation_config=SimpleNamespace())
        return target.DeepseekV41ModelState(config,SimpleNamespace(token_lookback_depth=depth,decoder_replay_layers=None),None,'cpu')
    return SimpleNamespace(patch=patch,target=target,state=state,originals=originals,calls=calls,native_path=native_path)


def apply(runtime):
    assert runtime.patch.apply_to_module(runtime.target) is True
    assert runtime.patch.apply_to_module(runtime.target) is False
    for name in ('prepare_attn','prepare_dummy_inputs','add_request'):
        assert getattr(runtime.target.DeepseekV41ModelState,name) is runtime.originals[name]


def history(slots=4,width=64):
    values=torch.arange(slots*width,dtype=torch.int32).reshape(slots,width)
    return SimpleNamespace(all_token_ids=SimpleNamespace(gpu=values),
        num_computed_tokens=SimpleNamespace(gpu=torch.full((slots,),19,dtype=torch.int32)),
        prefill_len=SimpleNamespace(gpu=torch.full((slots,),64,dtype=torch.int32)))


def batch(positions,offsets,indices,padding=None):
    positions=torch.tensor(positions,dtype=torch.int64)
    return SimpleNamespace(num_reqs=len(indices),positions=positions,
        query_start_loc=torch.tensor(offsets,dtype=torch.int32),
        idx_mapping=torch.tensor(indices,dtype=torch.int32),
        num_computed_tokens_np=np.full(len(indices),55,dtype=np.int32),
        is_padding=torch.zeros(positions.shape,dtype=torch.bool) if padding is None else torch.tensor(padding,dtype=torch.bool))


def expected(history_tensor,slots,cursors,depth=3):
    return torch.tensor([[int(history_tensor[slot,start-1-d]) if start-1-d>=0 else -1
                          for d in range(depth)] for slot,start in zip(slots,cursors)],dtype=torch.int32)


@pytest.mark.parametrize('max_reqs',[1,2,3,4,256])
def test_capacity_matches_current_manager_and_native_replay_state_is_preserved(runtime,max_reqs):
    apply(runtime)
    value=runtime.state(max_reqs=max_reqs)
    assert value.lookback_token_ids.shape==(3*max_reqs,3)
    assert value._replay_start.shape==(max_reqs,) and value._replay_start_np.shape==(max_reqs,)
    assert value._replay is None and value.decoder_replay_layers is None
    assert value.lookback_token_ids.dtype==torch.int32
    assert '3 * int(scheduler_config.max_num_seqs)' in (HCU/'vllm_hcu/v1/pcp_manager.py').read_text()


def test_repeated_request_fragments_use_corrected_device_positions_not_host_mirror(runtime):
    apply(runtime);state=runtime.state(max_reqs=2)
    data=batch([0,1,5,6,11,12,0],[0,2,4,6,7],[1,1,1,-1],padding=[False]*6+[True])
    req=history();saved=data.num_computed_tokens_np.copy();pointer=state.lookback_token_ids.data_ptr()
    result=state.prepare_inputs(data,req)
    assert result['lookback_token_ids'] is state.lookback_token_ids
    torch.testing.assert_close(state.lookback_token_ids[:3],expected(req.all_token_ids.gpu,[1,1,1],[0,5,11]))
    assert torch.all(state.lookback_token_ids[3:]==-1)
    assert np.array_equal(data.num_computed_tokens_np,saved) and runtime.calls==[]
    data.positions[:2]+=2
    state.prepare_inputs(data,req)
    assert state.lookback_token_ids.data_ptr()==pointer
    torch.testing.assert_close(state.lookback_token_ids[0],expected(req.all_token_ids.gpu,[1],[2])[0])
    dummy=state.prepare_dummy_inputs(1,1)
    assert dummy['lookback_token_ids'] is state.lookback_token_ids and torch.all(state.lookback_token_ids==-1)


@pytest.mark.parametrize('length',[1,17,33,64])
def test_eight_rank_manager_intervals_feed_fragment_history(runtime,length):
    apply(runtime)
    path=HCU/'vllm_hcu/v1/pcp_manager.py';tree=ast.parse(path.read_text())
    segment=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RankSegment')
    manager=model_class(path,'HcuPCPManager',{'rank_segments'},[])
    code=execute([segment,manager],{'dataclass':dataclass},path)
    req=history(width=128)
    for rank in range(8):
        intervals=code.HcuPCPManager.rank_segments(length,pcp_rank=rank,pcp_size=8)
        actual=[s for s in intervals if s.num_tokens>0]
        positions=[];offsets=[0];slots=[];cursors=[];pad=[]
        for span in actual:
            positions.extend(range(2+span.start,2+span.stop));pad.extend([False]*span.num_tokens)
            offsets.append(len(positions));slots.append(1);cursors.append(2+span.start)
        # A padding-only row, as the manager uses for equalized MLA/EP shapes.
        positions.append(0);pad.append(True);offsets.append(len(positions));slots.append(-1)
        data=batch(positions,offsets,slots,pad);value=runtime.state(max_reqs=2)
        result=value.prepare_inputs(data,req)['lookback_token_ids']
        if actual:torch.testing.assert_close(result[:len(actual)],expected(req.all_token_ids.gpu,[1]*len(actual),cursors))
        assert torch.all(result[len(actual):]==-1)
    assert runtime.calls==[]


def test_native_request_cursor_before_and_fragment_cursor_after(runtime):
    data=batch([5,6,11,12],[0,2,4],[1,1]);req=history()
    before=runtime.state(pcp=8)
    old=before.prepare_inputs(data,req)['lookback_token_ids'][:2].clone()
    torch.testing.assert_close(old,expected(req.all_token_ids.gpu,[1,1],[19,19]))
    apply(runtime)
    after=runtime.state(pcp=8)
    new=after.prepare_inputs(data,req)['lookback_token_ids'][:2]
    torch.testing.assert_close(new,expected(req.all_token_ids.gpu,[1,1],[5,11]))
    assert not torch.equal(old,new)


def test_device_fragment_history_ignores_optimistic_host_cursor(runtime):
    # Slot 1 stores IDs 64..127. Fragments start at positions 5 and 11,
    # so the three preceding IDs are fixed independently of the gather code.
    data=batch([5,6,11,12],[0,2,4],[1,1]);req=history()
    window=torch.full((6,3),-1,dtype=torch.int32)
    runtime.patch._fill_fragment_lookback(window,data,req)
    correct=torch.tensor([[68,67,66],[74,73,72]],dtype=torch.int32)
    torch.testing.assert_close(window[:2],correct)
    # Host/device request cursors deliberately disagree with fragment starts.
    # Advancing either request-level cursor must not change this chunk's IDs.
    data.num_computed_tokens_np[:]=57
    req.num_computed_tokens.gpu.fill_(21)
    runtime.patch._fill_fragment_lookback(window,data,req)
    torch.testing.assert_close(window[:2],correct)
    assert torch.all(window[2:]==-1)


@pytest.mark.parametrize('depth',[0,3])
def test_pcp1_delegates_native_prepare_and_keeps_native_capacity(runtime,depth):
    apply(runtime);value=runtime.state(pcp=1,max_reqs=4,depth=depth)
    data=batch([3,4],[0,2],[1]);req=history()
    result=value.prepare_inputs(data,req)
    if depth:
        assert value.lookback_token_ids.shape==(4,3) and runtime.calls==['native_gather']
        torch.testing.assert_close(result['lookback_token_ids'][0],expected(req.all_token_ids.gpu,[1],[19])[0])
    else:
        assert value.lookback_token_ids is None and 'lookback_token_ids' not in result and runtime.calls==[]


def test_default_rope_position_result_and_inputs_are_not_rewritten(runtime):
    apply(runtime);value=runtime.state()
    marker=torch.tensor([[9,8,7]])
    calls=[]
    value.rope_state=SimpleNamespace(prepare_positions=lambda *a:calls.append(a),get_positions=lambda n:marker)
    data=batch([5,6],[0,2],[1]);data.num_tokens_after_padding=2
    req=history();result=value.prepare_inputs(data,req)
    assert result['positions'] is marker and calls[0][0] is data.idx_mapping
    assert calls[0][1] is data.query_start_loc and calls[0][3] is req.num_computed_tokens.gpu


@pytest.mark.parametrize('case',['capacity','offsets','slot','cursor','missing_padding',
                                'offsets_rank','mapping_rank','padding_dtype','positions_float','history_rank'])
def test_invalid_fragment_metadata_fails_explicitly(runtime,case):
    window=torch.empty((2,3),dtype=torch.int32);data=batch([5,6],[0,2],[1]);req=history()
    if case=='capacity':data.num_reqs=3
    elif case=='offsets':data.query_start_loc[-1]=9
    elif case=='slot':data.idx_mapping[0]=99
    elif case=='cursor':data.positions[0]=1000
    elif case=='missing_padding':data.is_padding=torch.empty(0,dtype=torch.bool)
    elif case=='offsets_rank':data.query_start_loc=data.query_start_loc.unsqueeze(0)
    elif case=='mapping_rank':data.idx_mapping=data.idx_mapping.unsqueeze(0)
    elif case=='padding_dtype':data.is_padding=data.is_padding.long()
    elif case=='positions_float':data.positions=data.positions.float()+0.5
    else:req.all_token_ids.gpu=req.all_token_ids.gpu.unsqueeze(0)
    with pytest.raises(RuntimeError):runtime.patch._fill_fragment_lookback(window,data,req)


def test_zero_tokens_clear_persistent_tail_without_history_read(runtime):
    window=torch.full((6,3),99,dtype=torch.int32)
    runtime.patch._fill_fragment_lookback(window,batch([], [0,0],[0]),object())
    assert torch.all(window==-1)


def test_pcp8_without_engram_lookback_keeps_native_empty_result(runtime):
    apply(runtime)
    value=runtime.state(pcp=8,depth=0)
    result=value.prepare_inputs(batch([3,4],[0,2],[1]),object())
    assert result=={} and value.lookback_token_ids is None and runtime.calls==[]


def test_zero_requests_clear_reused_window_without_history_access(runtime):
    window=torch.full((6,3),99,dtype=torch.int32)
    runtime.patch._fill_fragment_lookback(window,batch([], [0], []),object())
    assert torch.all(window==-1)


def test_actual_state_binding_and_lazy_registration_are_explicit():
    target='vllm.models.deepseek_v41.nvidia.model_state'
    exchange=(HCU/'vllm_hcu/patch/module_exchange.py').read_text()
    assert target not in exchange
    amd=(CORE/'vllm/models/deepseek_v41/amd/model.py').read_text()
    amd_tree=ast.parse(amd)
    binding=next(n for n in ast.walk(amd_tree) if isinstance(n,ast.ImportFrom)
                 and any(alias.name=='DeepseekV41ModelState' for alias in n.names))
    assert binding.level==2 and binding.module=='nvidia.model_state'
    worker=(PATCH/'vllm_hcu/patch/worker/__init__.py').read_text()
    exports=(PATCH/'vllm_hcu/patch/worker/core_fix/__init__.py').read_text()
    assert worker.count('_adapter("core_fix", "patch_deepseek_v41_model_state")')==1
    assert exports.count('patch_deepseek_v41_model_state')==2


def test_source_drift_rejected_and_marker_integrity_checked(runtime):
    original=runtime.target.DeepseekV41ModelState.prepare_inputs
    node=copy.deepcopy(next(n for n in ast.walk(ast.parse(runtime.native_path.read_text()))
        if isinstance(n,ast.FunctionDef) and n.name=='prepare_inputs'))
    node.body.insert(0,ast.Pass())
    text=ast.unparse(ast.fix_missing_locations(ast.Module(body=[ast.parse('from __future__ import annotations').body[0],node],type_ignores=[])))+'\n'
    filename='<pcp-lookback-source-drift>';linecache.cache[filename]=(len(text),None,text.splitlines(keepends=True),filename)
    ns={};exec(compile(text,filename,'exec'),ns)
    runtime.target.DeepseekV41ModelState.prepare_inputs=ns['prepare_inputs']
    with pytest.raises(runtime.patch.PatchCompatibilityError,match='unreviewed'):
        runtime.patch.apply_to_module(runtime.target)
    runtime.target.DeepseekV41ModelState.prepare_inputs=original
    apply(runtime)
    runtime.target.DeepseekV41ModelState.prepare_inputs=original
    with pytest.raises(runtime.patch.PatchCompatibilityError,match='stale'):
        runtime.patch.apply_to_module(runtime.target)
