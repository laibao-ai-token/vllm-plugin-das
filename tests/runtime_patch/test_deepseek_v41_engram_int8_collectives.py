# SPDX-License-Identifier: Apache-2.0
"""Actual Engram and HCU Linear source bodies; collectives/vendor ops are CPU stubs."""
from test_deepseek_v41_engram_int8_empty_wkv import (
    ast, cls, execute, install_scheme, route, CORE, HCU, SimpleNamespace, pytest, torch,
)

@pytest.mark.parametrize('dp,tp,shared,offload,sp', [
    (1,1,False,False,False), (2,1,False,True,False),
    (2,2,False,True,False), (2,2,True,True,False),
    (2,2,False,False,True), (2,2,True,True,True),
])
@pytest.mark.parametrize('tokens', [0, 2])
def test_engram_actual_int8_collective_order(route, dp, tp, shared, offload, sp, tokens):
    sn = install_scheme(route)
    events = route.events
    def gather(x, dim=-1):
        events.append('tp_gather_output' if dim == -1 else 'tp_gather_heads')
        return torch.cat((x, x), dim=dim)
    sn['tensor_model_parallel_all_gather'] = gather
    linear = HCU / 'vllm_hcu/model_executor/layers/linear.py'
    execute([cls(linear, 'ColumnParallelLinear', {'forward', '_forward_with_hcu_quanted'})], sn, linear)
    layer = sn['ColumnParallelLinear']()
    layer.__class__.__call__ = layer.__class__.forward
    layer.bias = None
    layer.skip_bias_add = layer.return_bias = False
    layer.gather_output = True
    layer.tp_size = 1 if sp else tp
    effective_dp = 1 if shared else dp
    k = effective_dp * tp * 2
    layer.weight = torch.ones((6 // layer.tp_size, k), dtype=torch.int8)
    layer.weight_scale = torch.ones((6 // layer.tp_size, 1))
    layer.params_dtype = torch.bfloat16
    layer.scheme = sn['CompressedTensorsW8A8Int8']()
    layer.quant_method = sn['CompressedTensorsLinearMethod']()
    class DP:
        world_size = 2
        rank_in_group = 0
        def all_gather(self, x, dim=0):
            events.append('dp_gather')
            return torch.cat((x, x), dim=dim)
    class Post:
        def __getitem__(self, grid):
            def run(hidden, kv, q, k, mask, output, *args, **kwargs):
                events.append('post')
                output.copy_(hidden + 1)
            return run
    class Select:
        def __getitem__(self, grid):
            def run(gathered, output, source_tokens, start, count, local_width, width, **kwargs):
                for token in range(output.shape[0]):
                    for col in range(width):
                        index = (col // local_width * source_tokens + start + token) * local_width + col % local_width
                        output.reshape(-1)[token * width + col] = gathered.reshape(-1)[index] if start + token < source_tokens else 0
            return run
    path = CORE / 'vllm/models/deepseek_v41/common/engram.py'
    ens = dict(torch=torch, triton=SimpleNamespace(cdiv=lambda a,b:(a+b-1)//b, next_power_of_2=lambda v:1 << (v-1).bit_length()),
               get_engram_dp_group=lambda: DP(), engram_gathered_num_tokens=lambda: 2,
               tensor_model_parallel_all_gather=gather, get_tensor_model_parallel_rank=lambda: 0,
               get_tensor_model_parallel_world_size=lambda: 2,
               _fused_engram_post_wkv_kernel=Post(), _engram_select_rows_kernel=Select())
    functions = [n for n in ast.parse(path.read_text()).body if getattr(n, 'name', '') in ('_gather_engram_rows', '_engram_select_rows')]
    execute([*functions, cls(path, 'Engram', {'prepare_embeddings', '_ready_rows', 'embed', 'forward'})], ens, path)
    engram = ens['Engram']()
    engram.dim = engram.hc_mult = 2
    engram.use_sequence_parallel = sp
    engram.eps = engram.clamp_value = 1e-6
    engram._batch = None
    engram._prefetch_stream = object() if offload else None
    engram._prefetch_done = object() if offload else None
    engram._start_prefetch = lambda hashes, rows: events.append('lookup')
    engram._finish_prefetch = lambda event: events.append('prefetch_wait')
    heads = effective_dp * tp
    engram.embed_tokens = SimpleNamespace(dp_size=effective_dp, tp_size=tp, n_hash_cols=heads,
        cpu_offload=offload, dp_shared_memory=shared, lookup=lambda hashes, rows: events.append('lookup'))
    engram.staged_rows = torch.ones((max(2 * effective_dp, tokens), 1, 2), dtype=torch.bfloat16)
    engram.wkv = layer
    engram.q_weight = engram.k_weight = torch.zeros((2, 2), dtype=torch.bfloat16)
    hidden = torch.ones(((tokens + tp - 1) // tp if sp else tokens, 2, 2), dtype=torch.bfloat16)
    hashes = torch.empty((tokens, heads), dtype=torch.int32)
    engram.prepare_embeddings(hashes)
    output = engram.forward(hidden, hashes)
    assert output.shape == hidden.shape and output.dtype == hidden.dtype
    expected = ['lookup']
    if offload: expected.append('prefetch_wait')
    if dp > 1 and not shared: expected.append('dp_gather')
    if tp > 1: expected.append('tp_gather_heads')
    if tokens: expected += ['quant', 'gemm']
    if tp > 1 and not sp: expected.append('tp_gather_output')
    if tokens:
        expected.append('post')
        torch.testing.assert_close(output, hidden + 1)
    assert events == expected
    assert not torch.cuda.is_initialized()
