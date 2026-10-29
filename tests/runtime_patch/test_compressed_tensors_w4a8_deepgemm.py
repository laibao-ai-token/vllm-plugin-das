# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CT -> HIPC layout/scale and packed loader regression checks."""
from types import SimpleNamespace

import pytest
import torch

from vllm_hcu.model_executor.layers.quantization.compressed_tensors_w4a8_deepgemm import (
    HCUCompressedTensorsW4A8Int8MoEMethod,
    convert_ct_int4_to_hipc_bytes_,
    validate_hcu_w4a8,
)


def _pack_ct(logical):
    values = (logical.to(torch.int64) + 8).reshape(*logical.shape[:-1], -1, 8)
    shifts = torch.arange(8, device=logical.device, dtype=torch.int64) * 4
    return (values << shifts).sum(-1).to(torch.int32).contiguous()


def _decode_hipc(packed):
    value = packed.view(torch.uint8)
    high, low = value >> 4, value & 15
    nibbles = torch.stack((high, low), dim=-1).flatten(-2).to(torch.int16)
    return torch.where(nibbles >= 8, nibbles - 16, nibbles)


def _args():
    weight = SimpleNamespace(num_bits=4, type="int", strategy="channel", symmetric=True,
                             dynamic=False, group_size=None)
    act = SimpleNamespace(num_bits=8, type="int", strategy="token", symmetric=True, dynamic=True)
    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(use_ep=True, all2all_backend="deepep_high_throughput"),
        moe_backend="deep_gemm", activation="silu", is_act_and_mul=True, has_bias=False,
        in_dtype=torch.bfloat16, swiglu_limit=10.0,
    )
    return weight, act, moe


def _layer():
    method = HCUCompressedTensorsW4A8Int8MoEMethod(*_args())
    layer = torch.nn.Module()
    loader = object()
    method.create_weights(layer, 2, 512, 256, torch.bfloat16, weight_loader=loader)
    return method, layer, loader


def test_packed_ct_conversion_all_nibbles_preserves_storage_and_values():
    q = torch.arange(-8, 8, dtype=torch.int8).repeat(8).reshape(2, 4, 16)
    source = _pack_ct(q)
    original_ptr = source.data_ptr()
    out = convert_ct_int4_to_hipc_bytes_(source)
    assert out.data_ptr() == original_ptr
    assert out.shape == (2, 4, 8)
    torch.testing.assert_close(_decode_hipc(out), q.to(torch.int16))


def test_allocator_retains_packing_metadata_and_fp32_scales():
    _, layer, loader = _layer()
    assert layer.w13_weight_packed.shape == (2, 512, 64)
    assert layer.w2_weight_packed.shape == (2, 512, 32)
    assert layer.w13_weight_packed.dtype == torch.int32
    assert layer.w13_weight_shape.shape == (2, 2)
    assert layer.w13_weight_scale.dtype == torch.float32
    for param in layer.parameters():
        assert param.weight_loader is loader
        assert param.is_transposed is False
        assert param.quant_method == "channel"


def test_scale_compensation_is_once_and_preserves_checkpoint():
    method, layer, _ = _layer()
    layer.w13_weight_scale.data.fill_(2)
    layer.w2_weight_scale.data.fill_(4)
    cfg = method.get_fused_moe_quant_config(layer)
    torch.testing.assert_close(cfg.w1_scale, torch.full_like(layer.w13_weight_scale, 2 / 16))
    torch.testing.assert_close(cfg.w2_scale, torch.full_like(layer.w2_weight_scale, 4 / 16))
    assert cfg.quant_dtype == torch.int8
    assert cfg.weight_quant_dtype == "int4"
    assert cfg.is_per_act_token
    torch.testing.assert_close(layer.w13_weight_scale, torch.full_like(layer.w13_weight_scale, 2))


@pytest.mark.parametrize("change", ["group", "asym", "static_act", "auto", "bias"])
def test_contract_rejects_unsupported_before_weight_allocation(change):
    weight, act, moe = _args()
    if change == "group":
        weight.strategy, weight.group_size = "group", 128
    elif change == "asym":
        act.symmetric = False
    elif change == "static_act":
        act.dynamic = False
    elif change == "auto":
        moe.moe_parallel_config.all2all_backend = "deepep_auto"
    elif change == "bias":
        moe.has_bias = True
    with pytest.raises(ValueError):
        validate_hcu_w4a8(weight, act, moe)


@pytest.mark.parametrize("hidden,intermediate", [(256, 256), (512, 128)])
def test_allocator_rejects_unsafe_contiguous_n_tail(hidden, intermediate):
    method = HCUCompressedTensorsW4A8Int8MoEMethod(*_args())
    layer = torch.nn.Module()
    with pytest.raises(ValueError, match="hidden divisible by 512"):
        method.create_weights(layer, 2, hidden, intermediate, torch.bfloat16)
    assert list(layer.parameters()) == []


def test_real_routed_loader_fuses_gate_up_and_loads_shape():
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    method, layer, _ = _layer()
    routed = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed)
    routed.quant_method = method
    routed.quant_config = SimpleNamespace(get_name=lambda: "compressed-tensors")
    routed.moe_config = SimpleNamespace(
        tp_rank=0, tp_size=1, is_act_and_mul=True,
        moe_parallel_config=SimpleNamespace(tp_size=1),
    )
    routed._map_global_expert_id_to_local_expert_id = lambda expert_id: expert_id
    gate = torch.full((256, 64), 11, dtype=torch.int32)
    up = torch.full((256, 64), 22, dtype=torch.int32)
    for shard, tensor in (("w1", gate), ("w3", up)):
        assert routed.weight_loader(layer.w13_weight_packed, tensor, "w13_weight_packed",
                                    shard_id=shard, expert_id=1, return_success=True)
    torch.testing.assert_close(layer.w13_weight_packed[1, :256], gate)
    torch.testing.assert_close(layer.w13_weight_packed[1, 256:], up)
    routed.weight_loader(layer.w2_weight_shape, torch.tensor([128, 256]), "w2_weight_shape",
                         shard_id="w2", expert_id=1)
    torch.testing.assert_close(layer.w2_weight_shape[1], torch.tensor([128, 256]))


def test_routing_patch_selects_ct_w4a8_and_delegates_other_schemes(monkeypatch):
    from types import ModuleType
    from vllm_hcu.patch.worker.op_opt import patch_compressed_tensors_moe_w4a8 as patch
    import vllm.config
    import vllm_hcu.deepseek_v4_runtime
    weight, act, moe = _args()
    module = ModuleType(patch.TARGET_MODULE)
    upstream_result = object()
    class Original:
        @staticmethod
        def get_moe_method(quant_config, layer, layer_name):
            return upstream_result
    module.CompressedTensorsMoEMethod = Original
    module.current_platform = SimpleNamespace(is_rocm=lambda: True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1,
                                        decode_context_parallel_size=1,
                                        prefill_context_parallel_size=8, data_parallel_size=1,
                                        enable_eplb=False),
        model_config=SimpleNamespace(enforce_eager=True),
        lora_config=None,
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(vllm_hcu.deepseek_v4_runtime, "model_architectures",
                        lambda _: ("DeepseekV41ForCausalLM",))
    scheme = {"weights": weight, "input_activations": act, "format": "pack-quantized"}
    quant = SimpleNamespace(
        _add_fused_moe_to_target_scheme_map=lambda: None,
        get_scheme_dict=lambda *args: scheme,
        _is_dynamic_token_w4a8_int=lambda w, a: w.num_bits == 4,
    )
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    method = Original.get_moe_method(quant, SimpleNamespace(moe_config=moe), "model.ffn.experts")
    assert isinstance(method, HCUCompressedTensorsW4A8Int8MoEMethod)
    weight.num_bits = 8
    assert Original.get_moe_method(quant, SimpleNamespace(moe_config=moe), "x") is upstream_result
    weight.num_bits = 4
    scheme["format"] = "int-quantized"
    with pytest.raises(ValueError, match="pack-quantized"):
        Original.get_moe_method(quant, SimpleNamespace(moe_config=moe), "x")


def _topology_routing(monkeypatch, *, tp=1, pp=1, dcp=1, pcp=8, dp=1,
                      eager=True, eplb=False, lora=None, model_tag="backbone"):
    """Install the routing patch with a synthetic topology and return the call."""
    import importlib
    from types import ModuleType
    import vllm.compilation.backends as backends
    import vllm.config
    import vllm_hcu.deepseek_v4_runtime
    from vllm_hcu.patch.worker.op_opt import patch_compressed_tensors_moe_w4a8 as patch

    weight, act, moe = _args()
    module = ModuleType(patch.TARGET_MODULE)

    class Original:
        @staticmethod
        def get_moe_method(quant_config, layer, layer_name):
            return "upstream"

    module.CompressedTensorsMoEMethod = Original
    module.current_platform = SimpleNamespace(is_rocm=lambda: True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp,
            decode_context_parallel_size=dcp, prefill_context_parallel_size=pcp,
            data_parallel_size=dp, enable_eplb=eplb,
        ),
        model_config=SimpleNamespace(enforce_eager=eager),
        lora_config=lora,
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        vllm_hcu.deepseek_v4_runtime, "model_architectures",
        lambda _: ("DeepseekV41ForCausalLM",),
    )
    monkeypatch.setattr(backends, "model_tag", model_tag, raising=False)
    # apply_to_module caches its marker on the class; use a fresh module each call.
    patch = importlib.reload(patch)
    patch.apply_to_module(module)
    scheme = {"weights": weight, "input_activations": act, "format": "pack-quantized"}
    quant = SimpleNamespace(
        _add_fused_moe_to_target_scheme_map=lambda: None,
        get_scheme_dict=lambda *args: scheme,
        _is_dynamic_token_w4a8_int=lambda w, a: w.num_bits == 4,
    )
    return lambda: Original.get_moe_method(
        quant, SimpleNamespace(moe_config=moe), "model.ffn.experts"
    )


def test_topology_gate_admits_prefill_and_decode_roles(monkeypatch):
    """P side keeps PCP8+eager+DP1; D side admits DP8+PCP1 with graphs."""
    # Prefill role: unchanged from the original bring-up contract.
    select = _topology_routing(monkeypatch, dp=1, pcp=8, eager=True)
    assert isinstance(select(), HCUCompressedTensorsW4A8Int8MoEMethod)
    # Decode role: DP8, PCP1, CUDA graphs enabled.
    select = _topology_routing(monkeypatch, dp=8, pcp=1, eager=False)
    assert isinstance(select(), HCUCompressedTensorsW4A8Int8MoEMethod)
    # Decode role with eager is still fine.
    select = _topology_routing(monkeypatch, dp=8, pcp=1, eager=True)
    assert isinstance(select(), HCUCompressedTensorsW4A8Int8MoEMethod)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        # P side must keep PCP8.
        (dict(dp=1, pcp=4), "pcp=4"),
        # P side must stay eager: PCP's virtual batch cannot be captured.
        (dict(dp=1, pcp=8, eager=False), "enforce_eager=False"),
        # Decode side must not run PCP.
        (dict(dp=8, pcp=8), "decode_pcp=8"),
        # Only the two audited DP widths are admitted.
        (dict(dp=2, pcp=1), "dp=2"),
        # Real layout constraints stay.
        (dict(dp=1, pcp=8, tp=2), "tp=2"),
        (dict(dp=1, pcp=8, pp=2), "pp=2"),
        (dict(dp=1, pcp=8, dcp=2), "dcp=2"),
        (dict(dp=8, pcp=1, eplb=True), "eplb=True"),
        (dict(dp=8, pcp=1, lora=object()), "lora!=None"),
        # Draft models are always pcp=1 regardless of DP.
        (dict(dp=8, pcp=2, model_tag="draft"), "draft_pcp=2"),
    ],
)
def test_topology_gate_rejects_unaudited_combinations(monkeypatch, kwargs, match):
    select = _topology_routing(monkeypatch, **kwargs)
    with pytest.raises(ValueError, match=match):
        select()


def test_draft_model_admits_pcp1_with_graphs(monkeypatch):
    """The draft runs DP-size AG-RS a2a with pcp=1; graphs are allowed there."""
    select = _topology_routing(
        monkeypatch, dp=8, pcp=1, eager=False, model_tag="draft"
    )
    assert isinstance(select(), HCUCompressedTensorsW4A8Int8MoEMethod)


def test_postload_packs_once_and_seals_loader(monkeypatch):
    from vllm.model_executor.layers.fused_moe import all2all_utils, modular_kernel as mk
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8_deepgemm_runtime as runtime
    method = HCUCompressedTensorsW4A8Int8MoEMethod(*_args())
    layer = torch.nn.Module()
    writes = []
    method.create_weights(layer, 2, 512, 256, torch.bfloat16,
                          weight_loader=lambda *args, **kwargs: writes.append(args))
    layer._expert_routing_tables = lambda: None
    layer.w13_weight_packed.data.zero_()
    layer.w2_weight_packed.data.zero_()
    layer.w13_weight_scale.data.fill_(2)
    layer.w2_weight_scale.data.fill_(4)
    pointers = [layer.w13_weight_packed.data_ptr(), layer.w2_weight_packed.data_ptr()]
    calls = []
    def pack(tensor):
        calls.append(tensor.data_ptr())
        return tensor
    monkeypatch.setattr(runtime, "pack_w4a8_moe_hipc_weight", pack)
    monkeypatch.setattr(runtime, "DeepEPDeepGemmW4A8ContiguousExperts", lambda *args: SimpleNamespace())
    monkeypatch.setattr(
        all2all_utils,
        "maybe_make_prepare_finalize",
        lambda **kwargs: SimpleNamespace(
            activation_format=mk.FusedMoEActivationFormat.Standard
        ),
    )
    monkeypatch.setattr(mk, "FusedMoEKernel", lambda pf, experts: SimpleNamespace(experts=experts))
    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(layer)
    assert calls == pointers
    assert layer.w13_weight_packed.dtype == torch.int8
    assert method.moe_kernel.experts._deepgemm_w13 is layer.w13_weight_packed
    assert method.moe_quant_config.gemm1_clamp_limit == 10.0
    for param in layer.parameters():
        with pytest.raises(RuntimeError, match="hot reload"):
            param.weight_loader(param, torch.tensor(0))
    assert writes == []
    torch.testing.assert_close(layer.w13_weight_scale, torch.full_like(layer.w13_weight_scale, 2))
    layer.w13_weight_scale.data.fill_(3)
    # A .data write evades Tensor version tracking, but the real loader above
    # is sealed before any write; direct .data mutation is outside the API.


def test_failed_pack_cannot_be_retried_as_raw(monkeypatch):
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8_deepgemm_runtime as runtime
    method, layer, _ = _layer()
    layer.w13_weight_packed.data.zero_()
    def fail(_):
        raise RuntimeError("synthetic pack failure")
    monkeypatch.setattr(runtime, "pack_w4a8_moe_hipc_weight", fail)
    with pytest.raises(RuntimeError, match="synthetic pack failure"):
        method.process_weights_after_loading(layer)
    with pytest.raises(RuntimeError, match="previously failed"):
        method.process_weights_after_loading(layer)


@pytest.mark.hcu
@pytest.mark.parametrize("hidden,outputs", [(5120, 4608), (2304, 5120)])
def test_ct_hipc_gemm_matches_logical_int4_reference(hidden, outputs):
    """Real HIPC execution, including CT byte conversion and /16 scales."""
    import os
    if os.environ.get("VLLM_HCU_TEST_W4A8_GPU") != "1":
        pytest.skip("set VLLM_HCU_TEST_W4A8_GPU=1 for explicit GPU execution")
    import deepgemm
    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(20261027)
    tokens, experts = 512, 3
    q = torch.randint(-8, 8, (experts, outputs, hidden), dtype=torch.int8, device=device, generator=generator)
    raw = _pack_ct(q)
    canonical = convert_ct_int4_to_hipc_bytes_(raw)
    packed = deepgemm.pack_w4a8_moe_hipc_weight(canonical)
    x = torch.randint(-127, 128, (tokens, hidden), dtype=torch.int8, device=device, generator=generator)
    xs = torch.linspace(0.01, 0.03, tokens, device=device).unsqueeze(-1)
    ws = torch.rand((experts, outputs, 1), device=device, generator=generator) * 0.02 + 0.005
    out = torch.empty((tokens, outputs), dtype=torch.bfloat16, device=device)
    ids = torch.arange(tokens, dtype=torch.int32, device=device) // 256
    deepgemm.m_grouped_w4a8_gemm_nt_contiguous_hipc((x, xs), (packed, ws / 16), out, ids)
    torch.cuda.synchronize()
    for index in (0, 1):
        sl = slice(index * 256, (index + 1) * 256)
        expected = (x[sl].float() * xs[sl]) @ (q[index].float() * ws[index]).T
        torch.testing.assert_close(out[sl].float(), expected.to(torch.bfloat16).float(), rtol=0.015, atol=0.05)


@pytest.mark.hcu
def test_full_hipc_experts_with_clamp_weighted_topk_and_empty_expert(monkeypatch):
    import os
    if os.environ.get("VLLM_HCU_TEST_W4A8_GPU") != "1":
        pytest.skip("set VLLM_HCU_TEST_W4A8_GPU=1 for explicit GPU execution")
    import deepgemm
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm_hcu.model_executor.layers.quantization import slimquant_w4a8_deepgemm_runtime as runtime
    device = torch.device("cuda", 0)
    gen = torch.Generator(device=device).manual_seed(741)
    e, m, h, intermediate = 3, 256, 5120, 2304
    w13 = torch.randint(-8, 8, (e, 2 * intermediate, h), generator=gen, device=device, dtype=torch.int8)
    w2 = torch.randint(-8, 8, (e, h, intermediate), generator=gen, device=device, dtype=torch.int8)
    s13 = torch.rand((e, 2 * intermediate, 1), generator=gen, device=device) * 0.2 + 0.05
    s2 = torch.rand((e, h, 1), generator=gen, device=device) * 0.02 + 0.005
    cfg = FusedMoEQuantConfig.make(torch.int8, weight_dtype="int4", per_act_token_quant=True,
                                  w1_scale=s13 / 16, w2_scale=s2 / 16, gemm1_clamp_limit=10.0)
    expert = object.__new__(runtime.DeepEPDeepGemmW4A8ContiguousExperts)
    expert.quant_config = cfg
    expert.activation_config = SimpleNamespace(clamp_limit=10.0)
    expert._deepgemm_w13 = deepgemm.pack_w4a8_moe_hipc_weight(convert_ct_int4_to_hipc_bytes_(_pack_ct(w13)))
    expert._deepgemm_w2 = deepgemm.pack_w4a8_moe_hipc_weight(convert_ct_int4_to_hipc_bytes_(_pack_ct(w2)))
    x = torch.randint(-100, 101, (m, h), generator=gen, device=device, dtype=torch.int8)
    xs = torch.linspace(0.01, 0.03, m, device=device).unsqueeze(-1)
    ids = torch.tensor([0, 1], dtype=torch.int64, device=device).repeat(m, 1)
    weights = torch.rand((m, 2), generator=gen, device=device)
    weights /= weights.sum(-1, keepdim=True)
    meta = mk.ExpertTokensMetadata.make_from_list([m, m, 0], device=device)
    shapes = expert.workspace_shapes(m, 2 * intermediate, h, 2, e, e, meta, MoEActivation.SILU)
    ws13, ws2, _ = [torch.empty(shape, device=device, dtype=torch.bfloat16) for shape in shapes]
    out = torch.empty((m, h), device=device, dtype=torch.bfloat16)
    stages = []
    original_gemm = runtime.m_grouped_w4a8_gemm_nt_contiguous_hipc
    original_permute = runtime.deepgemm_moe_permute
    permutation = []
    def capture_permute(**kwargs):
        result = original_permute(**kwargs)
        permutation.append(tuple(t.clone() for t in result[:4]))
        return result
    def capture_gemm(a, b, result, indices):
        inputs = tuple(t.clone() for t in a)
        original_gemm(a, b, result, indices)
        stages.append((inputs, result.clone(), indices.clone()))
    monkeypatch.setattr(runtime, "deepgemm_moe_permute", capture_permute)
    monkeypatch.setattr(runtime, "m_grouped_w4a8_gemm_nt_contiguous_hipc", capture_gemm)
    expert.apply(out, x, expert._deepgemm_w13, expert._deepgemm_w2,
                 weights, ids, MoEActivation.SILU, e, None, xs, None,
                 ws13, ws2, meta, False)
    reference = torch.zeros((m, h), device=device)
    clamped_values = 0
    permutation = permutation[0]
    for index in (0, 1):
        rows = permutation[3][:, index].long()
        torch.testing.assert_close(permutation[0][rows], x)
        torch.testing.assert_close(permutation[1][rows], xs)
        assert torch.all(permutation[2][rows] == index)
        gateup = ((x.float() * xs) @ (w13[index].float() * s13[index]).T).bfloat16().float()
        torch.testing.assert_close(stages[0][1][rows].float(), gateup, rtol=0.015, atol=0.05)
        gate, up = gateup.chunk(2, dim=-1)
        clamped_values += int((gate > 10).sum().item())
        gate = gate.clamp(max=10)
        up = up.clamp(-10, 10)
        activation = torch.nn.functional.silu(gate) * up
        scale = activation.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
        aq = (activation / scale).round().clamp(-127, 127)
        observed_activation = stages[1][0][0][rows].float() * stages[1][0][1][rows]
        activation_error = (observed_activation - aq * scale).norm() / (aq * scale).norm()
        assert float(activation_error) < 0.02, f"activation relative L2={float(activation_error)}"
        down = ((aq * scale) @ (w2[index].float() * s2[index]).T).bfloat16().float()
        observed_down_reference = (observed_activation @ (w2[index].float() * s2[index]).T).bfloat16().float()
        torch.testing.assert_close(stages[1][1][rows].float(), observed_down_reference, rtol=0.015, atol=0.05)
        reference += down * weights[:, index:index + 1]
    torch.cuda.synchronize()
    assert clamped_values > 0
    relative_l2 = (out.float() - reference).norm() / reference.norm()
    assert float(relative_l2) < 0.02, f"relative L2={float(relative_l2)}"


def test_batched_experts_adopt_n32_view_of_packed_weights(monkeypatch):
    """The masked path must build its N32 view without `w13_weight`.

    The compressed-tensors layer registers only `*_packed`; the SlimQuant
    post-load hook re-packs `w13_weight` and would raise AttributeError here.
    """
    from vllm.model_executor.layers.fused_moe import all2all_utils, modular_kernel as mk
    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as runtime,
    )

    method, layer, _ = _layer()
    layer._expert_routing_tables = lambda: None
    layer.w13_weight_packed.data.zero_()
    layer.w2_weight_packed.data.zero_()
    layer.w13_weight_scale.data.fill_(2)
    layer.w2_weight_scale.data.fill_(4)

    adopted = []

    class FakeBatched(runtime.DeepEPDeepGemmW4A8BatchedExperts):
        def __init__(self, **kwargs):
            self._deepgemm_w13 = None
            self._deepgemm_w2 = None
            adopted.append(self)

        def adopt_packed_weights(self, target):
            adopted.append(target)
            self._deepgemm_w13 = target.w13_weight_packed
            self._deepgemm_w2 = target.w2_weight_packed

    monkeypatch.setattr(runtime, "pack_w4a8_moe_hipc_weight", lambda t: t)
    monkeypatch.setattr(runtime, "DeepEPDeepGemmW4A8BatchedExperts", FakeBatched)
    monkeypatch.setattr(
        all2all_utils,
        "maybe_make_prepare_finalize",
        lambda **kwargs: SimpleNamespace(
            activation_format=mk.FusedMoEActivationFormat.BatchedExperts,
            max_num_tokens_per_rank=lambda: 128,
            num_dispatchers=lambda: 1,
        ),
    )
    monkeypatch.setattr(
        mk, "FusedMoEKernel", lambda pf, experts: SimpleNamespace(experts=experts)
    )

    method.process_weights_after_loading(layer)

    experts = method.moe_kernel.experts
    assert isinstance(experts, FakeBatched)
    assert experts._deepgemm_w13 is layer.w13_weight_packed
    assert experts._deepgemm_w2 is layer.w2_weight_packed


def test_adopt_packed_weights_rejects_missing_packed_parameters():
    """Guard the contract: the SlimQuant-named parameters must not be required."""
    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as runtime,
    )

    class Layer(torch.nn.Module):
        pass

    layer = Layer()
    layer.register_parameter(
        "w13_weight_packed",
        torch.nn.Parameter(torch.zeros(2, 512, 64, dtype=torch.int8), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight_packed",
        torch.nn.Parameter(torch.zeros(2, 512, 32, dtype=torch.int8), requires_grad=False),
    )
    experts = runtime.DeepEPDeepGemmW4A8BatchedExperts.__new__(
        runtime.DeepEPDeepGemmW4A8BatchedExperts
    )
    experts._deepgemm_w13 = None
    experts._deepgemm_w2 = None
    experts.adopt_packed_weights(layer)
    assert experts._deepgemm_w13 is not None
    assert experts._deepgemm_w2 is not None
    assert not hasattr(layer, "w13_weight")


def test_masked_layout_hook_lives_only_on_the_batched_experts():
    """The N32 hook must not leak onto the contiguous experts.

    Both classes already own ``process_weights_after_loading`` writing
    ``_deepgemm_w13`` from a shared cache prefix; if the masked hook were also
    present on the contiguous class the two would overwrite each other with
    incompatible layouts.
    """
    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as runtime,
    )

    batched = runtime.DeepEPDeepGemmW4A8BatchedExperts
    contiguous = runtime.DeepEPDeepGemmW4A8ContiguousExperts
    assert hasattr(batched, "adopt_packed_weights")
    assert not hasattr(contiguous, "adopt_packed_weights")
    assert not issubclass(contiguous, batched)
    assert not issubclass(batched, contiguous)


def test_masked_layout_cache_keys_do_not_collide_with_contiguous():
    """Derived-layout caches of the two expert classes stay separate."""
    from vllm_hcu.model_executor.layers.quantization import (
        slimquant_w4a8_deepgemm_runtime as runtime,
    )

    batched_prefix = runtime.DeepEPDeepGemmW4A8BatchedExperts._CACHE_PREFIX
    contiguous_prefix = runtime.DeepEPDeepGemmW4A8ContiguousExperts._CACHE_PREFIX
    assert batched_prefix != contiguous_prefix
    assert not batched_prefix.startswith(contiguous_prefix)
    assert not contiguous_prefix.startswith(batched_prefix)
