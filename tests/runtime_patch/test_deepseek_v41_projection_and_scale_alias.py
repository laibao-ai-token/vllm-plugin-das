# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU contracts for narrow V41 layout/loader bridges, not GPU acceptance."""
from __future__ import annotations

from types import ModuleType, SimpleNamespace
import pytest
import torch

from vllm_hcu.patch.worker.core_fix import (
    patch_deepseek_v41_attention as projection,
    patch_deepseek_v41_dspark_load_weights as draft_alias,
)


def _module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


@pytest.mark.parametrize("stored_nn", (False, True))
def test_v41_compressor_selects_physical_orientation(stored_nn, monkeypatch):
    hidden = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16)
    logical_nt = torch.tensor([[1, 0, 1], [0, 1, -1]], dtype=torch.bfloat16)
    stored = logical_nt.T.contiguous() if stored_nn else logical_nt
    calls = []
    real_mm = torch.mm

    def cpu_reference_mm(a, b, *, out_dtype):
        # out_dtype float32 for BF16 torch.mm is a GPU API contract. This
        # CPU reference checks orientation/numbers and the forwarded dtype;
        # it does not claim to execute that GPU GEMM or validate its precision.
        calls.append((a.shape, b.shape, out_dtype))
        return real_mm(a.float(), b.float())

    monkeypatch.setattr(projection.torch, "mm", cpu_reference_mm)
    actual = projection._compressor_mm(hidden, stored)
    assert calls == [(torch.Size([2, 3]), torch.Size([3, 2]), torch.float32)]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.tensor([[4., -1.], [10., -1.]]))


@pytest.mark.parametrize("has_compressor", (False, True))
@pytest.mark.parametrize("has_indexer", (False, True))
def test_v41_projection_keeps_three_result_boundary(
    has_compressor, has_indexer, monkeypatch
):
    class DeepseekV4Attention:
        def _run_parallel_input_projections(self, hidden_states):
            raise AssertionError("unadapted projection used")

    observed = []
    main, score, weights = object(), object(), object()
    hidden = torch.ones((2, 3), dtype=torch.bfloat16)
    weight = object()

    def execute_in_parallel(main_fn, aux_fns, event, aux_events, streams, enable):
        observed.append((event, tuple(aux_events), tuple(streams), enable))
        return main_fn(), tuple(None if fn is None else fn() for fn in aux_fns)

    module = _module(
        projection.TARGET_MODULE, DeepseekV4Attention=DeepseekV4Attention,
        execute_in_parallel=execute_in_parallel,
        envs=SimpleNamespace(VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=4),
    )
    mm_calls = []

    def compressor_mm(h, w):
        mm_calls.append((h is hidden, w is weight))
        return score

    monkeypatch.setattr(projection, "_compressor_mm", compressor_mm)
    assert projection.apply_to_module(module) is True
    assert projection.apply_to_module(module) is False
    instance = DeepseekV4Attention()
    instance.aux_stream_list = ["a", "b", "unused"]
    instance.ln_events = ["main", "kv", "index", "unused"]
    instance.compressor = (
        SimpleNamespace(fused_wkv_wgate=SimpleNamespace(weight=weight))
        if has_compressor else None
    )
    instance.indexer = (
        SimpleNamespace(weights_proj=lambda h: (weights, None))
        if has_indexer else None
    )
    instance._fused_wqa_wkv_gemm = lambda h: main
    result = instance._run_parallel_input_projections(hidden)
    assert result == (
        main, score if has_compressor else None, weights if has_indexer else None
    )
    assert mm_calls == ([(True, True)] if has_compressor else [])
    assert observed == [("main", ("kv", "index"), ("a", "b"), True)]


@pytest.mark.parametrize(
    ("stored_name", "checkpoint_name"),
    (("attn.weight_scale", "attn.weight_scale_inv"),
     ("attn.weight_scale_inv", "attn.weight_scale"),
     ("experts.w13_weight_scale", "experts.w13_weight_scale_inv"),
     ("experts.w13_weight_scale_inv", "experts.w13_weight_scale")),
)
@pytest.mark.parametrize("instance_override", (False, True))
@pytest.mark.parametrize("loader_error", (False, True))
def test_v41_draft_aliases_restore_method_after_load(
    stored_name, checkpoint_name, instance_override, loader_error
):
    parameter = object()
    payload = [(checkpoint_name, object())]

    class DSparkDeepseekV4ForCausalLM:
        def named_parameters(self):
            return iter(((stored_name, parameter),))

        def load_weights(self, weights):
            assert weights is payload
            params = dict(self.named_parameters())
            assert params[checkpoint_name] is parameter
            if loader_error:
                raise RuntimeError("injected draft load failure")
            return parameter

    module = _module(
        draft_alias.TARGET_MODULE,
        DSparkDeepseekV4ForCausalLM=DSparkDeepseekV4ForCausalLM,
    )
    assert draft_alias.apply_to_module(module) is True
    assert draft_alias.apply_to_module(module) is False
    instance = DSparkDeepseekV4ForCausalLM()
    if instance_override:
        instance.named_parameters = lambda: iter(((stored_name, parameter),))
    before = instance.named_parameters
    if loader_error:
        with pytest.raises(RuntimeError, match="injected draft load failure"):
            instance.load_weights(payload)
    else:
        assert instance.load_weights(payload) is parameter
    assert instance.named_parameters == before
    assert ("named_parameters" in vars(instance)) is instance_override
    assert list(instance.named_parameters()) == [(stored_name, parameter)]


@pytest.mark.parametrize("name", ("attn.weight", "attn.input_scale", "attn.scale_inv"))
def test_v41_draft_does_not_alias_other_parameters(name):
    assert draft_alias._scale_alias(name) is None


def test_v41_draft_preserves_both_native_scale_parameters():
    scale, inverse = object(), object()
    native = (("attn.weight_scale", scale), ("attn.weight_scale_inv", inverse))

    class DSparkDeepseekV4ForCausalLM:
        def named_parameters(self):
            return iter(native)

        def load_weights(self, weights):
            return dict(self.named_parameters())

    module = _module(
        draft_alias.TARGET_MODULE,
        DSparkDeepseekV4ForCausalLM=DSparkDeepseekV4ForCausalLM,
    )
    assert draft_alias.apply_to_module(module) is True
    instance = DSparkDeepseekV4ForCausalLM()
    params = instance.load_weights(())
    assert params["attn.weight_scale"] is scale
    assert params["attn.weight_scale_inv"] is inverse
    assert "named_parameters" not in vars(instance)
