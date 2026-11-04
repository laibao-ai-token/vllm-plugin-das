# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU regressions for Step5 optimization and shared-loader scope."""
import hashlib
import inspect
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_hcu.patch.worker.framework_opt import patch_step5_mtp_heads as adapter
from vllm_hcu.runtime_compat.step5_patch_contract import Step5PatchCompatibilityError


@pytest.mark.parametrize(
    "opt_in,has_bias,eligible,head_dtype,expected",
    [
        (False, False, True, None, "native"),
        (True, False, True, None, "skinny"),
        (True, True, True, None, "native"),
        (True, False, False, None, "native"),
        (True, False, True, torch.float32, "native"),
    ],
)
def test_skinny_head_requires_step5_opt_in(
    monkeypatch, opt_in, has_bias, eligible, head_dtype, expected
):
    from vllm_hcu.patch.worker.framework_opt import patch_step5_skinny_lm_head as a
    from vllm_hcu.model_executor.layers import skinny_lm_head

    calls = []

    class Processor:
        def __init__(self):
            self.head_dtype = head_dtype
            if opt_in:
                self._step5_skinny_lm_head = True

        def _apply_head(self, lm_head, hidden_states, embedding_bias=None):
            calls.append("native")
            return "native"

    class Unquantized:
        pass

    target = ModuleType(a.TARGET_MODULE)
    target.LogitsProcessor = Processor
    target.UnquantizedEmbeddingMethod = Unquantized
    target.UnquantizedLinearMethod = Unquantized
    target.logger = SimpleNamespace(info_once=lambda *args: None)
    monkeypatch.setattr(skinny_lm_head, "usable", lambda *args: eligible)
    monkeypatch.setattr(
        skinny_lm_head, "skinny_linear",
        lambda *args: calls.append("skinny") or "skinny",
    )
    a.apply_to_module(target)
    head = SimpleNamespace(quant_method=Unquantized(), weight=torch.zeros(4, 4))
    hidden = torch.ones(1, 4, dtype=torch.bfloat16)
    bias = torch.zeros(4) if has_bias else None
    assert Processor()._apply_head(head, hidden, bias) == expected
    assert calls == [expected]


def test_skinny_head_master_off_uses_native_without_probing_provider(monkeypatch):
    """Master-off must branch before the optional skinny kernel is probed."""
    from vllm_hcu.patch.worker.framework_opt import patch_step5_skinny_lm_head as a
    from vllm_hcu.model_executor.layers import skinny_lm_head

    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    monkeypatch.setenv("STEP5_SKINNY_LM_HEAD", "1")

    class Processor:
        head_dtype = None
        _step5_skinny_lm_head = True

        def _apply_head(self, lm_head, hidden_states, embedding_bias=None):
            return "native"

    class Unquantized:
        pass

    target = ModuleType(a.TARGET_MODULE)
    target.LogitsProcessor = Processor
    target.UnquantizedEmbeddingMethod = Unquantized
    target.UnquantizedLinearMethod = Unquantized
    target.logger = SimpleNamespace(info_once=lambda *args: None)
    monkeypatch.setattr(
        skinny_lm_head,
        "usable",
        lambda *args: pytest.fail("master-off probed the skinny provider"),
    )
    a.apply_to_module(target)

    head = SimpleNamespace(quant_method=Unquantized(), weight=torch.zeros(4, 4))
    hidden = torch.ones(1, 4, dtype=torch.bfloat16)
    assert Processor()._apply_head(head, hidden) == "native"


def test_eplb_native_metric_preserves_cross_layer_cancellation():
    from vllm_hcu.distributed.eplb_runtime import _rank_load_imbalance

    mapping = torch.tensor([[0, 1], [0, 1]])
    # Equal totals across ranks, but opposite bottlenecks in the two layers.
    loads = torch.tensor([[9.0, 1.0], [1.0, 9.0]])
    assert _rank_load_imbalance(mapping, loads, 2, layerwise=False) == 1.0
    assert _rank_load_imbalance(mapping, loads, 2, layerwise=True) == pytest.approx(1.8)


@pytest.mark.parametrize("layerwise", [False, True])
def test_eplb_metric_handles_replicas_and_zero_load(layerwise):
    from vllm_hcu.distributed.eplb_runtime import _rank_load_imbalance

    mapping = torch.tensor([[0, 1, 0, 1]])
    loads = torch.tensor([[8.0, 4.0]])
    assert _rank_load_imbalance(mapping, loads, 2, layerwise=layerwise) == 1.0
    assert _rank_load_imbalance(mapping, loads * 0, 2, layerwise=layerwise) == 1.0


def _mtp_config(model_type, architectures=None, nested=False):
    hf = SimpleNamespace(model_type=model_type, architectures=architectures)
    if nested:
        hf = SimpleNamespace(text_config=hf, architectures=architectures)
    return SimpleNamespace(model_config=SimpleNamespace(hf_config=hf))


@pytest.fixture
def mtp_loaders(monkeypatch):
    def loader(module_name):
        module = ModuleType(module_name)
        calls, result = [], object()

        def load_eagle_model(target_model, vllm_config):
            calls.append((target_model, vllm_config))
            return result

        module.load_eagle_model = load_eagle_model
        return SimpleNamespace(module=module, calls=calls, result=result)

    native = loader(adapter.TARGET_MODULE)
    owned = loader("vllm_hcu.v1.step5_mtp_loading")
    monkeypatch.setitem(sys.modules, owned.module.__name__, owned.module)
    return native, owned


@pytest.mark.parametrize("model_type,reviewed", [
    ("deepseek_v41", False), ("deepseek_v4", False), ("llama", False),
    ("step5", True), ("step5_mtp", True),
])
def test_mtp_loader_dispatch_preserves_identity(
    monkeypatch, mtp_loaders, model_type, reviewed
):
    native, owned = mtp_loaders
    original = native.module.load_eagle_model
    if reviewed:
        # Exercise the real inspect/dedent/SHA256 audit with reviewed fixture code.
        source = textwrap.dedent(inspect.getsource(original))
        digest = hashlib.sha256(source.encode()).hexdigest()
        monkeypatch.setattr(
            adapter, "REVIEWED_SOURCES", ((adapter.TARGETS[0], (digest,)),)
        )
    assert adapter.apply_to_module(native.module) is True
    model, config = object(), _mtp_config(model_type)
    expected, other = (owned, native) if reviewed else (native, owned)
    assert native.module.load_eagle_model(model, config) is expected.result
    assert expected.calls == [(model, config)] and not other.calls
    assert inspect.unwrap(native.module.load_eagle_model) is original
    installed = native.module.load_eagle_model
    assert adapter.apply_to_module(native.module) is False
    assert native.module.load_eagle_model is installed


@pytest.mark.parametrize("model_type,architectures,nested", [
    ("step5", None, False), ("step5_mtp", None, False),
    ("step5", None, True), ("other", ["Step5ForConditionalGeneration"], False),
])
def test_mtp_unknown_step5_source_is_rejected_before_owned_loader(
    mtp_loaders, model_type, architectures, nested
):
    native, owned = mtp_loaders
    assert adapter.apply_to_module(native.module) is True
    with pytest.raises(Step5PatchCompatibilityError, match="Unreviewed Step5 target source"):
        native.module.load_eagle_model(
            object(), _mtp_config(model_type, architectures, nested)
        )
    assert not native.calls and not owned.calls


@pytest.mark.parametrize("failure,match", [("signature", "signature"), ("stale", "Stale")])
def test_mtp_loader_rejects_incompatible_installation(mtp_loaders, failure, match):
    native, _ = mtp_loaders
    if failure == "stale":
        assert adapter.apply_to_module(native.module) is True
        native.module.load_eagle_model = lambda target_model, vllm_config: None
    else:
        native.module.load_eagle_model = lambda incompatible: None
    with pytest.raises(Step5PatchCompatibilityError, match=match):
        adapter.apply_to_module(native.module)
    if failure == "signature":
        assert not hasattr(native.module, adapter._MARKER)


def test_mtp_non_step5_original_error_propagates(mtp_loaders):
    native, owned = mtp_loaders
    error = ValueError("original loader failure")

    def load_eagle_model(target_model, vllm_config):
        raise error

    native.module.load_eagle_model = load_eagle_model
    assert adapter.apply_to_module(native.module) is True
    with pytest.raises(ValueError) as caught:
        native.module.load_eagle_model(object(), _mtp_config("deepseek_v41"))
    assert caught.value is error
    assert not owned.calls
