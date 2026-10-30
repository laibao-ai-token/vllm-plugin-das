# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Regression coverage for DSpark scale aliases and loader tracking identity."""
from __future__ import annotations

from types import ModuleType

import pytest

from vllm_hcu.patch.worker.core_fix import (
    patch_deepseek_v41_dspark_load_weights as draft_alias,
)


def _module(name: str, **attributes: object) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _install_loader(native, checkpoint, return_factory):
    parameters = dict(native)
    parameter = next(iter(parameters.values()))

    class DSparkDeepseekV4ForCausalLM:
        def named_parameters(self):
            return iter(tuple(self._native))

        def load_weights(self, weights):
            del weights
            visible = dict(self.named_parameters())
            assert visible[checkpoint] is parameter
            return return_factory(visible)

    module = _module(
        draft_alias.TARGET_MODULE,
        DSparkDeepseekV4ForCausalLM=DSparkDeepseekV4ForCausalLM,
    )
    assert draft_alias.apply_to_module(module) is True
    assert draft_alias.apply_to_module(module) is False
    instance = DSparkDeepseekV4ForCausalLM()
    instance._native = tuple(native)
    return instance


@pytest.mark.parametrize(
    ("native_name", "checkpoint_name"),
    (
        ("attn.weight_scale", "attn.weight_scale_inv"),
        ("attn.weight_scale_inv", "attn.weight_scale"),
        ("experts.w13_weight_scale", "experts.w13_weight_scale_inv"),
        ("experts.w13_weight_scale_inv", "experts.w13_weight_scale"),
    ),
)
def test_set_result_normalizes_only_synthesized_alias(native_name, checkpoint_name):
    parameter = object()

    def result(visible):
        assert visible[checkpoint_name] is parameter
        return {checkpoint_name, "unknown.weight", "attn.scale_inv"}

    instance = _install_loader(
        ((native_name, parameter),),
        checkpoint_name,
        result,
    )
    loaded = instance.load_weights(())
    assert loaded == {native_name, "unknown.weight", "attn.scale_inv"}
    assert list(instance.named_parameters()) == [(native_name, parameter)]
    assert "named_parameters" not in vars(instance)


def test_dual_native_scale_names_are_not_remapped():
    scale, inverse = object(), object()
    native = (
        ("attn.weight_scale", scale),
        ("attn.weight_scale_inv", inverse),
    )

    class DSparkDeepseekV4ForCausalLM:
        def named_parameters(self):
            return iter(native)

        def load_weights(self, weights):
            del weights
            visible = dict(self.named_parameters())
            assert visible == dict(native)
            return {"attn.weight_scale_inv"}

    module = _module(
        draft_alias.TARGET_MODULE,
        DSparkDeepseekV4ForCausalLM=DSparkDeepseekV4ForCausalLM,
    )
    draft_alias.apply_to_module(module)
    instance = DSparkDeepseekV4ForCausalLM()
    assert instance.load_weights(()) == {"attn.weight_scale_inv"}
    assert "named_parameters" not in vars(instance)


def test_alias_identity_is_scoped_when_native_parameters_change():
    scale, inverse = object(), object()

    class DSparkDeepseekV4ForCausalLM:
        def __init__(self):
            self._native = (("attn.weight_scale", scale),)
            self.phase = 0

        def named_parameters(self):
            return iter(self._native)

        def load_weights(self, weights):
            del weights
            visible = dict(self.named_parameters())
            if self.phase == 0:
                assert visible["attn.weight_scale_inv"] is scale
                return {"attn.weight_scale_inv"}
            assert visible == {
                "attn.weight_scale": scale,
                "attn.weight_scale_inv": inverse,
            }
            return {"attn.weight_scale_inv"}

    module = _module(
        draft_alias.TARGET_MODULE,
        DSparkDeepseekV4ForCausalLM=DSparkDeepseekV4ForCausalLM,
    )
    draft_alias.apply_to_module(module)
    instance = DSparkDeepseekV4ForCausalLM()
    assert instance.load_weights(()) == {"attn.weight_scale"}
    instance.phase = 1
    instance._native = (
        ("attn.weight_scale", scale),
        ("attn.weight_scale_inv", inverse),
    )
    assert instance.load_weights(()) == {"attn.weight_scale_inv"}


@pytest.mark.parametrize("return_kind", ("none", "dict", "object"))
def test_non_set_fake_results_are_passed_through(return_kind):
    parameter = object()
    sentinel = object()

    def result(visible):
        assert visible["attn.weight_scale_inv"] is parameter
        if return_kind == "none":
            return None
        if return_kind == "dict":
            return {"attn.weight_scale_inv": parameter}
        return sentinel

    instance = _install_loader(
        (("attn.weight_scale", parameter),),
        "attn.weight_scale_inv",
        result,
    )
    loaded = instance.load_weights(())
    if return_kind == "none":
        assert loaded is None
    elif return_kind == "dict":
        assert loaded == {"attn.weight_scale_inv": parameter}
    else:
        assert loaded is sentinel


@pytest.mark.parametrize("instance_override", (False, True))
def test_exception_restores_class_or_instance_method(instance_override):
    parameter = object()

    class DSparkDeepseekV4ForCausalLM:
        def named_parameters(self):
            return iter((("attn.weight_scale", parameter),))

        def load_weights(self, weights):
            del weights
            assert dict(self.named_parameters())["attn.weight_scale_inv"] is parameter
            raise RuntimeError("injected draft load failure")

    module = _module(
        draft_alias.TARGET_MODULE,
        DSparkDeepseekV4ForCausalLM=DSparkDeepseekV4ForCausalLM,
    )
    draft_alias.apply_to_module(module)
    instance = DSparkDeepseekV4ForCausalLM()
    if instance_override:
        previous = lambda: iter((("attn.weight_scale", parameter),))
        instance.named_parameters = previous
    else:
        previous = instance.named_parameters

    with pytest.raises(RuntimeError, match="injected draft load failure"):
        instance.load_weights(())
    assert instance.named_parameters == previous
    assert ("named_parameters" in vars(instance)) is instance_override
