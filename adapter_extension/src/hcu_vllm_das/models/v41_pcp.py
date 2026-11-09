# SPDX-License-Identifier: Apache-2.0
"""Bounded native constructor namespaces for the HCU V41 PCP model.

Core has no decoder factory: its constructors resolve three hard-coded global
names. Reuse the exact reviewed code with private globals containing only those
name substitutions, preserving defaults/signatures/closures. This avoids copying
parameter lists or temporarily changing process-global Core symbols. Forward,
checkpoint loading and runner model-state methods stay native.
"""
from __future__ import annotations

import hashlib
import inspect
from contextvars import ContextVar
from types import FunctionType

from vllm.models.deepseek_v4.amd import model as native_v4
from vllm.models.deepseek_v41.amd import model as native
from vllm.models.deepseek_v41.amd import vl_model as native_vl
from .. import static_eplb_runtime as static_eplb
from ..v41_pcp import HcuV41PcpAttention, validate_config
from ..v41_engram import HcuV41PcpEngram
from ..v41_target_engram import HcuV41HostEngram, use_host_target, validate_host_config

_DECODER_NATIVE = native.DeepseekV4DecoderLayer.__init__
_BACKBONE_NATIVE = native.DeepseekV4Model.__init__
_OUTER_NATIVE = native_vl.DeepseekV41ForCausalLM.__init__
_BASE_MOE_NATIVE = native_v4.DeepseekV4MoE.__init__
_REVIEWED = (
    '919c98099d1ba3e443ea5a5c5569e6bd407caf52c40945fb792e9b4e85a13a86',
    'b804d0bd4d325397b0df3a094f47456a715b2d431cb06d2c7a06c6bc8332681b',
    '41b21ccf6dc5d66c98082650c8e046b977f0e61dce927aab890e23e508bab434',
)
_REVIEWED_BASE_MOE = '3396ecd9e48af5d6c094d392ffb62ad4271ec94cea4c42e6a037c5f398d26610'


def _constructor(function, expected_hash, substitutions):
    actual = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    if actual != expected_hash:
        raise RuntimeError(f'HCU V41 PCP constructor source changed: {function.__qualname__}: {actual}')
    if not substitutions or any(name not in function.__globals__ for name in substitutions):
        raise RuntimeError('HCU V41 PCP constructor class-name anchor missing')
    namespace = dict(function.__globals__)
    namespace.update(substitutions)
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__
    clone.__qualname__ = function.__qualname__
    return clone


def check_native_contract():
    if (native.DeepseekV4DecoderLayer.__init__ is not _DECODER_NATIVE
            or native.DeepseekV4Model.__init__ is not _BACKBONE_NATIVE
            or native_vl.DeepseekV41ForCausalLM.__init__ is not _OUTER_NATIVE
            or native_v4.DeepseekV4MoE.__init__ is not _BASE_MOE_NATIVE):
        raise RuntimeError('HCU V41 PCP native constructors changed after preparation')


def select_attention(cfg):
    if not validate_config(cfg):
        return native._select_dsv4_attn_cls(cfg)
    return HcuV41PcpAttention


_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                             {'_select_dsv4_attn_cls': select_attention,
                              'Engram': HcuV41PcpEngram})


class HcuV41PcpDecoderLayer(native.DeepseekV4DecoderLayer):
    __init__ = _DECODER_INIT


_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                              {'DeepseekV4DecoderLayer': HcuV41PcpDecoderLayer})


class HcuV41PcpBackbone(native.DeepseekV4Model):
    def __init__(self, *, vllm_config, prefix=''):
        if validate_config(vllm_config):
            _BACKBONE_INIT(self, vllm_config=vllm_config, prefix=prefix)
        else:
            super().__init__(vllm_config=vllm_config, prefix=prefix)


class HcuV41PcpLLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuV41PcpBackbone


_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                           {'DeepseekV41LLMForCausalLM': HcuV41PcpLLM})


_HOST_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                                   {'Engram': HcuV41HostEngram})


class HcuV41HostDecoderLayer(native.DeepseekV4DecoderLayer):
    __init__ = _HOST_DECODER_INIT


_HOST_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                                    {'DeepseekV4DecoderLayer': HcuV41HostDecoderLayer})


class HcuV41HostBackbone(native.DeepseekV4Model):
    def __init__(self, *, vllm_config, prefix=''):
        validate_host_config(vllm_config)
        _HOST_BACKBONE_INIT(self, vllm_config=vllm_config, prefix=prefix)


class HcuV41HostLLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuV41HostBackbone


_HOST_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                                 {'DeepseekV41LLMForCausalLM': HcuV41HostLLM})


# Static EPLB ladder. Core sizes the physical expert tables and builds the
# per-layer router state from the MoE construction keywords, so the only
# substitution needed is a factory adding them for owned backbone layers.
_MOE_CONFIG = ContextVar('hcu_static_eplb_moe_config', default=None)


def _eplb_factory(*args, **kwargs):
    cfg = _MOE_CONFIG.get()
    if cfg is None:
        raise RuntimeError('HCU static EPLB MoE factory used outside owned construction')
    extra = static_eplb.moe_eplb_kwargs(cfg, kwargs.get('prefix', ''))
    return native_v4.FusedMoEFactory(*args, **{**kwargs, **extra})


_EPLB_BASE_MOE_INIT = _constructor(_BASE_MOE_NATIVE, _REVIEWED_BASE_MOE,
                                    {'FusedMoEFactory': _eplb_factory})


class _HcuEplbBaseMoE(native_v4.DeepseekV4MoE):
    __init__ = _EPLB_BASE_MOE_INIT


class HcuV41EplbMoE(native.DeepseekV4MoE, _HcuEplbBaseMoE):
    """V41 MoE whose inherited V4 construction reaches the owned factory.

    The MRO places the substituted base constructor directly after the native
    V41 one, so the V41 ``super().__init__`` call stays native code and only
    the ``FusedMoEFactory`` global it resolves differs.
    """

    def __init__(self, vllm_config, prefix='', use_sequence_parallel=False):
        token = _MOE_CONFIG.set(vllm_config)
        try:
            super().__init__(vllm_config, prefix=prefix,
                             use_sequence_parallel=use_sequence_parallel)
        finally:
            _MOE_CONFIG.reset(token)
        static_eplb.publish_physical_metadata(self, vllm_config, prefix)


_EPLB_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                                   {'_select_dsv4_attn_cls': select_attention,
                                    'Engram': HcuV41PcpEngram,
                                    'DeepseekV4MoE': HcuV41EplbMoE})


class HcuV41EplbDecoderLayer(native.DeepseekV4DecoderLayer):
    __init__ = _EPLB_DECODER_INIT


_EPLB_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                                    {'DeepseekV4DecoderLayer': HcuV41EplbDecoderLayer})


class HcuV41EplbBackbone(native.DeepseekV4Model):
    def __init__(self, *, vllm_config, prefix=''):
        _EPLB_BACKBONE_INIT(self, vllm_config=vllm_config, prefix=prefix)


class HcuV41EplbLLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuV41EplbBackbone


_EPLB_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                                 {'DeepseekV41LLMForCausalLM': HcuV41EplbLLM})


class HcuV41PcpForCausalLM(native_vl.DeepseekV41ForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        from ..v41_pcp_activation import check_health
        check_health()
        check_native_contract()
        if vllm_config.parallel_config.prefill_context_parallel_size > 1:
            from ..v41_pcp_activation import pcp_enabled
            if not pcp_enabled():
                raise ValueError('PCP attention requires VLLM_HCU_DSV41_PCP_ATTENTION=1')
        self._hcu_static_eplb_config = None
        if static_eplb.enabled(vllm_config):
            # Fails closed unless this is the owned PCP target on a reviewed
            # topology; static placement composes with no other ladder.
            static_eplb.validate_config(vllm_config)
            self._hcu_static_eplb_config = vllm_config
            _EPLB_OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
        elif validate_config(vllm_config):
            _OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
        elif use_host_target(vllm_config):
            _HOST_OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
        else:
            # PCP1 without the explicit host route remains fully native.
            super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        cfg = getattr(self, '_hcu_static_eplb_config', None)
        if cfg is None:
            return super().load_weights(weights)
        # Bind rows before native loaders read any expert tensor, and publish
        # routing only once every physical slot has been populated.
        static_eplb.bind_placement(self, cfg)
        loaded = super().load_weights(weights)
        static_eplb.install_routing(self, cfg)
        # Only a calibration run has a session; deployment has nothing to step.
        self._hcu_static_eplb_session = static_eplb.record_session(self)
        return loaded

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        # Calibration counts engine steps here: the load counters themselves are
        # accumulated by Core's routing kernel, so this adds no device work.
        session = getattr(self, '_hcu_static_eplb_session', None)
        if session is not None:
            session.step()
        return output
