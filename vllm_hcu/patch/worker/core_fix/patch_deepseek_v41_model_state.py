# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""V41 Engram lookback at PCP fragment cursors; preserve native replay state."""
from __future__ import annotations
import functools
import hashlib
import inspect
import textwrap
from types import ModuleType
import torch
from ._common import (PatchCompatibilityError, load_exact_module, require_callable,
                      require_class, require_exact_signature)

TARGET_MODULE = "vllm.models.deepseek_v41.nvidia.model_state"
PATCH_ID = "worker.core_fix.deepseek_v41.pcp_lookback"
TARGETS = (f"{TARGET_MODULE}.DeepseekV41ModelState.__init__",
           f"{TARGET_MODULE}.DeepseekV41ModelState.prepare_inputs")
_MARKER = "_vllm_hcu_dsv41_pcp_lookback_applied"
_WRAPPER = "_vllm_hcu_dsv41_pcp_lookback_wrapper"
_NATIVE_INIT = "7052ad7124102008dff6410c8dde7703bf709517f92bd8e0d01b7c6439f979bc"
_NATIVE_PREPARE = "503aa0a67c815c9efb498271f006e740a4c9534ff7918bebba0697f9103aa00b"
_DEFAULT_PREPARE = "770b84d07eb0b5bad8f9d1dd2387bede305b61ed35acfbcb800a3569dda65967"


def _require_source(function, name, expected):
    try:
        actual = hashlib.sha256(textwrap.dedent(inspect.getsource(function)).encode()).hexdigest()
    except (OSError, TypeError) as exc:
        raise PatchCompatibilityError(f"cannot audit required PCP lookback target {name}") from exc
    if actual != expected:
        raise PatchCompatibilityError(f"unreviewed PCP lookback source {name}: expected={expected}, actual={actual}")


def _fill_fragment_lookback(window, input_batch, req_states):
    if not isinstance(window, torch.Tensor) or window.ndim != 2 or window.dtype != torch.int32:
        raise RuntimeError("PCP Engram lookback window must be rank-2 int32")
    fields = (input_batch.query_start_loc, input_batch.idx_mapping,
              input_batch.positions, input_batch.is_padding)
    if not all(isinstance(value, torch.Tensor) and value.ndim == 1 for value in fields):
        raise RuntimeError("PCP Engram requires rank-1 device batch metadata")
    if (any(value.dtype not in (torch.int32, torch.int64) for value in fields[:3])
            or input_batch.is_padding.dtype != torch.bool
            or any(value.device != window.device for value in fields)):
        raise RuntimeError("PCP Engram metadata requires integer offsets/slots/positions, bool padding and the window device")
    count = input_batch.num_reqs
    if not 0 <= count <= window.shape[0]:
        raise RuntimeError(f"PCP Engram lookback capacity exceeded: {count} / {window.shape[0]}")
    if (input_batch.query_start_loc.numel() < count + 1
            or input_batch.idx_mapping.numel() < count
            or input_batch.positions.ndim != 1
            or input_batch.is_padding.shape != input_batch.positions.shape):
        raise RuntimeError("PCP Engram requires device query offsets, request slots, positions and padding")
    if count == 0:
        window.fill_(-1)
        return
    starts = input_batch.query_start_loc[:count].long()
    ends = input_batch.query_start_loc[1:count + 1].long()
    if input_batch.positions.numel() == 0:
        torch._assert_async((ends == starts).all(), "nonempty PCP fragment has no positions")
        window.fill_(-1)
        return
    torch._assert_async(((starts >= 0) & (ends >= starts)
                         & (ends <= input_batch.positions.numel())).all(),
                        "PCP Engram query offsets exceed local positions")
    active = ends > starts
    safe_query = starts.clamp(0, input_batch.positions.numel() - 1)
    active = active & ~input_batch.is_padding[safe_query]
    # MRV2's host num_computed_tokens_np is an optimistic upper bound during
    # async MTP. HcuPCPManager preserves rejection-corrected device positions;
    # each fragment's first real query position is the exact history cursor.
    cursors = input_batch.positions[safe_query].long()
    slots = input_batch.idx_mapping[:count].long()
    history = req_states.all_token_ids.gpu
    if (not isinstance(history, torch.Tensor) or history.ndim != 2
            or history.shape[0] == 0 or history.shape[1] == 0
            or history.dtype not in (torch.int32, torch.int64) or history.device != window.device):
        raise RuntimeError("PCP Engram requires nonempty rank-2 integer token history on the window device")
    valid_slots = (slots >= 0) & (slots < history.shape[0])
    torch._assert_async((~active | valid_slots).all(), "PCP Engram request slot is outside token history")
    torch._assert_async((~active | ((cursors >= 0) & (cursors <= history.shape[1]))).all(),
                        "PCP Engram cursor is outside token history")
    offsets = cursors[:, None] - 1 - torch.arange(window.shape[1], device=window.device)
    values = history[slots.clamp(0, history.shape[0] - 1)[:, None],
                     offsets.clamp(0, history.shape[1] - 1)]
    valid = active[:, None] & valid_slots[:, None] & (offsets >= 0) & (offsets < history.shape[1])
    window[:count].copy_(torch.where(valid, values, -1))
    window[count:].fill_(-1)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    state = require_class(target, "DeepseekV41ModelState", TARGET_MODULE + ".DeepseekV41ModelState")
    if getattr(state, _MARKER, False):
        if not all(getattr(vars(state).get(name), _WRAPPER, False) for name in ("__init__", "prepare_inputs")):
            raise PatchCompatibilityError("stale V41 PCP lookback marker; restart worker")
        return False
    original_init = require_callable(state, "__init__", TARGETS[0])
    original_prepare = require_callable(state, "prepare_inputs", TARGETS[1])
    default = require_class(target, "DefaultModelState", TARGET_MODULE + ".DefaultModelState")
    base_prepare = require_callable(default, "prepare_inputs", "DefaultModelState.prepare_inputs")
    require_exact_signature(original_init, TARGETS[0], positional=("self", "vllm_config", "model", "encoder_cache", "device"))
    for function, name in ((original_prepare, TARGETS[1]), (base_prepare, "DefaultModelState.prepare_inputs")):
        require_exact_signature(function, name, positional=("self", "input_batch", "req_states"))
    for function, name, expected in ((original_init, TARGETS[0], _NATIVE_INIT),
            (original_prepare, TARGETS[1], _NATIVE_PREPARE),
            (base_prepare, "DefaultModelState.prepare_inputs", _DEFAULT_PREPARE)):
        _require_source(function, name, expected)

    @functools.wraps(original_init)
    def hcu_init(self, vllm_config, model, encoder_cache, device):
        original_init(self, vllm_config, model, encoder_cache, device)
        window = self.lookback_token_ids
        if vllm_config.parallel_config.prefill_context_parallel_size > 1 and window is not None:
            # Current HcuPCPManager._max_local_reqs is exactly 3*max_num_seqs.
            self.lookback_token_ids = window.new_full((3 * self.max_num_reqs, window.shape[1]), -1)

    @functools.wraps(original_prepare)
    def hcu_prepare(self, input_batch, req_states):
        if self.vllm_config.parallel_config.prefill_context_parallel_size <= 1:
            return original_prepare(self, input_batch, req_states)
        model_inputs = base_prepare(self, input_batch, req_states)
        if self.lookback_token_ids is not None:
            _fill_fragment_lookback(self.lookback_token_ids, input_batch, req_states)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
        return model_inputs

    for function in (hcu_init, hcu_prepare):
        setattr(function, _WRAPPER, True)
    state._vllm_hcu_original_lookback_init = original_init
    state._vllm_hcu_original_lookback_prepare = original_prepare
    state.__init__ = hcu_init
    state.prepare_inputs = hcu_prepare
    setattr(state, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGETS", "TARGET_MODULE", "apply", "apply_to_module"]
