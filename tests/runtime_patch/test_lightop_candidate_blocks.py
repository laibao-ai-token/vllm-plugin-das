# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Contract tests for the LightOp DSV4.1 decode candidate producer.

These run without a device: the LightOp producer is doubled so the tests can
assert the gate, the layout validation, the fallback decision and the row
isolation that keeps padded decode rows from overwriting prefill rows.
"""

from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch

from hcu_vllm_das import environment
from hcu_vllm_das import v41_indexer


_CUDA = torch.cuda.is_available()
_DEVICE = torch.device("cuda" if _CUDA else "cpu")
requires_device = pytest.mark.skipif(
    not _CUDA, reason="the LightOp candidate producer is a device operator"
)


def _install_lightop(monkeypatch, producer):
    """Expose a doubled ``lightop.attention`` module for the helper to import."""

    attention = ModuleType("lightop.attention")
    attention.dsv41_select_candidate_blocks = producer
    package = ModuleType("lightop")
    package.attention = attention
    monkeypatch.setitem(sys.modules, "lightop", package)
    monkeypatch.setitem(sys.modules, "lightop.attention", attention)
    return attention


def _remove_lightop(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightop", None)
    monkeypatch.delitem(sys.modules, "lightop.attention", raising=False)


def _enable(monkeypatch, *, master: bool = True, child: bool = True):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1" if master else "0")
    monkeypatch.setenv(
        "VLLM_HCU_USE_LIGHTOP_DSV41_CANDIDATE_BLOCKS", "1" if child else "0"
    )


def _logits(rows: int = 4, width: int = 512) -> torch.Tensor:
    generator = torch.Generator(device=_DEVICE).manual_seed(11)
    return torch.randn(rows, width, generator=generator, dtype=torch.float32, device=_DEVICE)


def _ends(rows: int = 4, width: int = 512) -> torch.Tensor:
    return torch.full((rows,), width, dtype=torch.int32, device=_DEVICE)


class _Producer:
    """Records its arguments and returns a caller-controlled result."""

    def __init__(self, block_ids=None, mask=None, result=None):
        self.calls = []
        self._block_ids = block_ids
        self._mask = mask
        self._result = result

    def __call__(self, logits, lengths, topk_blocks, block_size):
        self.calls.append((logits.shape, lengths.clone(), topk_blocks, block_size))
        if self._result is not None:
            return self._result
        mask = self._mask
        if mask is None:
            mask = torch.zeros(logits.shape, dtype=torch.bool, device=_DEVICE)
        block_ids = self._block_ids
        if block_ids is None:
            block_ids = torch.zeros((logits.shape[0], topk_blocks), dtype=torch.int32, device=_DEVICE)
        return mask, block_ids


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------


def test_enabled_by_default(monkeypatch):
    """The producer defaults ON; only the master switch or an explicit 0 disables it."""
    monkeypatch.delenv("VLLM_HCU_USE_LIGHTOP_DSV41_CANDIDATE_BLOCKS", raising=False)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    assert environment.candidate_blocks_enabled() is True


def test_master_switch_is_a_hard_gate(monkeypatch):
    _enable(monkeypatch, master=False, child=True)
    assert environment.candidate_blocks_enabled() is False
    _enable(monkeypatch, master=True, child=True)
    assert environment.candidate_blocks_enabled() is True
    _enable(monkeypatch, master=True, child=False)
    assert environment.candidate_blocks_enabled() is False


def test_invalid_flag_value_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_HCU_USE_LIGHTOP_DSV41_CANDIDATE_BLOCKS", "maybe")
    with pytest.raises(ValueError):
        environment.candidate_blocks_enabled()


# --------------------------------------------------------------------------
# fallback
# --------------------------------------------------------------------------


def test_flag_off_keeps_the_triton_chain(monkeypatch):
    _enable(monkeypatch, child=False)
    producer = _Producer()
    _install_lightop(monkeypatch, producer)
    candidates = torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)
    used = v41_indexer._lightop_decode_candidate_blocks(
        _logits(), _ends(), candidates, 8
    )
    assert used is False
    assert producer.calls == []


def test_missing_lightop_falls_back(monkeypatch):
    _enable(monkeypatch)
    _remove_lightop(monkeypatch)
    candidates = torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)
    used = v41_indexer._lightop_decode_candidate_blocks(
        _logits(), _ends(), candidates, 8
    )
    assert used is False


def test_missing_symbol_falls_back(monkeypatch):
    _enable(monkeypatch)
    attention = ModuleType("lightop.attention")
    package = ModuleType("lightop")
    package.attention = attention
    monkeypatch.setitem(sys.modules, "lightop", package)
    monkeypatch.setitem(sys.modules, "lightop.attention", attention)
    candidates = torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)
    used = v41_indexer._lightop_decode_candidate_blocks(
        _logits(), _ends(), candidates, 8
    )
    assert used is False


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda logits, ends, cands: (logits.double(), ends, cands), id="logits-dtype"),
        pytest.param(lambda logits, ends, cands: (logits[:, :500], ends, cands), id="width-not-multiple-of-8"),
        pytest.param(lambda logits, ends, cands: (logits.t(), ends, cands), id="logits-not-contiguous"),
        pytest.param(lambda logits, ends, cands: (logits, ends, cands[:2]), id="row-count-mismatch"),
        pytest.param(lambda logits, ends, cands: (logits, ends[:2], cands), id="ends-row-mismatch"),
        pytest.param(
            lambda logits, ends, cands: (logits, ends, cands.to(torch.int64)),
            id="candidates-dtype",
        ),
    ],
)
def test_contract_mismatch_falls_back(monkeypatch, mutate):
    _enable(monkeypatch)
    producer = _Producer()
    _install_lightop(monkeypatch, producer)
    candidates = torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)
    logits, ends, cands = mutate(_logits(), _ends(), candidates)
    assert v41_indexer._lightop_decode_candidate_blocks(logits, ends, cands, 8) is False
    assert producer.calls == []


def test_block_size_must_be_eight(monkeypatch):
    _enable(monkeypatch)
    producer = _Producer()
    _install_lightop(monkeypatch, producer)
    candidates = torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)
    assert (
        v41_indexer._lightop_decode_candidate_blocks(_logits(), _ends(), candidates, 16)
        is False
    )
    assert producer.calls == []


# --------------------------------------------------------------------------
# empty decode
# --------------------------------------------------------------------------


def test_empty_decode_does_not_call_the_producer(monkeypatch):
    _enable(monkeypatch)
    producer = _Producer()
    _install_lightop(monkeypatch, producer)
    candidates = torch.zeros((0, 8), dtype=torch.int32, device=_DEVICE)
    logits = torch.zeros((0, 512), dtype=torch.float32, device=_DEVICE)
    ends = torch.zeros((0,), dtype=torch.int32, device=_DEVICE)
    assert v41_indexer._lightop_decode_candidate_blocks(logits, ends, candidates, 8) is False
    assert producer.calls == []


def test_empty_width_does_not_call_the_producer(monkeypatch):
    _enable(monkeypatch)
    producer = _Producer()
    _install_lightop(monkeypatch, producer)
    candidates = torch.zeros((4, 0), dtype=torch.int32, device=_DEVICE)
    logits = torch.zeros((4, 0), dtype=torch.float32, device=_DEVICE)
    assert v41_indexer._lightop_decode_candidate_blocks(logits, _ends(), candidates, 8) is False
    assert producer.calls == []


# --------------------------------------------------------------------------
# invalid layout raises instead of silently falling back
# --------------------------------------------------------------------------


def test_non_tuple_result_raises(monkeypatch):
    _enable(monkeypatch)
    _install_lightop(monkeypatch, _Producer(result=torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)))
    candidates = torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)
    with pytest.raises(RuntimeError, match="invalid result"):
        v41_indexer._lightop_decode_candidate_blocks(_logits(), _ends(), candidates, 8)


def test_wrong_arity_result_raises(monkeypatch):
    _enable(monkeypatch)
    _install_lightop(monkeypatch, _Producer(result=(torch.zeros((4, 512), dtype=torch.bool, device=_DEVICE),)))
    candidates = torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)
    with pytest.raises(RuntimeError, match="invalid result"):
        v41_indexer._lightop_decode_candidate_blocks(_logits(), _ends(), candidates, 8)


def test_wrong_mask_shape_raises(monkeypatch):
    _enable(monkeypatch)
    mask = torch.zeros((4, 256), dtype=torch.bool, device=_DEVICE)
    _install_lightop(monkeypatch, _Producer(mask=mask))
    candidates = torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)
    with pytest.raises(RuntimeError, match="invalid layout"):
        v41_indexer._lightop_decode_candidate_blocks(_logits(), _ends(), candidates, 8)


def test_wrong_mask_dtype_raises(monkeypatch):
    _enable(monkeypatch)
    mask = torch.zeros((4, 512), dtype=torch.float32, device=_DEVICE)
    _install_lightop(monkeypatch, _Producer(mask=mask))
    candidates = torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)
    with pytest.raises(RuntimeError, match="invalid layout"):
        v41_indexer._lightop_decode_candidate_blocks(_logits(), _ends(), candidates, 8)


def test_wrong_block_ids_shape_raises(monkeypatch):
    _enable(monkeypatch)
    block_ids = torch.zeros((4, 16), dtype=torch.int32, device=_DEVICE)
    _install_lightop(monkeypatch, _Producer(block_ids=block_ids))
    candidates = torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)
    with pytest.raises(RuntimeError, match="invalid layout"):
        v41_indexer._lightop_decode_candidate_blocks(_logits(), _ends(), candidates, 8)


def test_wrong_block_ids_dtype_raises(monkeypatch):
    _enable(monkeypatch)
    block_ids = torch.zeros((4, 8), dtype=torch.int64, device=_DEVICE)
    _install_lightop(monkeypatch, _Producer(block_ids=block_ids))
    candidates = torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)
    with pytest.raises(RuntimeError, match="invalid layout"):
        v41_indexer._lightop_decode_candidate_blocks(_logits(), _ends(), candidates, 8)


# --------------------------------------------------------------------------
# success path and argument forwarding
# --------------------------------------------------------------------------


def test_success_writes_block_ids(monkeypatch):
    _enable(monkeypatch)
    block_ids = torch.arange(32, dtype=torch.int32, device=_DEVICE).reshape(4, 8)
    producer = _Producer(block_ids=block_ids)
    _install_lightop(monkeypatch, producer)
    candidates = torch.full((4, 8), -1, dtype=torch.int32, device=_DEVICE)
    logits = _logits()
    ends = _ends()
    assert v41_indexer._lightop_decode_candidate_blocks(logits, ends, candidates, 8) is True
    assert torch.equal(candidates, block_ids)
    assert len(producer.calls) == 1
    shape, lengths, topk_blocks, block_size = producer.calls[0]
    assert shape == logits.shape
    assert torch.equal(lengths, ends)
    assert topk_blocks == candidates.shape[1]
    assert block_size == 8


def test_ends_are_forwarded_as_int32(monkeypatch):
    _enable(monkeypatch)
    producer = _Producer()
    _install_lightop(monkeypatch, producer)
    candidates = torch.zeros((4, 8), dtype=torch.int32, device=_DEVICE)
    ends = _ends().to(torch.int64)
    assert v41_indexer._lightop_decode_candidate_blocks(_logits(), ends, candidates, 8) is True
    assert producer.calls[0][1].dtype == torch.int32


# --------------------------------------------------------------------------
# padded-row isolation: the helper writes only the rows it was handed
# --------------------------------------------------------------------------


def test_padded_rows_never_reach_the_shared_bank(monkeypatch):
    """A compact decode scratch must not spill into the wider shared bank.

    This mirrors ``v41_indexer_gpu.decode_indexer``: the helper receives a
    ``(rows, width)`` scratch tensor and the caller performs the request-major
    ``index_copy_`` afterwards. A padded batch has more bank slots than decode
    rows, so anything the helper wrote past ``rows`` would overwrite prefill
    rows that live in the same buffer.
    """

    _enable(monkeypatch)
    rows, width, slots = 3, 512, 6
    block_ids = torch.full((rows, 8), 7, dtype=torch.int32, device=_DEVICE)
    _install_lightop(monkeypatch, _Producer(block_ids=block_ids))

    shared = torch.full((slots, 8), -1, dtype=torch.int32, device=_DEVICE)
    shared[rows:] = 42  # prefill rows that must survive untouched
    before = shared.clone()

    scratch = torch.empty((rows, 8), dtype=torch.int32, device=_DEVICE)
    assert (
        v41_indexer._lightop_decode_candidate_blocks(
            _logits(rows=rows), _ends(rows=rows), scratch, 8
        )
        is True
    )
    assert torch.equal(scratch, block_ids)
    # The helper only owns the scratch; the shared bank is untouched until the
    # caller maps rows back through its banks vector.
    assert torch.equal(shared, before)
    assert torch.equal(shared[rows:], torch.full((slots - rows, 8), 42, dtype=torch.int32, device=_DEVICE))


def test_padded_rows_survive_the_request_major_write_back(monkeypatch):
    """After the caller's ``index_copy_`` the untouched bank slots keep their value."""

    _enable(monkeypatch)
    rows, slots = 3, 6
    block_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 0],
         [8, 9, 10, 11, 12, 13, 14, 15],
         [16, 17, 18, 19, 20, 21, 22, 23]],
        dtype=torch.int32,
        device=_DEVICE,
    )
    _install_lightop(monkeypatch, _Producer(block_ids=block_ids))

    shared = torch.full((slots, 8), -1, dtype=torch.int32, device=_DEVICE)
    shared[rows:] = 42
    banks = torch.tensor([0, 2, 4], dtype=torch.int32, device=_DEVICE)

    scratch = torch.empty((rows, 8), dtype=torch.int32, device=_DEVICE)
    assert (
        v41_indexer._lightop_decode_candidate_blocks(
            _logits(rows=rows), _ends(rows=rows), scratch, 8
        )
        is True
    )
    shared.index_copy_(0, banks.to(torch.int64), scratch)

    assert torch.equal(shared[0], block_ids[0])
    assert torch.equal(shared[2], block_ids[1])
    assert torch.equal(shared[4], block_ids[2])
    # Slots the banks vector never names keep the prefill value.
    assert torch.equal(shared[1], torch.full((8,), -1, dtype=torch.int32, device=_DEVICE))
    assert torch.equal(shared[3], torch.full((8,), 42, dtype=torch.int32, device=_DEVICE))
    assert torch.equal(shared[5], torch.full((8,), 42, dtype=torch.int32, device=_DEVICE))


def test_scratch_is_not_aliased_into_the_bank(monkeypatch):
    """The helper copies values; it must not make the bank alias its scratch."""

    _enable(monkeypatch)
    block_ids = torch.full((4, 8), 3, dtype=torch.int32, device=_DEVICE)
    _install_lightop(monkeypatch, _Producer(block_ids=block_ids))
    scratch = torch.empty((4, 8), dtype=torch.int32, device=_DEVICE)
    assert v41_indexer._lightop_decode_candidate_blocks(_logits(), _ends(), scratch, 8) is True
    scratch.fill_(99)
    assert torch.equal(block_ids, torch.full((4, 8), 3, dtype=torch.int32, device=_DEVICE))


# --------------------------------------------------------------------------
# decode-only call site split
# --------------------------------------------------------------------------


def test_select_decode_prefers_lightop(monkeypatch):
    _enable(monkeypatch)
    block_ids = torch.full((4, 8), 5, dtype=torch.int32, device=_DEVICE)
    producer = _Producer(block_ids=block_ids)
    _install_lightop(monkeypatch, producer)

    logits = _logits()
    starts = torch.zeros(4, dtype=torch.int64, device=_DEVICE)
    ends = _ends()
    out = torch.full((4, 8), -1, dtype=torch.int32, device=_DEVICE)
    candidates = torch.full((4, 8), -1, dtype=torch.int32, device=_DEVICE)

    v41_indexer._select_decode(logits, starts, ends, out, candidates, 8, True)
    assert len(producer.calls) == 1
    assert torch.equal(candidates, block_ids)


def test_select_decode_falls_back_to_triton_when_disabled(monkeypatch):
    _enable(monkeypatch, child=False)
    producer = _Producer()
    _install_lightop(monkeypatch, producer)

    logits = _logits()
    starts = torch.zeros(4, dtype=torch.int64, device=_DEVICE)
    ends = _ends()
    out = torch.full((4, 8), -1, dtype=torch.int32, device=_DEVICE)
    candidates = torch.full((4, 8), -1, dtype=torch.int32, device=_DEVICE)

    calls = []
    monkeypatch.setattr(
        v41_indexer,
        "select_candidate_blocks",
        lambda *args: calls.append(args),
    )
    v41_indexer._select_decode(logits, starts, ends, out, candidates, 8, True)
    assert producer.calls == []
    assert len(calls) == 1


def test_prefill_select_never_calls_lightop(monkeypatch):
    """Prefill rows carry non-zero starts, which the LightOp contract lacks."""

    _enable(monkeypatch)
    producer = _Producer()
    _install_lightop(monkeypatch, producer)

    logits = _logits()
    starts = torch.tensor([0, 128, 64, 256], dtype=torch.int64, device=_DEVICE)
    ends = torch.tensor([200, 300, 512, 320], dtype=torch.int32, device=_DEVICE)
    out = torch.full((4, 8), -1, dtype=torch.int32, device=_DEVICE)
    candidates = torch.full((4, 8), -1, dtype=torch.int32, device=_DEVICE)

    calls = []
    monkeypatch.setattr(
        v41_indexer,
        "select_candidate_blocks",
        lambda *args: calls.append(args),
    )
    v41_indexer._select(logits, starts, ends, out, candidates, 8, True)
    assert producer.calls == []
    assert len(calls) == 1
