# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""CPU candidate regressions without importing vLLM or HCU runtime modules."""
import ast
import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _source(relative):
    return (Path(__file__).resolve().parents[2] / relative).read_text()


def _helper():
    source = _source("vllm_hcu/v1/attention/ops/indexer_candidates.py")
    names = {"_candidate_bounds", "candidate_topk"}
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<candidate-helper>", "exec"), namespace)
    return namespace["candidate_topk"]


def test_candidate_short_rows_keep_holes_and_nan():
    # Inputs are already masked by the native candidate mask. Short-row
    # ordering must preserve holes instead of resurrecting every causal token.
    logits = torch.full((3, 11), -float("inf"))
    logits[0, 2] = 9
    logits[0, 4] = float("nan")
    logits[0, 7] = 5
    logits[1, 0] = 8
    logits[1, 2] = 5
    starts = torch.tensor([2, 0, 1], dtype=torch.int32)
    ends = torch.tensor([9, 4, 5], dtype=torch.int32)
    result = _helper()(logits, 10, starts, ends)
    expected = torch.tensor([[0, 2, 5] + [-1] * 7,
                             [0, 2] + [-1] * 8,
                             [-1] * 10], dtype=torch.int32)
    assert torch.equal(result, expected)
    assert torch.isnan(logits[0, 4])


def test_candidate_long_rows_rank_scores_and_keep_nan():
    logits = torch.tensor([[-float("inf"), 7., -float("inf"),
                            float("nan"), 4., -float("inf")]])
    result = _helper()(logits, 2, torch.tensor([0]), torch.tensor([6]))
    assert result.tolist() == [[3, 1]]


def test_candidate_empty_and_zero_rows():
    fn = _helper()
    result = fn(torch.empty(2, 0), 3, None, torch.zeros(2, dtype=torch.int32))
    assert result.shape == (2, 3) and (result == -1).all()
    assert fn(torch.empty(0, 5), 3, None, torch.empty(0, dtype=torch.int32)).shape == (0, 3)
    with pytest.raises(ValueError, match="bounds"):
        fn(torch.empty(2, 5), 3, None, torch.zeros(1, dtype=torch.int32))


def test_candidate_constructor_and_opaque_mutation_contract():
    source = _source("vllm_hcu/model_executor/layers/sparse_attn_indexer.py")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SparseAttnIndexer")
    ctor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert ctor.args.kwarg is None
    ctor.body = [ast.Pass()]
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ctor], type_ignores=[])),
                 "<indexer-constructor>", "exec"), namespace)
    signature = inspect.signature(namespace["__init__"])
    expected = {"candidate_blocks": None, "candidate_block_size": 0, "candidate_write": False}
    for name, default in expected.items():
        assert signature.parameters[name].default == default
    pool = object()
    bound = signature.bind(None, None, 128, "ue8m0", 512, 128, 32, 32, None,
                           candidate_blocks=pool, candidate_block_size=8, candidate_write=True)
    assert bound.arguments["candidate_blocks"] is pool
    registrations = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name) and n.func.id == "direct_register_custom_op"]
    call = next(n for n in registrations if any(k.arg == "op_name" and
                isinstance(k.value, ast.Constant) and k.value.value == "hcu_sparse_attn_indexer"
                for k in n.keywords))
    mutations = ast.literal_eval(next(k.value for k in call.keywords if k.arg == "mutates_args"))
    assert set(mutations) == {"kv_cache", "topk_indices_buffer", "candidate_blocks"}
    assert "cudagraph_unsafe" in ast.unparse(call)
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_hip")
    op = next(n for n in ast.walk(forward) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "hcu_sparse_attn_indexer")
    assert [ast.unparse(arg) for arg in op.args[-3:]] == [
        "self.candidate_blocks", "self.candidate_block_size", "self.candidate_write"]
