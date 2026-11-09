"""AITER mHC provider ABI checks.

Run: PYTHONPATH=adapter_extension/src pytest -q
tests/vllm033/test_mhc_runtime_provider_abi.py
"""

from types import SimpleNamespace

import pytest

from hcu_vllm_das import mhc_runtime


class _Argument:
    def __init__(self, type_name, *, kwarg_only=False):
        self.type = type_name
        self.kwarg_only = kwarg_only


class _Schema:
    def __init__(self, arguments, returns):
        self.arguments = tuple(_Argument(type_name) for type_name in arguments)
        self.returns = tuple(_Argument(type_name) for type_name in returns)

    def __str__(self):
        return 'fake_mhc_schema'


def _schemas():
    schemas = {}
    for name, (argument_types, return_types) in (
        mhc_runtime._PROVIDER_SCHEMA_TYPES.items()
    ):
        schemas[name] = _Schema(
            [types[0] if isinstance(types, tuple) else types
             for types in argument_types],
            return_types,
        )
    return schemas


def _generic_provider():
    def provider(*args, **kwargs):
        return None

    return provider


def test_load_provider_accepts_torch_compile_guard_wrappers(monkeypatch):
    functions = {
        name: _generic_provider()
        for name in mhc_runtime._PROVIDER_PARAMETER_NAMES
    }
    module = SimpleNamespace(**functions)
    schemas = _schemas()
    monkeypatch.setattr(
        mhc_runtime,
        'import_module',
        lambda name: module if name == 'aiter.ops.mhc' else None,
    )
    monkeypatch.setattr(mhc_runtime, '_provider_schema', schemas.__getitem__)

    loaded = mhc_runtime.load_provider()

    assert loaded.mhc_pre is functions['mhc_pre']
    assert loaded.mhc_post is functions['mhc_post']
    assert loaded.mhc_pre_gemm_sqrsum is functions['mhc_pre_gemm_sqrsum']
    assert loaded.mhc_pre_big_fuse is functions['mhc_pre_big_fuse']


def test_generic_wrapper_with_wrong_schema_is_rejected(monkeypatch):
    functions = {
        name: _generic_provider()
        for name in mhc_runtime._PROVIDER_PARAMETER_NAMES
    }
    module = SimpleNamespace(**functions)
    schemas = _schemas()
    schemas['mhc_pre'] = _Schema(
        ('Tensor', 'Tensor', 'Tensor', 'float', 'float', 'float', 'float',
         'SymInt', 'bool'),
        ('Tensor', 'Tensor', 'Tensor'),
    )
    monkeypatch.setattr(
        mhc_runtime,
        'import_module',
        lambda name: module if name == 'aiter.ops.mhc' else None,
    )
    monkeypatch.setattr(mhc_runtime, '_provider_schema', schemas.__getitem__)

    with pytest.raises(RuntimeError, match='provider schema changed: mhc_pre'):
        mhc_runtime.load_provider()
