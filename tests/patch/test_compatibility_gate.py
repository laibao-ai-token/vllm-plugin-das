# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import vllm_hcu as plugin
from vllm_hcu import compatibility
from vllm_hcu import version as hcu_version
from vllm_hcu.patch import (
    IMPORT_COORDINATOR,
    PATCH_REGISTRY,
    apply_platform_patches,
    apply_worker_patches,
)


class _Distribution:
    def __init__(self, version: str, location: Path) -> None:
        self.version = version
        self._location = location

    def locate_file(self, path: str) -> Path:
        assert path == ""
        return self._location


@pytest.fixture(autouse=True)
def _clean_process_state(monkeypatch: pytest.MonkeyPatch):
    IMPORT_COORDINATOR.reset_for_tests()
    monkeypatch.setattr(plugin, "_PLATFORM_INIT_FAILURE", None)
    yield
    IMPORT_COORDINATOR.reset_for_tests()


def _install_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    location = tmp_path / "site-packages"
    monkeypatch.setattr(
        compatibility.importlib_metadata,
        "distribution",
        lambda name: _Distribution(value, location),
    )


def test_fixed_community_scm_release_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Observed via the fixed core SCM recipe without writing its _version.py.
    value = "0.32.0rc1.dev16+g5df63a720"
    _install_version(monkeypatch, tmp_path, value)

    result = compatibility.ensure_vllm_compatible()

    assert result.compatible
    assert result.expected == "0.32.0"
    assert result.actual_version == value
    assert result.vllm_location == str((tmp_path / "site-packages").resolve())


@pytest.mark.parametrize(
    "value",
    (
        "0.32.0",
        "0.32.0+local.test",
        "0.32.0rc0",
        "0.32.0rc1.dev16+g5df63a720",
        "0.32.0.dev999+local.test",
        "0.32.0.post1",
    ),
)
def test_v0320_release_line_accepts_rolling_builds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    _install_version(monkeypatch, tmp_path, value)

    result = compatibility.ensure_vllm_compatible()

    assert result.compatible
    assert result.actual_version == value
    assert result.reason == (
        "installed vLLM build matches supported release line 0.32.0"
    )


@pytest.mark.parametrize(
    "value",
    (
        "0.28.1",
        "0.29.1rc0.dev283",
        "0.31.1",
        "0.32.1",
        "0.33.0",
        "0.32.0.0",
        "0.32",
        "1!0.32.0",
        "not a version",
    ),
)
def test_unsupported_or_invalid_vllm_metadata_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    _install_version(monkeypatch, tmp_path, value)

    with pytest.raises(compatibility.VllmCompatibilityError) as raised:
        compatibility.ensure_vllm_compatible()

    message = str(raised.value)
    assert "expected=0.32.0" in message
    assert f"actual={value!r}" in message
    assert "upstream_sha=5df63a7203cdb9328d824a27b682e77c7b9928be" in message
    assert "opendas_sha=;" in message
    assert "vllm_hcu=" in message
    assert "vllm_location=" in message
    assert "vllm_hcu_location=" in message


def test_missing_vllm_distribution_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str):
        raise compatibility.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(compatibility.importlib_metadata, "distribution", missing)

    with pytest.raises(
        compatibility.VllmCompatibilityError,
        match="actual='not installed'",
    ):
        compatibility.ensure_vllm_compatible()


def test_runtime_hcu_version_prefers_metadata_with_source_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hcu_version.importlib_metadata,
        "version",
        lambda name: "0.32.0.dev0+local.test",
    )
    assert (
        hcu_version.get_hcu_version()
        == "0.32.0.dev0+local.test"
    )

    def missing(name: str):
        raise hcu_version.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(hcu_version.importlib_metadata, "version", missing)
    assert hcu_version.get_hcu_version() == hcu_version.__version__


def test_every_apply_boundary_fails_before_registry_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vllm_hcu.patch.platform as platform_dispatcher
    import vllm_hcu.patch.worker as worker_dispatcher

    _install_version(monkeypatch, tmp_path, "0.22.0")

    for apply in (
        apply_platform_patches,
        apply_worker_patches,
        platform_dispatcher.apply_platform_patches,
        worker_dispatcher.apply_worker_patches,
    ):
        with pytest.raises(compatibility.VllmCompatibilityError):
            apply()
        assert IMPORT_COORDINATOR.registrations() == ()
        assert PATCH_REGISTRY.report()["patches"] == {}


def test_all_plugin_entries_gate_and_platform_probe_latches_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_version(monkeypatch, tmp_path, "0.24.0")

    # vLLM probes the platform plugin under a broad exception handler.
    # Preserve selection once, then surface the exact compatibility class.
    assert plugin.hcu_platform_plugin() == plugin._PLATFORM_CLASS_PATH
    with pytest.raises(
        compatibility.VllmCompatibilityError,
        match="previously failed",
    ):
        plugin.hcu_platform_plugin()

    for entry in (
        plugin.hcu_platform_register_model,
        plugin.hcu_platform_register_ops,
    ):
        with pytest.raises(compatibility.VllmCompatibilityError):
            entry()

    assert IMPORT_COORDINATOR.registrations() == ()
    assert PATCH_REGISTRY.report()["patches"] == {}


def test_source_and_build_metadata_share_fixed_community_contract():
    from packaging.version import Version

    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "setup.py").read_text(encoding="utf-8"))
    base = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BASE_VERSION" for t in node.targets)
    )
    setup_call = next(
        node for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == "setup"
    ).value
    python_requires = next(
        ast.literal_eval(kw.value) for kw in setup_call.keywords
        if kw.arg == "python_requires"
    )
    assert base == hcu_version.__version__ == "0.32.0.dev0"
    assert Version(base).release == hcu_version.__version_tuple__ == (0, 32, 0)
    assert hcu_version.__vllm_target_version__ == "0.32.0"
    assert hcu_version.__vllm_upstream_sha__ == (
        "5df63a7203cdb9328d824a27b682e77c7b9928be"
    )
    assert hcu_version.__vllm_opendas_sha__ == ""
    assert python_requires == ">=3.11,<3.15"
