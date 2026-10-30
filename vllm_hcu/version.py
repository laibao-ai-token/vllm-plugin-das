# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""vLLM-HCU release-series and installed-distribution version metadata."""

from __future__ import annotations

import importlib.metadata as importlib_metadata


# This migration targets the community vLLM 0.32.0 release line.
# Matching metadata is a prerequisite, not proof of runtime compatibility.
# ``setup.py`` computes build provenance for wheel metadata without rewriting
# this tracked file.
__version__ = "0.32.0.dev0"
__version_tuple__ = (0, 32, 0)
# Verified from the fixed core SCM recipe with write_to=None: the
# observed base version is 0.32.0rc1.dev16+g5df63a720, release (0, 32, 0).
__vllm_target_version__ = "0.32.0"
__vllm_upstream_sha__ = "5df63a7203cdb9328d824a27b682e77c7b9928be"
# The paired core is community vLLM, not an OpenDAS wheel build.
__vllm_opendas_sha__ = ""


def get_hcu_version() -> str:
    """Return installed wheel metadata, or the source-tree base version."""

    try:
        return importlib_metadata.version("vllm_hcu")
    except importlib_metadata.PackageNotFoundError:
        return __version__


__hcu_version__ = get_hcu_version()


__all__ = [
    "__hcu_version__",
    "__version__",
    "__version_tuple__",
    "get_hcu_version",
]
