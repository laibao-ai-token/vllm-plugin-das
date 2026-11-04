# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Offload and parallel-shard DeepSeek V4.1 Engram tables on HCU.

Re-hosted from the 0.29 line (``patch_deepseek_v41_engram_offload.py``, 759 lines,
lost by the v0.32.0 migration checkpoint).  Without it the v0.32 Engram table is
replicated whole on every rank of a TP1/PCP8 node: ``sum(head_sizes)`` equals
``engram_num_embeddings`` (384,006,168 / 384,016,682 rows, verified against the
checkpoint through the core's own ``EngramLayout``), which at
``head_dim + head_dim // 32 = 264`` bytes per row is 94.42 GiB per table, i.e.
188.83 GiB per rank for the two Engram layers and 1510.7 GiB per node.  That is
what ``serve-attempt-09`` died on (``engram.py:1066``, ``torch.AcceleratorError:
CUDA error: out of memory``).

Two independent things are restored here:

1. **Sharding.**  For PCP>1 the shard count extends to ``TP*PCP``; PCP ranks with
   a fixed TP rank own adjacent hash-head shards, the PCP collective restores
   those heads, and the TP collective then completes head order.  At TP1/PCP8
   this is 3 of the 24 heads per rank -> 11.80 GiB per table, 23.60 GiB per rank,
   188.83 GiB per node.
2. **Storage.**  Large ``pin_memory=True`` requests are unreliable on HCU, so the
   table is allocated as anonymous ``mmap`` pages and registered with
   ``cudaHostRegister`` instead of going through ``torch.empty(pin_memory=True)``.
   Observed on this stack (nmz16, 2026-10-31): one process failed a 464.90 GiB
   cumulative ``pin_memory`` request with the exact ``serve-attempt-09`` error
   while ``MemAvailable`` was still 797 GiB, with ``Mlocked``/``Unevictable``
   staying at 0.

Why the core class is replaced by name instead of re-calling it with a mirrored
argument list: core ``Engram._create_embedding`` (``engram.py:1577-1589``)
constructs the **module-global** name ``ParallelEngramEmbedding``.  Rebinding that
global to the HCU subclass makes the core build the table, so every keyword the
core decides to pass (``cpu_offload``, ``dp_shared_memory``, ``use_thp``, and
whatever a later core adds) flows through unchanged.  Wrapping ``_create_embedding``
and hand-writing the argument list would silently drift the moment the core adds or
renames a keyword -- the drift surface shrinks to zero with the name replacement.

Deliberately NOT reimplemented, because v0.32 core already owns these and a second
copy would drift: ``dummy_weight_value`` attributes (core ``engram.py:1016-1020``),
the ``is_uva_available()`` gate (core ``:981-982``), and ``_storage()`` / the UVA
view cache (core ``:1095-1108`` with ``_views``/``_view_src`` at ``:983-984``).

The v0.32 ``EngramBatch`` staging path is out of reach for this topology by
construction: ``EngramBatch.create`` returns ``None`` unless ``dp_size > 1``
(core ``:1440-1453``), and PCP sharding does not touch ``dp_size``, so the
per-layer ``prepare_embeddings``/``embed`` path this adapter extends is the one
that runs.  Register this as "to be covered when PCP+DP is opened up".

Checked refusals (fail closed, with greppable messages, never a silent fallback):
  * ``dp_shared_memory=true``                 -> ValueError, "...does not yet
    support dp_shared_memory=true; use dp_shared_memory=false"
  * ``embedding_across_dp=true`` with PCP>1   -> ValueError, "...does not yet
    support embedding_across_dp=true together with PCP>1"
  * runtime DP shard count > 1 with PCP > 1   -> RuntimeError, "HCU Engram does
    not yet support PCP+DP sharding"
  * sequence parallelism with PCP > 1         -> RuntimeError, "HCU Engram PCP
    sharding does not yet support sequence parallelism"

Mismatch behaviour: this adapter is *required* for the V4.1 Engram capacity path,
so a missing core class, a class that cannot be substituted (not a type, not the
module global, ``__init__`` no longer accepting the three config keywords), or a
missing ``Engram`` staging method raises ``PatchCompatibilityError`` rather than
falling open.  The one deliberate fall-open is the platform capability probe: if
this build exposes no ``torch.cuda.cudart().cudaHostRegister`` the adapter
installs nothing and logs an error naming the fallback it left in place.
"""

from __future__ import annotations

import functools
import inspect
import mmap
import weakref
from types import ModuleType

import torch

from vllm.logger import init_logger

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
)

TARGET_MODULE = "vllm.models.deepseek_v41.common.engram"
PATCH_ID = "worker.core_fix.deepseek_v41.engram_offload"
TARGETS = (
    f"{TARGET_MODULE}.ParallelEngramEmbedding",
    f"{TARGET_MODULE}.Engram._init_staging",
    f"{TARGET_MODULE}.Engram.prepare_embeddings",
    f"{TARGET_MODULE}.Engram.embed",
)
_CLASS_MARKER = "_vllm_hcu_dsv41_engram_offload_storage"
_NAME_MARKER = "_vllm_hcu_dsv41_engram_class_replaced"
_ENGRAM_METHOD_MARKER = "_vllm_hcu_dsv41_engram_pcp_methods"
# Keywords the core `_create_embedding` passes and the subclass must be able to
# receive unchanged for the name replacement to stay a drop-in.
_REQUIRED_KEYWORDS = ("cpu_offload", "dp_shared_memory", "use_thp")
logger = init_logger(__name__)

_REFUSE_DP_SHARED = (
    "HCU DeepSeek V4.1 Engram CPU offload does not yet support "
    "dp_shared_memory=true; use dp_shared_memory=false"
)
_REFUSE_ACROSS_DP_WITH_PCP = (
    "HCU DeepSeek V4.1 Engram DP sharding does not yet support "
    "embedding_across_dp=true together with PCP>1"
)
_REFUSE_PCP_PLUS_DP = "HCU Engram does not yet support PCP+DP sharding"
_REFUSE_PCP_WITH_SP = (
    "HCU Engram PCP sharding does not yet support sequence parallelism"
)


def _engram_config():
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    return None if config is None else getattr(config, "engram_config", None)


def _pcp_group():
    from vllm.distributed.parallel_state import get_pcp_group

    try:
        return get_pcp_group()
    except AssertionError:
        # Model-construction probes and PCP=1 startup can precede group setup.
        return None


def _pcp_size() -> int:
    group = _pcp_group()
    return 1 if group is None else int(group.world_size)


def _pcp_rank() -> int:
    group = _pcp_group()
    return 0 if group is None else int(group.rank_in_group)


def _pcp_shard_info(
    tp_size: int, tp_rank: int, pcp_size: int, pcp_rank: int
) -> tuple[int, int]:
    """Community-compatible TP-major Engram shard coordinates, extended by PCP.

    PCP ranks sharing a TP rank own adjacent head shards, so the PCP collective
    restores head order before the TP collective completes it.
    """
    return tp_size * pcp_size, tp_rank * pcp_size + pcp_rank


def _offload_flags(base: type, args: tuple, kwargs: dict) -> dict:
    """Resolve the three core storage keywords without mirroring its call site.

    The core passes them as keywords; binding through the base signature also
    covers a positional call, and any *new* keyword simply rides along in
    ``kwargs`` to ``super().__init__()`` untouched.
    """
    flags = {}
    try:
        bound = inspect.signature(base.__init__).bind_partial(None, *args, **kwargs)
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} cannot bind its arguments: {exc}"
        ) from exc
    bound.apply_defaults()
    for name in _REQUIRED_KEYWORDS:
        flags[name] = bool(bound.arguments.get(name, False))
    return flags


def _require_supported_config(flags: dict) -> None:
    if flags["dp_shared_memory"]:
        raise ValueError(_REFUSE_DP_SHARED)
    config = _engram_config()
    if (
        config is not None
        and getattr(config, "embedding_across_dp", False)
        and _pcp_size() > 1
    ):
        raise ValueError(_REFUSE_ACROSS_DP_WITH_PCP)


def _gather_padded_pcp_hash_ids(hash_ids: torch.Tensor):
    """Return PCP-rank-major hash rows and the common token slot."""
    group = _pcp_group()
    if group is None or int(group.world_size) == 1:
        return hash_ids, int(hash_ids.shape[0])
    # Callers hand in a layer slice of [tokens, layers, heads], which is not
    # contiguous and cannot feed all_gather_into_tensor directly.
    hash_ids = hash_ids.contiguous()
    count = torch.tensor(
        [hash_ids.shape[0]], dtype=torch.int64, device=hash_ids.device
    )
    counts = group.all_gather(count, dim=0)
    slot = int(counts.max().item())
    if hash_ids.shape[0] < slot:
        padding = hash_ids.new_full(
            (slot - hash_ids.shape[0], *hash_ids.shape[1:]), -1
        )
        hash_ids = torch.cat((hash_ids, padding), dim=0)
    return group.all_gather(hash_ids, dim=0), slot


def _shard_label(table) -> str:
    if _pcp_size() > 1 and table.dp_size == 1:
        return "TPxPCP"
    return "DP" if table.dp_size > 1 else "TP"


def _pcp_sharded(engram) -> bool:
    """True when this layer's table is PCP-sharded by this adapter.

    ``dp_size > 1`` means the core already shards the table over the Engram DP
    group, and PCP+DP is refused, so the staging extensions below only apply to a
    PCP-sharded table built by the HCU subclass.
    """
    table = engram.embed_tokens
    return (
        _pcp_size() > 1
        and table.dp_size == 1
        and bool(getattr(table, _CLASS_MARKER, False))
    )


def _install_pcp_engram_methods(engram_class) -> None:
    """Install PCP token staging around the community Engram implementation."""
    if getattr(engram_class, _ENGRAM_METHOD_MARKER, False):
        return

    original_init_staging = require_callable(engram_class, "_init_staging", TARGETS[1])
    original_prepare = require_callable(engram_class, "prepare_embeddings", TARGETS[2])
    original_embed = require_callable(engram_class, "embed", TARGETS[3])

    @functools.wraps(original_init_staging)
    def hcu_init_staging(self, max_tokens, head_dim):
        # v0.32 core already multiplies by `embed_tokens.dp_size` here
        # (engram.py:1593-1594), so only the PCP factor is added; multiplying
        # both would double-count.
        if not _pcp_sharded(self):
            return original_init_staging(self, max_tokens, head_dim)
        return original_init_staging(self, max_tokens * _pcp_size(), head_dim)

    @functools.wraps(original_prepare)
    def hcu_prepare_embeddings(self, hash_ids):
        if not _pcp_sharded(self):
            return original_prepare(self, hash_ids)
        gathered, slot = _gather_padded_pcp_hash_ids(hash_ids)
        # The core decides inline-vs-prefetch and asserts the staging size, so
        # hand it the gathered rows instead of re-implementing the lookup here.
        original_prepare(self, gathered)
        self._hcu_pcp_local_tokens = int(hash_ids.shape[0])
        self._hcu_pcp_slot = slot

    @functools.wraps(original_embed)
    def hcu_embed(self, hash_ids):
        if not _pcp_sharded(self):
            return original_embed(self, hash_ids)
        group = _pcp_group()
        assert group is not None
        table = self.embed_tokens
        # No silent fallback on the PCP path: staging geometry must come from this
        # step's prepare_embeddings.  Core call order guarantees it -- `engram_hashes`
        # is produced in amd/model.py:597-626 and the very same block runs
        # prepare_embeddings for every owned layer at :628-633, while the only
        # Engram.forward -> embed() call site (amd/model.py:341-348 ->
        # common/engram.py:1688) requires `engram_hashes is not None`.  The V4.1
        # draft model has no Engram at all (amd/dspark.py: no `engram`).
        missing = [
            name
            for name in ("_hcu_pcp_local_tokens", "_hcu_pcp_slot")
            if not hasattr(self, name)
        ]
        if missing:
            raise RuntimeError(
                "HCU Engram PCP staging state is missing (%s): embed() ran without "
                "prepare_embeddings() this step. Core call order makes that "
                "impossible, so this is an adapter or model-side programming "
                "error, not a case to guess a token count for." % ", ".join(missing)
            )
        local_tokens = int(self._hcu_pcp_local_tokens)
        slot = int(self._hcu_pcp_slot)
        staged = self._ready_rows(slot * _pcp_size())
        rows = group.all_gather(staged, dim=1)
        start = int(group.rank_in_group) * slot
        rows = rows[start : start + local_tokens]
        if table.tp_size > 1:
            if self.use_sequence_parallel:
                raise RuntimeError(_REFUSE_PCP_WITH_SP)
            from vllm.distributed import tensor_model_parallel_all_gather

            rows = tensor_model_parallel_all_gather(rows, dim=1)
        return rows[:, : table.n_hash_cols]

    setattr(engram_class, "_vllm_hcu_original_init_staging", original_init_staging)
    setattr(engram_class, "_vllm_hcu_original_prepare_embeddings", original_prepare)
    setattr(engram_class, "_vllm_hcu_original_embed", original_embed)
    setattr(engram_class, "_init_staging", hcu_init_staging)
    setattr(engram_class, "prepare_embeddings", hcu_prepare_embeddings)
    setattr(engram_class, "embed", hcu_embed)
    setattr(engram_class, _ENGRAM_METHOD_MARKER, True)


def _host_register(pointer: int, num_bytes: int) -> None:
    """Register an existing host allocation as device-visible pinned memory."""
    result = torch.cuda.cudart().cudaHostRegister(pointer, num_bytes, 0)
    value = int(getattr(result, "value", result))
    if value != 0:
        raise RuntimeError(
            f"cudaHostRegister failed: code={value}, pointer={pointer:#x}, "
            f"bytes={num_bytes}, page_offset={pointer % mmap.PAGESIZE}, "
            f"result={result}"
        )


def _host_unregister(pointer: int) -> None:
    try:
        result = torch.cuda.cudart().cudaHostUnregister(pointer)
        value = getattr(result, "value", result)
        if int(value) != 0:
            logger.warning("HCU Engram cudaHostUnregister failed: %s", result)
    except Exception as exc:  # best-effort teardown
        logger.warning("HCU Engram cudaHostUnregister raised: %r", exc)


def _register_host_tensor(tensor: torch.Tensor, owner: object) -> torch.Tensor:
    """Pin an anonymous host tensor for zero-copy lookup.

    Large ``hipHostMalloc``/``pin_memory=True`` requests are unreliable on HCU, so
    anonymous host pages are allocated first and then registered.  The finalizer is
    anchored to *owner* (the embedding module) because ``nn.Parameter`` shares
    storage with, but does not keep alive, the tensor returned here.
    """
    pointer = tensor.data_ptr()
    num_bytes = tensor.numel() * tensor.element_size()
    _host_register(pointer, num_bytes)
    try:
        if not tensor.is_pinned():
            raise RuntimeError("CUDA did not recognize the registered Engram storage")
    except BaseException:
        # Roll back so a failed check cannot leave a stale registration.
        _host_unregister(pointer)
        raise
    finalizer = weakref.finalize(owner, _host_unregister, pointer)
    finalizer.atexit = False  # type: ignore[misc]
    return tensor


def _allocate_registered_host_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, owner: object
) -> torch.Tensor:
    """Allocate anonymous pages before registering them with the HCU driver.

    The mmap object must stay on the owner: ``torch.frombuffer`` does not make its
    lifetime part of the storage contract, and the registered pages must outlive
    the lookup.
    """
    numel = 1
    for dim in shape:
        numel *= dim
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    mapping = mmap.mmap(
        -1,
        nbytes,
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    raw = torch.frombuffer(mapping, dtype=dtype, count=numel).view(shape)
    try:
        registered = _register_host_tensor(raw, owner)
        mappings = getattr(owner, "_hcu_engram_mappings", None)
        if mappings is None:
            mappings = []
            setattr(owner, "_hcu_engram_mappings", mappings)
        mappings.append(mapping)
        return registered
    except BaseException:
        mapping.close()
        raise


def _make_offload_embedding_class(base: type) -> type:
    if getattr(base, _CLASS_MARKER, False):
        return base

    class HcuDeepseekV41ParallelEngramEmbedding(base):
        """Community head shards with HCU PCP-aware host storage."""

        _vllm_hcu_dsv41_engram_offload_storage = True

        def __init__(self, *args, **kwargs):
            # Refuse before touching the allocator: the core's own keyword values
            # decide this, and nothing here re-reads the core call site.
            _require_supported_config(_offload_flags(base, args, kwargs))
            super().__init__(*args, **kwargs)
            offloaded = (
                self.weight.numel() * self.weight.element_size()
                + self.weight_scale_inv.numel()
                * self.weight_scale_inv.element_size()
            )
            logger.info(
                "HCU DeepSeek V4.1 Engram CPU offload active: %.2f GiB per rank "
                "(weight=%s, weight_scale_inv=%s, shards=%d as %s, heads/rank=%d)",
                offloaded / 1024**3,
                tuple(self.weight.shape),
                tuple(self.weight_scale_inv.shape),
                self.tp_size * (_pcp_size() if self.dp_size == 1 else self.dp_size),
                _shard_label(self),
                self.part_n_hash_cols,
            )

        def _get_shard_info(self):
            # v0.32 core owns the DP branch (engram.py:1031-1033); only a
            # PCP-sharded table with no DP group is taken over here.
            pcp_size = _pcp_size()
            if self.dp_size > 1:
                if pcp_size > 1:
                    raise RuntimeError(_REFUSE_PCP_PLUS_DP)
                return super()._get_shard_info()
            if pcp_size == 1:
                return super()._get_shard_info()
            from vllm.distributed import get_tensor_model_parallel_rank

            return _pcp_shard_info(
                self.tp_size,
                get_tensor_model_parallel_rank(),
                pcp_size,
                _pcp_rank(),
            )

        def _allocate_weights(self):
            # Device-resident, DP-shared and THP-backed tables keep the core
            # implementation; only the plain host-offload case is ours.
            if not self.cpu_offload or self.dp_shared_memory or self.use_thp:
                return super()._allocate_weights()
            requested = (
                self.part_num_embeddings
                * (self.dim + self.dim // self.block_size)
                / 1024**3
            )
            try:
                weight = _allocate_registered_host_tensor(
                    (self.part_num_embeddings, self.dim),
                    torch.float8_e4m3fn,
                    self,
                )
                scales = _allocate_registered_host_tensor(
                    (self.part_num_embeddings, self.dim // self.block_size),
                    torch.uint8,
                    self,
                )
            except (RuntimeError, TypeError) as exc:
                raise RuntimeError(
                    "HCU DeepSeek V4.1 Engram CPU offload failed to allocate "
                    f"{requested:.2f} GiB of registered host memory for this rank"
                ) from exc
            return weight, scales

    setattr(HcuDeepseekV41ParallelEngramEmbedding, _CLASS_MARKER, True)
    return HcuDeepseekV41ParallelEngramEmbedding


def _require_substitutable(target: ModuleType, base: type) -> None:
    """The core builds this class by module global, so verify a drop-in swap."""
    if getattr(target, "ParallelEngramEmbedding", None) is not base:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is not the module-global "
            "class the core constructs"
        )
    try:
        parameters = inspect.signature(base.__init__).parameters
    except (TypeError, ValueError) as exc:
        raise PatchCompatibilityError(
            f"cannot inspect required HCU patch target {TARGETS[0]}: {exc}"
        ) from exc
    missing = [name for name in _REQUIRED_KEYWORDS if name not in parameters]
    if missing:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} no longer accepts "
            f"{missing}; the name replacement would drop the core's storage "
            "keywords"
        )
    for method in ("_get_shard_info", "_allocate_weights", "lookup", "_storage"):
        require_callable(base, method, f"{TARGETS[0]}.{method}")


def _host_register_available() -> bool:
    if not hasattr(torch.cuda, "cudart"):
        return False
    try:
        return callable(torch.cuda.cudart().cudaHostRegister)
    except Exception as exc:  # pragma: no cover - platform probe
        logger.warning("%s: torch.cuda.cudart() probe raised %r", PATCH_ID, exc)
        return False


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    engram_class = require_class(target, "Engram", f"{TARGET_MODULE}.Engram")
    base = require_class(
        target, "ParallelEngramEmbedding", f"{TARGET_MODULE}.ParallelEngramEmbedding"
    )
    if getattr(target, _NAME_MARKER, False):
        if not getattr(target.ParallelEngramEmbedding, _CLASS_MARKER, False):
            raise PatchCompatibilityError(
                f"required HCU patch marker for {PATCH_ID} is stale"
            )
        return False
    if getattr(base, _CLASS_MARKER, False):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} is already an HCU subclass "
            "but the module global was not replaced"
        )
    _require_substitutable(target, base)
    if not _host_register_available():
        logger.error(
            "%s not installed: this HCU build exposes no "
            "torch.cuda.cudart().cudaHostRegister, so the Engram table cannot be "
            "placed in registered host memory. The core `pin_memory=True` path "
            "will be used instead.",
            PATCH_ID,
        )
        return False

    setattr(target, "_vllm_hcu_original_engram_embedding_class", base)
    # The core's `_create_embedding` looks this name up in the module namespace,
    # so replacing the global makes the core build the HCU subclass with its own
    # argument list -- nothing here has to keep up with core signature changes.
    setattr(target, "ParallelEngramEmbedding", _make_offload_embedding_class(base))
    setattr(target, _NAME_MARKER, True)
    _install_pcp_engram_methods(engram_class)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "PATCH_ID",
    "TARGETS",
    "TARGET_MODULE",
    "apply",
    "apply_to_module",
]
