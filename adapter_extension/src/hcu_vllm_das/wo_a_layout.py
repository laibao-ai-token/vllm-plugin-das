# SPDX-License-Identifier: Apache-2.0
"""Write native BF16 WO_A directly in the following INT8 consumer's layout."""
from __future__ import annotations

import hashlib
import inspect
import os
from importlib.metadata import PackageNotFoundError, version
from functools import lru_cache

import torch
from vllm.models.deepseek_v41.amd import rocm
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as native

_O_PROJ = rocm.DeepseekV41ROCMAiterMLAAttention._o_proj
_CONSUMER = rocm.DeepseekV41ROCMAiterMLAAttention._wo_b_after_wo_a
_EINSUM = native.rocm_inv_rope_einsum
_CACHE = inspect.unwrap(native._get_cached_wo_a_bf16)
_NN_WRAPPER_HASH = '8781211398faf2f6e5b48debe330a148e4c6fc9e9ea7267635423f6ac10a9b5e'
_HASHES = (
    '050ac04cf793582bac2c53ed88d0d8d25e102474962e77264530ee68ecda6825',
    'adbefe2f49bd0900d4c87bc015f605399260bd769bcd76a5bd592d1a19ae5f61',
    '9b4fe284921cedf43cb649c736a80f5d4612d70f24d5368a023b1545a4af55c3',
    '1c9dd6d7149af49550ce9ce5290cfe723de55904dae3697bed821adf997455b8',
)


@lru_cache(maxsize=2)
def _cache_contract(function):
    # The worker's existing NN-layout fix can install after general plugin
    # discovery. Accept that exact wrapper without freezing import order.
    if function is _CACHE:
        return True, False
    try:
        verified = (getattr(function, '__wrapped__', None) is _CACHE
                    and getattr(function, '_vllm_hcu_wo_a_nn_weight_layout_wrapper', False)
                    and hashlib.sha256(inspect.getsource(function.__code__).encode()).hexdigest()
                    == _NN_WRAPPER_HASH)
        return verified, verified
    except (OSError, TypeError, AttributeError):
        return False, False


def _source_matches():
    try:
        hashes = tuple(hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
                       for fn in (_O_PROJ, _CONSUMER, _EINSUM, _CACHE))
        return (hashes == _HASHES
                and _cache_contract(native._get_cached_wo_a_bf16)[0]
                and version('torch') == '2.13.0+dtk2610.2611091606.ge8aa84'
                and not any(os.environ.get(name) for name in (
                    'ROCBLAS_TENSILE_LIBPATH', 'ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH',
                    'HIPBLASLT_TENSILE_LIBPATH', 'HIPBLASLT_TUNING_OVERRIDE_FILE')))
    except (OSError, TypeError, AttributeError, PackageNotFoundError):
        return False


_SOURCE_MATCHES = _source_matches()


def configured(config):
    from .quantization.ct_w4a8 import HcuDSV41CTW4A8Config
    if not _SOURCE_MATCHES or not isinstance(config.quant_config, HcuDSV41CTW4A8Config):
        return False
    parallel = config.parallel_config
    model = config.model_config.hf_config
    return (_cache_contract(native._get_cached_wo_a_bf16)[0]
            and model.model_type == 'deepseek_v41' and model.hidden_size == 5120
            and parallel.pipeline_parallel_size == 1
            and parallel.tensor_parallel_size == parallel.prefill_context_parallel_size == 1
            and parallel.data_parallel_size == 8 and parallel.enable_expert_parallel)


def supported(owner, attn_out):
    # Out= BMM is an inference-only route. Autocast callers keep the native
    # einsum's own casting policy, even when the supplied tensors are BF16.
    if (not _SOURCE_MATCHES or not isinstance(attn_out, torch.Tensor)
            or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')
            or not attn_out.is_cuda or attn_out.dtype != torch.bfloat16
            or attn_out.ndim != 3 or attn_out.shape[0] not in (8, 48, 64)
            or attn_out.shape[1] not in (64, 128) or attn_out.shape[2] != 512
            or not attn_out.is_contiguous()
            or (owner.n_local_heads, owner.n_local_groups, owner.o_lora_rank) != (64, 8, 1024)
            or owner._wo_b_scale is not None):
        return False
    if (rocm.DeepseekV41ROCMAiterMLAAttention._o_proj is not _O_PROJ
            or rocm.DeepseekV41ROCMAiterMLAAttention._wo_b_after_wo_a is not _CONSUMER
            or rocm.rocm_inv_rope_einsum is not _EINSUM
            or native.rocm_inv_rope_einsum is not _EINSUM
            or getattr(owner._wo_b_after_wo_a, '__func__', None) is not _CONSUMER):
        return False
    cache_valid, cache_nn = _cache_contract(native._get_cached_wo_a_bf16)
    if not cache_valid:
        return False
    weight = owner.wo_a.weight
    cached = getattr(owner.wo_a, '_dsv4_wo_a_bf16', None)
    allowed_weights = ((8192, 4096), (4096, 8192)) if cache_nn else ((8192, 4096),)
    if (not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16
            or tuple(weight.shape) not in allowed_weights or not weight.is_contiguous()
            or weight.device != attn_out.device):
        return False
    if cached is not None and (not isinstance(cached, torch.Tensor)
            or cached.shape != (8, 1024, 4096) or cached.dtype != torch.bfloat16
            or cached.device != attn_out.device or not cached.is_contiguous()):
        return False
    properties = torch.cuda.get_device_properties(attn_out.device)
    return (getattr(properties, 'gcnArchName', '').split(':')[0] == 'gfx938'
            and properties.multi_processor_count == 64)


def project(owner, attn_out):
    if not supported(owner, attn_out):
        raise ValueError('WO_A token-major output is outside the verified W4A8 contract')
    rows = attn_out.shape[0]
    inputs = attn_out[:, :64, :].reshape(rows, 8, 4096)
    weight = native._get_cached_wo_a_bf16(owner.wo_a, 8, 1024, 4096)
    result = torch.empty((rows, 8, 1024), dtype=attn_out.dtype, device=attn_out.device)
    torch.bmm(inputs.transpose(0, 1), weight.transpose(1, 2), out=result.transpose(0, 1))
    # This is a view. Native einsum produces group-major storage and flatten
    # must copy it; the following native INT8 consumer now receives the same
    # byte-verified BF16 values in contiguous token-major storage.
    return result.flatten(1)
