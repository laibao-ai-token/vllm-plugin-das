# SPDX-License-Identifier: Apache-2.0
"""Instance-owned admission for exact W4A8 greedy draft sampling."""
from functools import lru_cache
import hashlib
import inspect
import logging
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _native_contract():
    from vllm.models.deepseek_v41.amd.dspark import DSparkDeepseekV4ForCausalLM
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    # Unknown native loops retain their own sampling, without partial execution.
    expected = ((DSparkDeepseekV4ForCausalLM,
                 '3ddc30bf30261b46d83b33588fc31bdbb355d1404bc9966c08637283c2270889'),
                (DSparkSpeculator,
                 '3c22193d18f0d2f5f5dcd4ec907f52e10710a5b39b084dfa03eef254d3c174eb'))
    valid = all(hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() == sha
                for cls, sha in expected)
    return valid, DSparkDeepseekV4ForCausalLM, DSparkSpeculator


def supported(model, spec, num_reqs, hidden):
    if (not getattr(model, '_hcu_markov_pointwise', False)
            or getattr(spec, 'model', None) is not model
            or type(num_reqs) is not int or not 1 <= num_reqs <= 32
            or getattr(spec, 'num_speculative_steps', None) != 5
            or getattr(spec, 'draft_logits', False) is not None
            or getattr(spec, 'draft_watermarker', False) is not None
            or getattr(spec, '_d2t_scatter_index', False) is not None
            or getattr(spec, '_draft_topk', False) is not None
            or getattr(spec, 'use_confidence_head', True) is not False
            or getattr(spec, 'acceptance_estimator', False) is not None
            or hasattr(model, 'apply_output_logit_transform')
            or torch.is_grad_enabled() or torch.is_autocast_enabled('cuda')):
        return False
    head = getattr(getattr(model, 'model', None), 'markov_head', None)
    if head is None or type(head.markov_w1) is not torch.nn.Embedding:
        return False
    embedding = head.markov_w1
    w1, w2 = embedding.weight, head.markov_w2.weight
    if (embedding.max_norm is not None or embedding._forward_hooks
            or embedding._forward_pre_hooks
            or w1.shape != (129280, 256) or w2.shape != w1.shape
            or w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16
            or not (w1.is_cuda and w1.is_contiguous() and w2.is_contiguous())
            or w2.device != w1.device or hidden.device != w1.device
            or hidden.dtype != torch.bfloat16 or hidden.ndim != 2
            or hidden.shape[1] != 5120):
        return False
    processor = model.logits_processor
    if (head.markov_w2.quant_method.__class__.__name__ != 'UnquantizedEmbeddingMethod'
            or head.markov_w2.tp_size != 1
            or processor.head_dtype not in (None, torch.bfloat16)
            or processor.scale != 1.0 or processor.soft_cap is not None
            or processor.logits_as_input or processor.org_vocab_size != 129280):
        return False
    tokens = spec.draft_tokens
    if (tokens.ndim != 2 or tokens.shape[0] < num_reqs or tokens.shape[1] != 5
            or tokens.dtype != torch.int64 or tokens.stride(1) != 1
            or tokens.device != w1.device):
        return False
    valid, native_model, native_spec = _native_contract()
    if not valid:
        return False
    for name in ('markov_embed', 'markov_bias', 'map_draft_to_target'):
        if getattr(getattr(model, name), '__func__', None) is not getattr(native_model, name):
            return False
    if getattr(spec._sample_logits, '__func__', None) is not native_spec._sample_logits:
        return False
    props = torch.cuda.get_device_properties(w1.device)
    return props.gcnArchName.split(':')[0] == 'gfx938' and props.multi_processor_count == 64


@lru_cache(maxsize=1)
def _log_admission():
    logging.getLogger('vllm.' + __name__).info(
        'HCU W4A8 exact greedy Markov pointwise admitted: V129280 R256 DSpark5; native bias projection retained')


def sample_sequential(model, spec, num_reqs, head_hidden):
    if not supported(model, spec, num_reqs, head_hidden):
        return False
    from .dspark_greedy_pointwise import greedy
    n = num_reqs * 5
    sample_hidden = head_hidden[spec.sample_indices[:n]]
    base = model.compute_draft_logits(sample_hidden)
    # Typed native model/heads establish these before any custom kernel writes.
    # A changed provider cannot silently feed another dtype or padded layout.
    if base.dtype != torch.bfloat16 or base.shape != (n, 129280) or not base.is_contiguous():
        raise RuntimeError('Verified W4A8 Markov base-logits contract changed')
    anchors = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    head = model.model.markov_head
    greedy(head.markov_w1.weight, head.markov_w2.weight,
           base.view(num_reqs, 5, 129280), anchors,
           bias_fn=model.markov_bias, embed_fn=model.markov_embed,
           out=spec.draft_tokens[:num_reqs])
    _log_admission()
    return True
