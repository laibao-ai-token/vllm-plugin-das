# SPDX-License-Identifier: Apache-2.0
"""Private AG-RS construction for the HCU V41 draft model.

The native loader and the compressed-tensors scale-alias load view belong to the
adapter (`hcu_adapter.dsv41_draft_weight_alias`): they are the naming bridge the
distinct channel-FP8 route creates, which is an adapter route this plugin does
not own. What stays here is the instance-owned AG-RS construction config, which
depends on this plugin's own communicator, so the class overrides `__init__`
only and inherits enumeration/loading unchanged.
"""
from __future__ import annotations

from hcu_adapter.dsv41_draft_weight_alias import HcuDSV41DraftWeightAliasModel


class HcuDSparkDeepseekV41ForCausalLM(HcuDSV41DraftWeightAliasModel):
    """Keep native draft construction/loading; own CT DP communication policy."""

    def __init__(self, *, vllm_config, prefix=""):
        from vllm.config import set_current_vllm_config
        from hcu_vllm_das.dspark_agrs import _owned_quant_kind, draft_config
        from hcu_vllm_das.dspark_backend import native_draft_ll_requested
        from hcu_vllm_das.dspark_post import requested, initialize_owned

        native_ll = native_draft_ll_requested(vllm_config)
        constructor = super().__init__
        if requested(vllm_config):
            constructor = lambda **kwargs: initialize_owned(self, **kwargs)

        # CT W4A8 and Channel-FP8 drafts both take a private AG-RS construction
        # config so they do not allocate a second DeepEP/RocSHMEM LL buffer on
        # top of the target's already-resident weights.
        if (
            _owned_quant_kind(vllm_config) is not None
            and vllm_config.parallel_config.data_parallel_size > 1
            and not native_ll
        ):
            local = draft_config(vllm_config)
            # Native draft decoder construction reads current_config as well as
            # its explicit argument. Keep both references on the same private
            # copy and retain native compiler live tables for layer discovery.
            with set_current_vllm_config(local):
                constructor(vllm_config=local, prefix=prefix)
        else:
            constructor(vllm_config=vllm_config, prefix=prefix)
        if native_ll:
            from hcu_vllm_das.dspark_ll import install_model_entry_reset
            install_model_entry_reset(self, vllm_config, 128)
