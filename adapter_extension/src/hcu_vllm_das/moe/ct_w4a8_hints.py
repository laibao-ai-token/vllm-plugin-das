# SPDX-License-Identifier: Apache-2.0
"""Host-only scheduling hint for masked W4A8 experts."""


def expected_m(local_tokens: int, dispatchers: int, topk: int,
               global_experts: int, capacity: int) -> int:
    if (local_tokens < 0 or dispatchers <= 0 or topk <= 0
            or global_experts <= 0 or capacity <= 0):
        raise ValueError('Invalid low-latency MoE scheduling dimensions')
    # dyldu/vllm-plugin-das#8 uses a one-row margin, then rounds up to 16.
    average = local_tokens * dispatchers * topk // global_experts + 1
    rounded = ((average + 15) // 16) * 16
    return min(capacity, rounded)
