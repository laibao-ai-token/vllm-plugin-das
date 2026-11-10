# HCU vLLM DAS optimization plugin

Independent Flash public-registry publication and health contract:
[v0.33 provider ownership](../docs/validation/vllm033-upgrade.md#ownership-and-changes). Its
activation dependencies are independent of the optional RMS provider.

`hcu-vllm-das-plugin` is the optimization-only extension for official vLLM
0.33 and `adapter-das`. The adapter owns the HCU platform and runtime
lifecycle; this wheel contributes optional, independently removable operator
optimizations through the adapter extension API.

Version `0.3.0.dev0` targets the `v0.33.0` branch. Migration status and current
evidence are recorded in [v0.33 validation](../docs/validation/vllm033-upgrade.md);
the v0.32 documents below are historical evidence and do not certify v0.33.
The operator extension currently advertises only the default-OFF LightOp
ordinary and fused-add `RMSNorm` candidate. INT8 model selection, weight
conversion, quantization and kernel construction remain official vLLM owners.

It also provides an explicit HCU FlashAttention backend selected by
`--attention-backend FLASH_ATTN`. The backend keeps vLLM's official metadata,
scheduler and allocation. Its public `AttentionImpl` owns the HCU provider call
and dense HND/NHD cache views, routes BF16 writes to the native vLLM writer,
and retains a Plugin E5M2 writer and descale adapter. The kernel page is 64
tokens. It neither copies nor replaces the official Flash forward method and
does not require private HCU hooks in the Core.

## E2M1 packer lowering (opt-in, default stock)

`das_hcu_fp4_packer` rebinds Core's `_fp32x2_to_fp4x2` E2M1 packer at startup.
The audited upstream body emits PTX `cvt.rn.satfinite.e2m1x2.f32` with the
`=r,f,f` constraint, which DTK clang cannot lower (029 `BLOCKED_TOOLCHAIN`).
The replacement is Plugin-owned arithmetic: `min(|x|, 6)` saturation, seven
thresholds, tie-to-even on the magnitude code, and the IEEE sign bit (so `-0`
stays `0x8`). The first argument keeps the low nibble.

Activation requires real HCU `gfx938`, vLLM `0.33.*`, an exact AST match with
the audited definition plus identical aliases in six namespaces, an unhashed
and uncompiled caller set, no foreign compile factor, and no existing runtime
record. Stock (`HCU_VLLM_FP4_PACKER_IMPL=stock`, the default) or
`HCU_DISABLE_OPS=vllm_fp4_packer` leaves every original object in place; a
policy change needs a fresh process because JIT and Graph caches may already
refer to the published kernel. A failure after publication is commit-fatal and
is never presented as a rollback.

Upstreaming position: the arithmetic body is a workaround for a toolchain gap,
so the intended end state is deleting it once DTK lowers the PTX instruction or
Core ships a lowering it can compile. This module therefore declares no
upstream-format contribution; it carries no cache format, scale recipe or
roofline claim of its own.

Evidence level: **spec-consistent, hardware-unverified**. CPU/mock contracts
only (83 tests): explicit E2M1 code literals, FP32 tie neighbours, saturation
including `2**28`/`FLT_MAX`, NaN/Inf policy, nibble order, the audit gate, and
policy/rollback refusals. The Triton kernel was neither compiled nor executed;
native compilation, kernel hit, Graph replay, accuracy and performance remain
separate gates.

vLLM's native `SiluAndMul` remains the selected implementation. Warmed
compiled A/B measurements on `gfx936` did not justify replacing the upstream
operator, so the plugin neither registers nor ships a `SiluAndMul` override.

No model, topology, FP8-KV, or broad performance support is implied by these
operator-level capabilities.

## DSV4.1 Channel-FP8 configuration routing

The independent `das_hcu_dsv41` Adapter extension advertises only
`dsv41-channel-fp8-routing` and registers the distinct public quantization name
`hcu_dsv41_channel_fp8`. It does not replace the official `fp8`,
`deepseek_v4_fp8`, or `compressed-tensors` entries or their methods, and does
not depend on the RMS/Flash extensions or vendor optimizations for discovery.

Automatic selection requires the HCU platform, a V4.1 model configuration,
`quant_method=fp8` (or `deepseek_v4_fp8`), `is_per_channel=true`, no weight block
size, dynamic activations, and a compatible user quantization choice. Other
models and schemes preserve their official selection. The configuration reuses
the official V4.1 FP8 policy, checks native exclusions before delegating to the
official CT channel-weight/token-activation scheme, and leaves BF16 `wo_a`
unquantized. At normal serialized startup, an existing foreign owner of the
new name is rejected; partial registration failure refuses further activation
in that process. The Core registry has no atomic reservation API: this extension
does not guarantee inter-owner atomicity against concurrent external writes or
roll back arbitrary Core registrations. Its health check detects a later lost
owner without repairing it.

This increment is configuration/routing infrastructure. It does **not** certify
checkpoint or target/draft scale loading, HCU kernel support, model inference,
W4A8, PCP, DSpark, Graph, accuracy or performance. Those are separate migration
gates. Requiring `dsv41-channel-fp8-routing` at startup proves activation only,
not those execution properties. See the [routing contract and evidence](../docs/validation/dsv41-channel-fp8-routing.md).

For the selected Channel-FP8 W8A8 Linear scheme, a Plugin-owned CT LinearMethod
returns correctly shaped empty outputs before local quantization/GEMM. It
retains CT weight creation/finalization and native nonempty execution, registers
through the public weight-loader-v2 API, and leaves the surrounding Linear
forward and collectives intact. Native `QuantizedActivation` is reused.
The pinned v0.33 V41 target already resolves direct/inverse scale spellings;
small CPU source replay verifies ordinary and fused projection loading without
adding parameter aliases or changing that loader. Hardware admission,
quantization and GEMM are doubled in that replay. DSpark draft scale aliases,
actual checkpoint loading, distributed execution and GPU correctness remain
unverified. See [target loading and empty-row validation](../docs/validation/dsv41-channel-fp8-loading.md).

## Compatibility

- vLLM release line: `>=0.33.0,<0.34.0`; prerelease, postrelease,
  development, patch, and local version suffixes are accepted when the
  required operator APIs and LightOp capabilities are present. FLASH_ATTN uses
  the official public backend registry and AttentionImpl contract.
- Adapter distribution: `hcu-adapter>=0.5.22`, extension API v1 from
  `hcu_adapter.extensions`.
- Runtime: a prepared HCU PyTorch/DTK/LightOp/AITER stack. LightOp and AITER
  are intentionally not fetched by pip because production deployments must
  use ABI-matched local builds.
- The plugin does not own `vllm.platform_plugins` or `vllm.general_plugins`.

The repository root remains the vLLM 0.28.1 development reference. Build only
this directory for vLLM 0.33:

```bash
python3 -m pip wheel --no-deps --wheel-dir ./dist ./adapter_extension
```

## Production installation

Start with a fresh virtual environment or image containing the matching HCU
PyTorch, DTK, LightOp, and AITER builds. Install audited local wheels without
dependency resolution, in this order:

```bash
python3 -m pip install --no-deps /wheels/vllm-0.33.0+*.whl
python3 -m pip install --no-deps /wheels/sgl_kernel-*.whl
python3 -m pip install --no-deps /wheels/hcu_adapter-*.whl
python3 -m pip install --no-deps /wheels/hcu_vllm_das_plugin-*.whl
```

If the shell exports `PYTHONPATH`, clear it or invoke Python with `-E` during
verification so an older system vLLM cannot precede the environment's wheel:

```bash
VLLM_PLUGINS=hcu_adapter python3 -E adapter_extension/tools/verify_installed.py
```

The verifier fails closed unless all three distributions have non-empty wheel
records, their imported modules are owned by those records, the adapter is the
sole expected vLLM entry-point owner, the RMS extension is active, and
vLLM's public OOT registry selects `hcu_vllm_das.rms_norm.HcuRMSNorm`.
Schema4 additionally verifies that the seven official INT8 oracle symbols
remain identical across activation, the module is owned by the Core wheel,
and explicit Triton selection resolves to an official expert.

## Static EPLB calibration and dummy loading

Static EPLB calibration excludes the startup memory-profile forward, whose
attention metadata is absent even though Core marks its dummy tokens as real
for MoE memory accounting. Other padded warmup rows are excluded by the
published unpadded-token counter. Real client calibration requests contribute
to the cumulative counters and periodic map proposals.

With `--load-format dummy`, deployment binds and verifies the offline layout
in the model post-load hook. It preserves packed dummy tensors and publishes
routing without rebinding checkpoint weight loaders. Normal checkpoint loading
still binds replica fan-out before quantization. Static EPLB uses the PCP
decoder replay gate. Select `HCU_VLLM_FP4_INDEXER_IMPL=stock` for PCP static
EPLB. The separate PCP1 FP4 model rejects static EPLB.

CPU regressions run through `tests/vllm033/run_v41_static_eplb_cpu.sh` and cover
profile/warmup exclusion, real requests, and dummy post-load routing. GPU
router execution, full-model output and performance require device validation.

## Eager PCP DSpark with static EPLB

The owned V41 PCP target accepts DSpark with an eager V41 draft, TP1 and the
existing PP1/DP1/DCP1 topology. Static placement applies to target backbone MoE
layers. The draft retains its native expert layout. Keep global `--enable-eplb`
off and supply the placement through `additional_config.hcu_static_eplb`.

The owned compressor metadata retains the complete circular-buffer window
across PCP ranks. This supplies earlier CR2 predecessor states after a rejected
draft restarts verification behind the previous end position. Native metadata
building, compression and ring-store kernels consume the owned history. Private
constructor namespaces select the owned cache and metadata backend.

`tests/vllm033/test_pcp_dspark.py` covers every N5 rejection position, selected
N13 positions, multiple requests, changing decode owners, and real two-process
Gloo gathers with an empty rank. These CPU tests replace device compression and
ring-store kernels with checking doubles. Source replay and CPU tests do not
establish full-model accuracy or device performance.

The Channel-FP8 device comparison uses
`tests/vllm033/launch_pcp_dspark_gpu.sh` and
`tests/vllm033/validate_pcp_dspark_gpu.py`. It runs static EPLB with DSpark off
and on, reuses the same three prompts, and generates 32 tokens per prompt twice.
It saves commands, responses, metrics, server logs and token comparisons under
`HCU_VALIDATION_OUTPUT` (by default a unique directory under `test-results`). The comparison
explicitly disables SWA replay and Layer Split and selects stock Adapter FP4
indexer/packer and mHC, allowing the Plugin's mHC registration.

Use an existing runtime with the matching Core, Adapter, Plugin and HCU vendor
libraries. Configure the source workspace, model and calibrated layout before
running the device checks. The source workspace contains `vllm/` and
`adapter-das/`; installed Core and Adapter packages can also supply those imports.

```bash
export HCU_VALIDATION_ROOT=/path/to/source-workspace
export HCU_VALIDATION_MODEL=/path/to/Channel-FP8-model
export HCU_VALIDATION_EXPERT_MAP=/path/to/expert_map.json
export HCU_VALIDATION_PYTHON=/path/to/runtime/bin/python
export HCU_VALIDATION_OUTPUT=/path/to/new-results-directory
# Optional overrides include HCU_VALIDATION_PORT, VLLM_HOST_IP,
# VLLM_TARGET_ROOT and HCU_ADAPTER_SOURCE.
bash tests/vllm033/launch_pcp_dspark_gpu.sh --dry-run
bash tests/vllm033/launch_pcp_dspark_gpu.sh
```

The HumanEval wrapper uses the included `evaluate_humaneval.py` HTTP client.
Install EvalScope 1.14.1 and datasets 4.8.4 in its interpreter, then provide the
local dataset directory containing
`openai_humaneval/test-00000-of-00001.parquet`. It verifies all 164 task IDs
before launching the service and runs two 16-request warmup rounds before scoring.

```bash
export HCU_HUMANEVAL_DATA=/path/to/local-humaneval
export HCU_HUMANEVAL_PYTHON=/path/to/evalscope/bin/python
export VALIDATION_SCRIPT="$PWD/tests/vllm033/run_pcp_dspark_humaneval.py"
bash tests/vllm033/launch_pcp_dspark_gpu.sh --dry-run
bash tests/vllm033/launch_pcp_dspark_gpu.sh
```

The 2026-11-10 nmz22 run completed both services and both generation rounds.
Each phase produced 192 output tokens. DSpark reported 76 drafts, 380 draft
tokens and 114 accepted tokens. Two of six paired completions matched exactly.
All three baseline prompts also differed between the baseline's own repeats.
Consequently device startup and continuous generation were exercised, while
output equivalence remains unresolved. The validation services were stopped.

The full HumanEval run on 2026-11-10 used the existing
`accuracy/humaneval.py` EvalScope 1.14.1 client, concurrency 16, temperature 0,
thinking disabled, and at most 2048 output tokens. After two 16-request warmup
rounds it completed 164 unique tasks with static EPLB32 and DSpark5. The result
was 156/164, Pass@1 95.12195%, with zero API errors and 164 normal stop reasons.
Failed task IDs were 32, 67, 97, 103, 115, 132, 145 and 163. Commands, data
fingerprint, generated answers, code-test reviews and reports are retained under
`logs/eplb_dspark_20261110/humaneval_20261110_215343_40039` in the surrounding
project. `summary.json` and `failed_cases.json` provide the verified counts and
failure details. This is code functional accuracy for the recorded configuration.

## INT8 ownership and unavailable AITER model integration

Use official `--moe-backend auto` or `--moe-backend triton` for INT8 models.
The retired Plugin installer extended/rebound the official oracle enum and
functions; that private integration conflicts with the upstream-first skill
and has been removed. No `aiter-int8-moe` capability is advertised. Calling
its old `install()` entry point explicitly rejects before importing or
modifying Core. An Adapter request requiring that capability must fail.

The official v0.33 INT8 oracle provides Triton/Humming/CPU and has no public
expert registry for adding AITER. The general public quantization-config
registry is not equivalent: a future distinct quantization method would need
its own loader/lifecycle/full-dataflow evidence. It must not overwrite the
official `compressed-tensors` entry or rebind oracle globals.

The standalone INT8 provider, ASM composition and unregistered expert factory
are retained for operator experiments. Their prior native/Graph and full-model
results are frozen historical evidence; they do not make AITER selectable in
the corrected candidate. Preserving official ownership does not certify that
unavailable capability. See [ownership correction](../docs/validation/vllm033-official-ownership.md).

## HCU FlashAttention varlen backend

The supported path is BF16/FP16 dense MHA/GQA with head size 64, 128, 192, or
256, LBHNC (HND) or LBNHC (NHD) cache layout, and 64-token pages. The public
vendor ABI is
`flash_attn.flash_attn_varlen_func`; no classic, custom, or alternate public
attention entry point is registered.

Run with Graph enabled (the vLLM default):

```bash
vllm serve /model/Qwen2-0.5B-Instruct \
  --attention-backend FLASH_ATTN \
  --block-size 64
```

The Plugin keeps LBHNC first in its preference list to preserve the existing
default. Select either layout through vLLM's native setting when a deployment
needs an explicit layout:

```bash
# LBHNC storage, vendor layout="bhsd"
VLLM_KV_CACHE_LAYOUT=HND vllm serve /model/Qwen2-0.5B-Instruct \
  --attention-backend FLASH_ATTN --block-size 64

# LBNHC storage, vendor layout="bshd"
VLLM_KV_CACHE_LAYOUT=NHD vllm serve /model/Qwen2-0.5B-Instruct \
  --attention-backend FLASH_ATTN --block-size 64
```

The Plugin reads vLLM's resolved cache layout rather than inferring it from
tensor geometry. It reinterprets each allocated page as dense
`[K/V, H, N, D]` or `[K/V, N, H, D]` storage. This is required because the
vendor kernel does not consume vLLM's non-dense views produced by splitting
`[..., 2D]`. The official stride-aware HIP cache writer always receives its
NHD contract through a zero-copy alias. The Plugin does not copy the full KV
cache on decode. The vendor's regular prefill path may gather paged KV
internally; rectangular short prefill uses that path by passing a conservative
`max_seqlen_q` upper bound, avoiding the vendor wrapper's incompatible MTP tile
selection.

FA4-only `dynamic_causal`, `mask_mod`, and `aux_tensors` are accepted only
when inactive (`None`). An active feature fails closed instead of silently
changing its mask. FP8 KV, DCP, hybrid caches, and layouts other than LBHNC and
LBNHC remain unvalidated and are not claimed by this release.

## RMSNorm eligibility

Provider admission checks the actual four-positional-argument call ABI before
compile-factor or operator publication. It accepts normal Python signatures and,
only for uninspectable native builtins, a compatible concrete declaration from
generated binding documentation, including overloads and optional additions.
Declarations are parsed without evaluating annotations or defaults. Missing or
incompatible documentation, uninspectable Python callables, and asynchronous or
generator providers are rejected. This extends the existing activation preflight;
its provider checks still run even when the RMS execution flags are OFF.
It neither invokes a provider nor proves kernel, Graph or model correctness.

The LightOp RMSNorm path is selected only when all of the following hold:

- the master and RMSNorm-specific switches are enabled;
- vLLM batch-invariant mode is disabled;
- the operation uses a real weight and no variance-size override;
- the input dtype is BF16 or FP16;
- input width equals the layer hidden size and is at least 64;
- input, weight, and optional residual are contiguous;
- input, weight, and optional residual use the same dtype; and
- an optional residual has the same shape as the input.

All other cases use vLLM's native RMSNorm implementation. `GemmaRMSNorm`,
`RMSNormGated`, quantized fusion, model patches, and performance claims are out
of scope for this release.

## Opt in and rollback

The optimized path defaults to OFF. Enable it explicitly; the master switch
remains the highest-priority rollback and returns RMSNorm to vLLM's complete
implementation without uninstalling the plugin:

```bash
# Explicitly enable the retained RMSNorm candidate.
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_CUSTOM_RMS_NORM=1

# Roll back all optional DAS custom operators.
export VLLM_HCU_USE_CUSTOM_OPS=0

# Or roll back only the LightOp RMSNorm path.
export VLLM_HCU_USE_CUSTOM_RMS_NORM=0

# Roll back FlashAttention without changing the Plugin installation.
vllm serve /model/Qwen2-0.5B-Instruct --attention-backend TRITON_ATTN
```

Device acceptance for v0.33 is tracked separately in the
[current validation record](../docs/validation/vllm033-upgrade.md).
Historical gfx936 results do not certify this release or another architecture.
Keep RMSNorm disabled until the intended stack has a live-device validation
record:

```bash
export VLLM_HCU_USE_CUSTOM_RMS_NORM=0
```

For complete removal, uninstall `hcu-vllm-das-plugin`; the adapter and vLLM
remain installed and retain platform ownership.

## Historical v0.32 evidence

The [`SiluAndMul`](../docs/validation/v032-adapter-extension-silu.md) and
[`RMSNorm`](../docs/validation/v032-adapter-extension-rmsnorm.md) records are
historical candidate evidence. The v0.32 three-way identities, upstream
takeover decision, and base final-wheel gates were recorded in the
[`vLLM 0.32 upstream-first validation`](../docs/validation/vllm032-upstream-first-operator-ab.md).
The explicit AITER W8A8 route, public vendor-kernel evidence, Graph replay,
model smoke, and isolated HumanEval64 result are recorded in the
[`vLLM 0.32 AITER INT8 MoE validation`](../docs/validation/vllm032-aiter-int8-moe.md).
