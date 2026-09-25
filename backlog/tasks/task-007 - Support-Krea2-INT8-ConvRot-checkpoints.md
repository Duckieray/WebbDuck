---
id: TASK-007
title: Support Krea 2 INT8 ConvRot single-file checkpoints
status: Done
assignee: []
created_date: '2026-09-23 00:00'
updated_date: '2026-09-23 15:20'
labels:
  - backend
  - krea2
  - quantization
  - models
dependencies: []
references:
  - models/model_descriptor.py
  - models/quantization.py
  - core/backends/krea2_worker_impl.py
  - core/backends/krea2_host_impl.py
  - tests/test_krea2_single_file.py
  - docs/HARDWARE_SMOKE_MATRIX.md
documentation:
  - docs/HARDWARE_SMOKE_MATRIX.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Krea 2 single-file checkpoints produced by ai-toolkit INT8-Fast (`aitk_quantization:
convrot8`) store their transformer weights as `int8_tensorwise` with an online
Hadamard "ConvRot" rotation (`comfy_quant` payload `{"format": "int8_tensorwise",
"convrot": true, "convrot_groupsize": 256}`, INT8 `.weight` + F32 per-output-row
`.weight_scale`). WebbDuck currently classifies these files as `quantization ==
"convrot"` and blocks them (descriptor `supported == False`, loader hard-raises).
This task makes the existing Krea single-file diffusers loader dequantize INT8
ConvRot tensors at load time: `W ≈ (scale ⊙ q) @ H_block` using a power-of-4
Sylvester Hadamard (size = `convrot_groupsize`, normalized by `1/sqrt(size)`),
which cancels the rotation so the normal dense `nn.Linear`/`Conv2d` forward path
is unchanged. No new runtime or comfy-kitchen dependency is required; all existing
Krea features (turbo/base defaults, LoRA fuse, identity, image editing modes) work
on these checkpoints. Target file: `snofsSexNudesAndOther_krea2TurboV14.safetensors`
(turbo variant, so 8 steps / CFG 0 defaults).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Krea 2 single file whose `comfy_quant` configuration is `int8_tensorwise` + `convrot` is `supported == True` in the model catalog and is discoverable as runnable (turbo token in the name continues to select turbo defaults).
- [x] #2 The Krea single-file overlay dequantizes INT8 ConvRot weights to the compute dtype and applies the Hadamard derotation (`W ≈ (scale⊙q) @ H_block`, groupsize from the payload) with no activation-rotation changes in the inference path.
- [x] #3 INT8 `int8_tensorwise` tensors without the convrot flag dequantize via the per-output-row scale only; bare INT8 tensors without a quant contract and genuinely unsupported formats still raise clear errors.
- [x] #4 The loader/`supported` gates that previously refused convrot are removed; unsupported/unknown quant formats remain blocked.
- [x] #5 Focused tests: Hadamard builder matches the Comfy convention (orthogonal, normalized), dequant+derotation round-trip on synthetic INT8 ConvRot weights, convrot/plain-INT8 detection, and descriptor support. Full `tests/test_krea2_single_file.py` passes.
- [x] #6 Docs updated (HARDWARE_SMOKE_MATRIX convrot note) and runtime smoke run of the real snofs checkpoint exercised for verification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `models/quantization.py`: add `is_int8_tensorwise(config)` format helper (independent of `classify_comfy_quant`).
2. `core/backends/krea2_worker_impl.py`: add `_hadamard_matrix(size)` (power-of-4 Sylvester, normalized) and `_dequantize_int8_tensorwise(...)`; route INT8 + convrot through them in `_dequantize_source_tensor`; drop the convrot rejections in `overlay_single_file_transformer` and `_load_pipeline`.
3. `models/model_descriptor.py`: remove `convrot` from the rejected-quantization set in `ModelDescriptor.supported`.
4. Update `tests/test_krea2_single_file.py`: flip the convrot "not runnable" cases to supported; add a synthetic INT8 ConvRot dequant+derotation roundtrip and a Hadamard convention test.
5. Update `docs/HARDWARE_SMOKE_MATRIX.md` ConvRot note.
6. Run focused tests, then a real smoke run of the snofs checkpoint through the Krea worker.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Snofs INT8 ConvRot checkpoint generates through the real Krea worker pipeline (not mock only)
- [x] #2 Catalog reports supported and turbo defaults; generation is deterministic for a fixed seed
- [x] #3 LoRA/identity paths unchanged for dense Krea checkpoints; no memory leak or extra VRAM regression vs fp8 (identity + LoRA install validated end-to-end, see Evidence)
- [x] #4 Focused test suite passes and docs updated
- [ ] #5 Committed with a descriptive message
<!-- DOD:END -->

## Identity-path validation (2026-09-23, follow-up on DOD #3)

After the convrot smoke run, a full Krea identity (krea2edit v1.2.4 face-aware) request on the same snofs checkpoint was validated through the worker:

- Request mirrored a previously "stuck" real-user request: 896x1344, 12 steps, CFG 0, `lora_rank: r128`, ref_boost 4, grounding 768, seed 2586045233.
- Result: `ok: true`, image produced (576x864 RGB, healthy: full 0..255 range, ~141K colors).
- Runtime: quantization `convrot`, `execution_quantization: dense`, `parallel_transformer: true`, offload `paired-block`; `lora_install_seconds: 38.6`, pipeline load 53.5 s, denoise 148.6 s, worker total 325.6 s.
- Roots out a real perf bug surfaced by dense checkpoints: identity LoRA installation fused deltas via a bf16 CPU `mm` (slow fallback kernel ~8 s per 6144^2 module), making a full-r128 install take ~50 min on CPU-only transform streams. `_fuse_linear_lora` now computes the delta in fp32 and down-casts before `add_` (~80x faster, ~40 s install). Doc note added in `docs/KREA2_IDENTITY_QUALITY_FIX.md`.

## Verification Evidence (2026-09-23)

Real smoke run (Tier 1) of `snofsSexNudesAndOther_krea2TurboV14.safetensors` through the isolated Krea worker on RTX 5070 Ti (15.5 GB, bf16):

- Request: 768x768, 8 steps, CFG 0, seed 12345, variant turbo, component_source = drive Krea-2-Turbo snapshot (local path).
- Result: `ok: true`, image `krea2_000.png` (768x768 RGB, healthy stats: full 0..251 range, per-channel std 62-73, ~124K unique colors, not degenerate).
- Runtime: quantization `convrot`, `mapped_tensors: 430`, `ignored_tensors: 0`, `preserved_fp8_linears: 0` (all convrot dequantized to dense bf16), execution `dense`, offload `transformer-block-sync-2`, text encoder resident.
- Timing: pipeline load 66.9 s, prompt encode 22.7 s, denoise 177.1 s (~22 s/step), worker total 270.4 s.
- Tests: `tests/test_krea2_single_file.py` 12 passed; `test_model_descriptor.py` + `test_model_discovery.py` 21 passed. Cross-checked Hadamard + dequant/derotation bit-identical vs comfy_kitchen on the real 263 quantized layers.
- Not committed yet (DOD #5 open by intent); 7 unrelated suite failures reproduce at clean HEAD (pre-existing).