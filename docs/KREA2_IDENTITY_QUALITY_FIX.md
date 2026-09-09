# Krea2 Identity quality parity fixes

The first WebbDuck Krea2Edit renders exposed a model-math issue rather than a
simple resolution/upscaling problem. This patch layer keeps the existing
hardware/offload work intact while bringing identity conditioning back toward
the upstream v1.2/v1.2.4 reference contract.

Changes:

- Correct source VAE latent normalization to `(z - mean) / std` and use the
  explicit inverse `z * std + mean` on decode.
- Keep the requested target aspect ratio. Krea2Edit v1.2 FIT prepares the
  reference separately instead of forcing output AR to source AR.
- Port current pixel-space FIT behavior: near-matched AR minimal crop, genuine
  AR mismatch /16 floor alignment, crop-to-grid before resize, and centered
  fractional RoPE offsets.
- Treat missing Qwen3-VL grounded conditioning as an identity error rather than
  silently rendering with text-only conditioning. Debug-only opt-out:
  `WEBBDUCK_KREA2_IDENTITY_ALLOW_TEXT_ONLY=1`.
- Use v1.2 baseline identity defaults when ordinary Krea defaults are still in
  the request: Turbo 10 steps / CFG 0; Raw 20 steps / CFG 3; ref_boost 4;
  grounding 768; LoRA scale 1.0.
- Resolve the identity LoRA rank with host GPU VRAM when available. Set
  `WEBBDUCK_KREA2_IDENTITY_QUALITY_BASELINE=1` to force the full v1.2 weight for
  parity testing.
- Post-hoc Real-ESRGAN is disabled by default while identity parity is being
  validated. Explicitly set `WEBBDUCK_KREA2_IDENTITY_UPSCALE=1` to re-enable it.

Recommended first parity run: Krea-2-Turbo, 10 steps, CFG 0, LoRA 1.0,
ref_boost 4, grounding 768, fixed seed, simple daylight portrait/restaging
prompt, and no post-hoc upscale.
