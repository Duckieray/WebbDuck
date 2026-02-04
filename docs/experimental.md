# Experimental Features

⚠️ All experimental features are **disabled by default**.

They exist to explore new ideas safely, not to replace stable workflows.

---

## Experimental Toggle

Experimental features are enabled via an explicit toggle.

When enabled:
- logs include ⚠️ EXPERIMENTAL markers
- behavior may differ from standard Diffusers
- failures are not considered bugs

---

## Current Experiments

### Two-Pass Generation

Two-pass generation injects late prompt concepts after base composition.

The second pass may use:
- a true SDXL refiner (special conditioning path), or
- a generic SDXL img2img model

WebbDuck validates the selected model before generation and will refuse
to run if the requested second-pass mode does not match the model's architecture.

### Prompt Decomposition
- Splits prompt into early vs late concepts
- Token-aware
- Still under evaluation

---

## Rules

Experimental features:
- must not alter standard behavior when disabled
- must be removable without breaking the app
- must live in `experimental/`
