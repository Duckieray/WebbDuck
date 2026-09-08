"""Adaptive request planner for the hardened Krea 2 isolated worker.

This layer sits above ``krea2_worker_safe`` and makes request-level decisions
that are easier to reason about than low-level offload policy:

* constrain large Krea requests to accelerator- and VRAM-aware image-token
  budgets so the worker stops repeatedly attempting pathological shapes;
* reduce default-looking step counts on constrained cards to bring generation
  time down to something less punishing than 28-step offloaded denoising; and
* report the effective request and post-run cleanup state back to WebbDuck.

The underlying hardened worker still owns dtype coercion, offload fallback, and
all Krea-specific execution details. When the installed CUDA/PyTorch runtime
passes a native FP8 GEMM probe, scaled Krea linears also use the optional
``krea2_native_fp8`` kernel instead of reconstructing BF16 weights every call.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Fragmentation can become visible during Krea's text-encoder -> transformer ->
# VAE phase transitions. Set the PyTorch allocator policy before importing torch
# while preserving an explicit operator/user override.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backends import krea2_native_fp8 as native_fp8
from core.backends import krea2_worker_safe as safe
from runtime_hardware import detect_torch_hardware

_EFFECTIVE_TOKEN_MULTIPLE = 16


def _snap(value: int, multiple: int = _EFFECTIVE_TOKEN_MULTIPLE) -> int:
    return max(multiple, int(round(float(value) / multiple) * multiple))


def _image_tokens(width: int, height: int) -> int:
    width = _snap(width)
    height = _snap(height)
    return (width // _EFFECTIVE_TOKEN_MULTIPLE) * (height // _EFFECTIVE_TOKEN_MULTIPLE)


def _base_token_budget(
    total_vram_gb: float,
    variant: str,
    accelerator: str,
) -> int | None:
    lowered_variant = str(variant or "base").lower()
    accelerator = str(accelerator or "cpu").lower()

    if accelerator == "cuda":
        if lowered_variant == "turbo":
            if total_vram_gb <= 12.5:
                return 3584
            if total_vram_gb <= 16.5:
                return 4096
            if total_vram_gb <= 20.5:
                return 4608
            return None

        if total_vram_gb <= 12.5:
            return 3072
        if total_vram_gb <= 16.5:
            return 3584
        if total_vram_gb <= 20.5:
            return 4096
        if total_vram_gb <= 24.5:
            return 4608
        return None

    if accelerator == "rocm":
        # ROCm currently uses synchronous group offload in the hardened worker,
        # so keep a slightly more conservative request envelope until we have
        # broad AMD hardware coverage.
        if lowered_variant == "turbo":
            if total_vram_gb <= 16.5:
                return 3584
            if total_vram_gb <= 24.5:
                return 4096
            return None

        if total_vram_gb <= 16.5:
            return 3072
        if total_vram_gb <= 24.5:
            return 4096
        return None

    return None


def _token_budget(hardware: dict[str, Any], variant: str) -> int | None:
    accelerator = str(hardware.get("accelerator") or "cpu").lower()
    if accelerator not in {"cuda", "rocm"}:
        return None

    total_vram_gb = float(hardware.get("total_vram_gb") or 0.0)
    free_vram_gb = float(hardware.get("free_vram_gb") or 0.0)
    budget = _base_token_budget(total_vram_gb, variant, accelerator)
    if budget is None:
        return None

    # The same physical GPU can have materially different safe request sizes on
    # a headless machine versus a desktop driving multiple displays/browser
    # windows. Use live free VRAM to reduce the request budget under pressure.
    if total_vram_gb > 0 and free_vram_gb > 0:
        occupied_gb = max(0.0, total_vram_gb - free_vram_gb)
        free_fraction = free_vram_gb / total_vram_gb
        if free_vram_gb < 10.0 or occupied_gb >= 4.0 or free_fraction < 0.67:
            budget = min(budget, 3072)
        elif free_vram_gb < 11.5 or occupied_gb >= 3.0 or free_fraction < 0.74:
            budget = min(budget, 3328)
        elif occupied_gb >= 2.25 or free_fraction < 0.80:
            budget = min(budget, 3584)

    return budget


def _fit_token_budget(width: int, height: int, token_budget: int) -> tuple[int, int]:
    width = _snap(width)
    height = _snap(height)
    current_tokens = _image_tokens(width, height)
    if current_tokens <= max(1, int(token_budget)):
        return width, height

    target_ratio = float(width) / float(max(1, height))
    ideal_width_tokens = math.sqrt(float(token_budget) * target_ratio)
    ideal_height_tokens = math.sqrt(float(token_budget) / target_ratio)
    center_w = max(1, int(round(ideal_width_tokens)))
    center_h = max(1, int(round(ideal_height_tokens)))

    best: tuple[float, int, int] | None = None
    for width_tokens in range(max(1, center_w - 10), center_w + 11):
        for height_tokens in range(max(1, center_h - 10), center_h + 11):
            area = width_tokens * height_tokens
            if area > token_budget:
                continue
            ratio = float(width_tokens) / float(height_tokens)
            ratio_error = abs(math.log(max(1e-9, ratio / target_ratio)))
            unused_fraction = float(token_budget - area) / float(max(1, token_budget))
            # Preserve composition shape much more strongly than squeezing the
            # final few percent of theoretical token budget out of the GPU.
            score = (ratio_error * 10.0) + unused_fraction
            candidate = (score, width_tokens, height_tokens)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        scale = math.sqrt(float(token_budget) / float(current_tokens))
        return (
            _snap(max(_EFFECTIVE_TOKEN_MULTIPLE, int(width * scale))),
            _snap(max(_EFFECTIVE_TOKEN_MULTIPLE, int(height * scale))),
        )

    return best[1] * _EFFECTIVE_TOKEN_MULTIPLE, best[2] * _EFFECTIVE_TOKEN_MULTIPLE


def _cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    """Read a bounded optional float, falling back to ``default`` on junk."""
    value = (cfg or {}).get(key)
    try:
        return float(value) if value is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _reference_source_size(identity_cfg: dict[str, Any]) -> tuple[int, int, bool]:
    """Best-effort pixel size of the identity reference; ``(w, h, readable)``.

    Reads only the image header (PIL lazy decode) so the planner can predict the
    worker's ``edit_target_size`` containment without GPU work. Returns
    ``(0, 0, False)`` when there is no reference or it cannot be opened — the
    planner then takes no geometry action (the worker validates the real file).
    """
    path = str((identity_cfg or {}).get("reference_image") or "").strip()
    if not path:
        return 0, 0, False
    try:
        from PIL import Image

        with Image.open(path) as im:
            width, height = im.size
        return max(1, int(width)), max(1, int(height)), True
    except Exception:
        return 0, 0, False


def _identity_token_budget(hardware: dict[str, Any], variant: str) -> int | None:
    """Conservative target-token budget for identity edits on this accelerator.

    An identity forward packs the reference and target grids into one combined
    image sequence, so the *same* output grid costs roughly twice the
    image-side tokens of a text2img step. The planner therefore halves the card's
    text2img budget (which already reflects live free-VRAM pressure) before it
    compares the requested output grid — a strictly safer envelope.
    """
    base = _token_budget(hardware, variant)
    if base is None:
        return None
    multiple = _EFFECTIVE_TOKEN_MULTIPLE
    return max(multiple, (int(base) // (2 * multiple)) * multiple)


def _identity_token_plan(
    identity_cfg: dict[str, Any],
    source_size: tuple[int, int],
    requested_box: tuple[int, int],
    budget: int | None,
) -> tuple[dict[str, Any], tuple[int, int]]:
    """Estimate identity edit token cost and the box that keeps it in budget.

    Mirrors ``krea2_worker._run_identity`` exactly: ``edit_target_size``
    contains the source within the requested box (``fit_mode``/``max_megapixels``
    honored, 16-px grid), then ``grid_dims`` -> token counts. ``reference_tokens``
    is the packed source latent grid; ``combined_image_tokens`` counts the whole
    image side ``[ref | target]`` (text sequence excluded).

    Returns ``(accounting, tuned_box)``. When the reference is not readable the
    planner returns the requested box unchanged (no geometry action without
    ground truth); otherwise it shrinks the box (same-AR fit) only as far as
    needed to bring the *target* grid under the conservative budget.
    """
    from core.backends.krea2_identity import (
        combined_token_count,
        edit_target_size,
        grid_dims,
        source_token_count,
    )

    src_w, src_h = source_size
    req_w, req_h = requested_box
    account: dict[str, Any] = {
        "source_size": [src_w, src_h],
        "requested_size": [req_w, req_h],
        "effective_size": [req_w, req_h],
        "reference_tokens": 0,
        "target_tokens": 0,
        "combined_image_tokens": 0,
        "token_budget": budget,
        "reference_readable": src_w > 0 and src_h > 0,
        "resolution_scaled": False,
    }
    if not (src_w > 0 and src_h > 0):
        return account, (req_w, req_h)

    box = (req_w, req_h)
    fit_mode = str((identity_cfg or {}).get("fit_mode") or "fit")
    max_megapixels = _cfg_float(identity_cfg, "max_megapixels", 1.0)
    for _ in range(8):
        eff_h, eff_w = edit_target_size(
            (src_w, src_h),
            box,
            fit_mode=fit_mode,
            max_megapixels=max_megapixels,
            multiple=_EFFECTIVE_TOKEN_MULTIPLE,
        )
        grid_h, grid_w = grid_dims(eff_w, eff_h)
        target_tokens = source_token_count(grid_h, grid_w)
        account.update(
            {
                "effective_size": [eff_w, eff_h],
                "reference_tokens": source_token_count(grid_h, grid_w),
                "target_tokens": target_tokens,
                "combined_image_tokens": combined_token_count(0, grid_h, grid_w),
            }
        )
        if budget is None or target_tokens <= budget:
            break
        box = _fit_token_budget(eff_w, eff_h, budget)

    scaled = box != (req_w, req_h)
    account["resolution_scaled"] = scaled
    return account, box


def _tuned_default_steps(
    steps: int,
    hardware: dict[str, Any],
    variant: str,
    *,
    resized: bool,
) -> int:
    accelerator = str(hardware.get("accelerator") or "cpu").lower()
    if accelerator not in {"cuda", "rocm"}:
        return steps

    total_vram_gb = float(hardware.get("total_vram_gb") or 0.0)
    lowered_variant = str(variant or "base").lower()

    # Only tune the common Raw default-looking path. Custom non-default values
    # remain user intent and pass through unchanged.
    if lowered_variant == "turbo" or int(steps) != 28:
        return steps

    if accelerator == "rocm":
        if total_vram_gb <= 16.5:
            return 12 if resized else 14
        if total_vram_gb <= 24.5:
            return 16 if resized else 18
        return steps

    if total_vram_gb <= 12.5:
        return 12 if resized else 14
    if total_vram_gb <= 16.5:
        return 14 if resized else 16
    if total_vram_gb <= 20.5:
        return 18 if resized else 20
    return steps


def _adaptive_request(
    request: dict[str, Any],
    hardware: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    tuned = dict(request)
    variant = str(tuned.get("variant") or "base").lower()
    identity_enabled = bool(tuned.get("identity"))
    requested_width = _snap(int(tuned.get("width") or 1024))
    requested_height = _snap(int(tuned.get("height") or 1024))
    requested_steps = max(
        1,
        int(
            tuned.get("steps")
            or (10 if identity_enabled else (8 if variant == "turbo" else 28))
        ),
    )

    identity_accounting: dict[str, Any] | None = None
    budget: int | None = None
    tuned_width, tuned_height = requested_width, requested_height

    if identity_enabled:
        # Identity edits derive their output size from the reference AR
        # (edit_target_size) and use the LoRA card's own step recipe, so the
        # default-step tuning never applies. Geometry is still planned against a
        # conservative token budget: an identity forward packs the reference and
        # target grids into one sequence, so the same output grid costs ~2x the
        # image-side tokens of a text2img step on this card.
        identity_cfg = tuned.get("identity")
        identity_cfg = identity_cfg if isinstance(identity_cfg, dict) else {}
        src_w, src_h, readable = _reference_source_size(identity_cfg)
        identity_budget = _identity_token_budget(hardware, variant)
        identity_accounting, identity_box = _identity_token_plan(
            identity_cfg,
            (src_w, src_h),
            (requested_width, requested_height),
            identity_budget,
        )
        if readable:
            tuned_width, tuned_height = identity_box
    else:
        budget = _token_budget(hardware, variant)
        if budget:
            tuned_width, tuned_height = _fit_token_budget(
                requested_width,
                requested_height,
                budget,
            )

    resized = tuned_width != requested_width or tuned_height != requested_height
    tuned_steps = (
        requested_steps
        if identity_enabled
        else _tuned_default_steps(
            requested_steps,
            hardware,
            variant,
            resized=resized,
        )
    )

    tuned["width"] = tuned_width
    tuned["height"] = tuned_height
    tuned["steps"] = tuned_steps

    plan = {
        "requested_width": requested_width,
        "requested_height": requested_height,
        "effective_width": tuned_width,
        "effective_height": tuned_height,
        "requested_steps": requested_steps,
        "effective_steps": tuned_steps,
        "requested_image_tokens": _image_tokens(requested_width, requested_height),
        "effective_image_tokens": _image_tokens(tuned_width, tuned_height),
        "token_budget": budget,
        "resolution_scaled": resized,
        "steps_tuned": tuned_steps != requested_steps,
        "identity_enabled": identity_enabled,
        "identity": identity_accounting,
        "variant": variant,
        "accelerator": str(hardware.get("accelerator") or "cpu"),
        "hardware_total_vram_gb": round(
            float(hardware.get("total_vram_gb") or 0.0),
            3,
        ),
        "hardware_free_vram_gb": round(
            float(hardware.get("free_vram_gb") or 0.0),
            3,
        ),
    }
    return tuned, plan


def _cleanup_accelerator_state() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _post_cleanup_hardware() -> dict[str, Any]:
    try:
        return detect_torch_hardware(torch)
    except Exception:
        return {}


def _run(
    request: dict[str, Any],
    output_dir: Path,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    hardware = detect_torch_hardware(torch)
    native_fp8.enable_for_hardware(hardware, safe.base)
    tuned_request, plan = _adaptive_request(request, hardware)

    if progress_path is not None and (plan["resolution_scaled"] or plan["steps_tuned"]):
        safe.base._write_progress(
            progress_path,
            "Adapting Krea request for this GPU",
            0.03,
        )

    try:
        result = safe._run(tuned_request, output_dir, progress_path)
    finally:
        _cleanup_accelerator_state()

    post_cleanup = _post_cleanup_hardware()
    runtime = result.setdefault("runtime", {})
    if isinstance(runtime, dict):
        runtime["adaptive_request"] = plan
        runtime["requested_width"] = plan["requested_width"]
        runtime["requested_height"] = plan["requested_height"]
        runtime["requested_steps"] = plan["requested_steps"]
        runtime["native_fp8"] = native_fp8.stats()
        runtime["post_cleanup_free_vram_gb"] = round(
            float(post_cleanup.get("free_vram_gb") or 0.0),
            3,
        )
        runtime["post_cleanup_total_vram_gb"] = round(
            float(post_cleanup.get("total_vram_gb") or 0.0),
            3,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Krea 2 worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress", default=None)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    progress_path = Path(args.progress) if args.progress else None

    try:
        result = _run(request, Path(args.output_dir), progress_path)
        payload = result if isinstance(result, dict) else {"ok": True, "result": result}
        if "ok" not in payload:
            payload["ok"] = True
        exit_code = 0
    except Exception as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1

    Path(args.result).write_text(json.dumps(payload), encoding="utf-8")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
