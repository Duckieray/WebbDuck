"""WebbDuck-native Krea 2 Identity Edit persona adapter (provider: ``krea2_identity_edit``).

This is the pure-Python contract layer for running the community LoRA
``conradlocke/krea2-identity-edit`` on top of a Krea 2 checkpoint through the
isolated Krea worker. It owns everything that can be reasoned about and tested
*without* a GPU:

* the identity settings snapshot / validation contract (``identity_settings_snapshot``);
* identity weight resolution (repo, rank preference, env overrides) — weights are
  never checked into the repo;
* the ai-toolkit/ComfyUI -> Diffusers LoRA key conversion
  (``convert_lora_keys``) that fails loudly on unknown keys and reports
  mapped/ignored/unresolved tensor counts;
* the dual-conditioning geometry (grounded VLM resolution, VAE target size,
  3-axis RoPE position IDs, packed-latent token counts);
* the ``ref_boost`` additive attention bias and the GQA-safe processor factory
  that applies it; and
* the source-preserving transformer forward mirroring ComfyUI-Krea2Edit's
  ``krea2_edit_forward`` against ``Krea2Transformer2DModel``.

The LoRA uses *dual conditioning*: (1) semantics via the image-grounded
Qwen3-VL text encoder (instruction + source image -> Krea prompt embeddings),
and (2) appearance via the VAE-encoded, normalized source latent prepended as
clean tokens. The sequence becomes ``[text | source(frame=1) | target(frame=0)]``
and only the target tokens are kept as the velocity prediction. The actual
execution (phased worker encode/denoise/decode) lives in the isolated Krea
runtime and consumes this contract.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROVIDER_ID = "krea2_identity_edit"
DEFAULT_REPO = "conradlocke/krea2-identity-edit"
DEFAULT_WEIGHT_FILE = "krea2_identity_edit_v1_2.safetensors"


def apply_identity_perf_overrides(steps: int, guidance: float) -> tuple[int, float]:
    """Identity-only A/B performance overrides (env-gated; default: no change).

    Each identity run pays for ``steps`` x (CFG rows) full transformer forwards
    of a ~30B fp8 model streamed over PCIe, so GPU work scales with both knobs.
    These overrides tune them *without touching the shared text2img path* for
    side-by-side A/B on live hardware:

    * ``WEBBDUCK_KREA2_IDENTITY_STEPS`` — int override for identity denoise steps;
    * ``WEBBDUCK_KREA2_IDENTITY_GUIDANCE`` — float override replacing the request
      cfg; ``<= 0`` skips the negative (uncond) forward entirely;
    * ``WEBBDUCK_KREA2_IDENTITY_CFG_FREE``=1 — shorthand forcing ``guidance = 0``
      (single forward per step).
    """
    import os

    raw_steps = os.getenv("WEBBDUCK_KREA2_IDENTITY_STEPS")
    if raw_steps is not None and str(raw_steps).strip():
        steps = max(1, int(raw_steps))
    raw_guidance = os.getenv("WEBBDUCK_KREA2_IDENTITY_GUIDANCE")
    if raw_guidance is not None and str(raw_guidance).strip():
        guidance = float(raw_guidance)
    if os.getenv("WEBBDUCK_KREA2_IDENTITY_CFG_FREE") == "1":
        guidance = 0.0
    return steps, guidance


def _probe_mem(stage: str, **shapes: Any) -> None:
    """Env-gated (``WEBBDUCK_KREA2_DEBUG_PRINT``) CUDA-stage memory probe.

    Prints to stderr so it lands in a failed job's captured error payload.
    Lazy imports keep this importable with no runtime torch installed.
    """
    if not os.environ.get("WEBBDUCK_KREA2_DEBUG_PRINT"):
        return
    try:
        import sys as _sys

        import torch as _t

        if not _t.cuda.is_available():
            return
        shape_str = " ".join("%s=%s" % (k, v) for k, v in shapes.items())
        print(
            "KREA2_MEM %s %salloc=%.3fGb reserved=%.3fGb" % (
                stage,
                shape_str + " " if shape_str else "",
                _t.cuda.memory_allocated() / 1e9,
                _t.cuda.memory_reserved() / 1e9,
            ),
            file=_sys.stderr,
            flush=True,
        )
    except Exception:
        pass

DEFAULT_REF_BOOST = 2.0       # v1.2 likeness dial: 1.0 = off, ~2 = balanced, strong locks composition
DEFAULT_GROUNDING_PX = 768    # LoRA trained dial 384-768; higher often still works
DEFAULT_FIT_MODE = "fit"
DEFAULT_LORA_SCALE = 1.0
DEFAULT_MAX_MEGAPIXELS = 1.0  # edit path prepends the source latent (~2x image tokens)

_REF_BOOST_RANGE = (0.0, 10.0)
_GROUNDING_PX_RANGE = (512, 1536)
_LORA_SCALE_RANGE = (0.0, 1.5)

# Weight files ship a full-rank (r=2048) and two reduced-rank variants. The
# reduced ranks trade a little likeness for much smaller downloads and lower
# VRAM pressure. ``approx_bytes`` is informational; the real probe happens on
# disk/HF. Filenames must match the v1_2 naming in the upstream repo.
WEIGHT_SPECS: dict[str, dict[str, Any]] = {
    "full": {
        "filename": DEFAULT_WEIGHT_FILE,
        "approx_bytes": 1_830_000_000,
        "rank": 2048,
    },
    "r128": {
        "filename": "krea2_identity_edit_v1_2_r128.safetensors",
        "approx_bytes": 910_000_000,
        "rank": 128,
    },
    "r64": {
        "filename": "krea2_identity_edit_v1_2_r64.safetensors",
        "approx_bytes": 460_000_000,
        "rank": 64,
    },
}
DEFAULT_RANK = "r64"

_VALID_FIT_MODES = {"fit", "full"}


class KreaIdentityError(RuntimeError):
    """Identity/persona configuration or weight resolution failed for Krea 2."""


# --------------------------------------------------------------------------------------
# Identity settings snapshot (the request contract)
# --------------------------------------------------------------------------------------

@dataclass
class IdentitySnapshot:
    """Normalized, validated Krea identity request as consumed by the worker."""

    reference_image: str                 # absolute filesystem path (single anchor, v1)
    ref_boost: float = DEFAULT_REF_BOOST
    grounding_px: int = DEFAULT_GROUNDING_PX
    fit_mode: str = DEFAULT_FIT_MODE
    lora_scale: float = DEFAULT_LORA_SCALE
    max_megapixels: float = DEFAULT_MAX_MEGAPIXELS
    lora_rank: str = DEFAULT_RANK
    face_crop: str = "off"               # reserved (mirrors FLUX persona knob); v1: off
    warnings: list[str] = field(default_factory=list)

    @property
    def reference_count_used(self) -> int:
        return 1


def identity_repo() -> str:
    return str(os.getenv("WEBBDUCK_KREA2_IDENTITY_REPO") or DEFAULT_REPO).strip()


def identity_weight_override() -> Path | None:
    """Local safetensors override from ``WEBBDUCK_KREA2_IDENTITY_WEIGHT``."""
    raw = str(os.getenv("WEBBDUCK_KREA2_IDENTITY_WEIGHT") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise KreaIdentityError(
            f"WEBBDUCK_KREA2_IDENTITY_WEIGHT points at a missing file: {path}"
        )
    return path


def _weight_spec_for_rank(rank: str | None) -> dict[str, Any]:
    key = str(rank or DEFAULT_RANK).strip().lower()
    spec = WEIGHT_SPECS.get(key)
    if spec is None:
        raise KreaIdentityError(
            f"Unknown Krea identity LoRA rank {key!r}; expected one of "
            + ", ".join(sorted(WEIGHT_SPECS))
        )
    return spec


def preferred_rank(total_vram_gb: float | None) -> str:
    """Pick the identity weight rank for a card, favoring small downloads on
    constrained hardware (r64 ~457MB). Unspecified VRAM keeps the default."""
    if total_vram_gb is None or not math.isfinite(total_vram_gb):
        return DEFAULT_RANK
    if total_vram_gb < 12.5:
        return "r64"
    if total_vram_gb < 16.5:
        return "r128"
    return "full"


def _normalize_face_crop(value: Any) -> str:
    if value in ("off", "false", "0", "", None):
        return "off"
    if value in ("auto", "true", "1"):
        return "auto"
    raise KreaIdentityError(f"Invalid Krea identity face_crop value: {value!r}")


def identity_settings_snapshot(
    adapter_cfg: dict[str, Any] | None,
    *,
    total_vram_gb: float | None = None,
) -> IdentitySnapshot | None:
    """Normalize and validate a request's ``identity_adapter`` config for Krea.

    Returns ``None`` when identity is disabled or the config targets another
    provider. Raises :class:`KreaIdentityError` on malformed values so a bad
    persona request fails fast with an actionable message instead of rendering
    default-looking output silently.
    """
    if not isinstance(adapter_cfg, dict):
        return None
    if not adapter_cfg:
        return None
    if adapter_cfg.get("enabled") is False:
        return None
    provider = str(adapter_cfg.get("type") or "").strip()
    if provider and provider != PROVIDER_ID:
        return None

    raw_refs = adapter_cfg.get("reference_images") or adapter_cfg.get("refs") or []
    if not isinstance(raw_refs, list) or not raw_refs:
        raise KreaIdentityError(
            "Krea identity persona requires at least one reference image."
        )
    first = raw_refs[0]
    if not isinstance(first, str) or not first.strip():
        raise KreaIdentityError("Krea identity persona reference must be a path string.")
    if len(raw_refs) > 1:
        raise KreaIdentityError(
            "The Krea identity edit adapter is v1: exactly one anchor reference is "
            "supported per request (received %d). Multi-reference fusion is a "
            "future provider feature." % len(raw_refs)
        )

    ref_path = resolve_reference_path(raw_refs[0])

    def _float_in(name: str, value: Any, low: float, high: float, default: float) -> float:
        if value is None:
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise KreaIdentityError(
                f"Krea identity {name} must be a number, got {value!r}."
            ) from exc
        if not low <= parsed <= high:
            raise KreaIdentityError(
                f"Krea identity {name} {parsed} is outside the supported range "
                f"[{low}, {high}]."
            )
        return parsed

    warnings: list[str] = []
    ref_boost = _float_in(
        "ref_boost", adapter_cfg.get("ref_boost"),
        *_REF_BOOST_RANGE, DEFAULT_REF_BOOST,
    )

    grounding_px = int(
        _float_in(
            "grounding_px", adapter_cfg.get("grounding_px"),
            *_GROUNDING_PX_RANGE, DEFAULT_GROUNDING_PX,
        )
    )

    lora_scale = _float_in(
        "lora_scale", adapter_cfg.get("lora_scale"),
        *_LORA_SCALE_RANGE, DEFAULT_LORA_SCALE,
    )

    fit_mode = str(adapter_cfg.get("fit_mode") or DEFAULT_FIT_MODE).strip().lower()
    if fit_mode not in _VALID_FIT_MODES:
        raise KreaIdentityError(
            f"Krea identity fit_mode {fit_mode!r} is invalid; expected "
            + ", ".join(sorted(_VALID_FIT_MODES))
        )

    max_mp = _float_in(
        "max_megapixels", adapter_cfg.get("max_megapixels"),
        0.125, 2.0, DEFAULT_MAX_MEGAPIXELS,
    )

    lora_rank = preferred_rank(total_vram_gb)
    requested_rank = str(adapter_cfg.get("lora_rank") or "").strip().lower()
    if requested_rank:
        _weight_spec_for_rank(requested_rank)  # validates
        if requested_rank != lora_rank:
            warnings.append(
                f"requested identity rank {requested_rank} differs from the "
                f"GPU-preferred {lora_rank}; honoring the explicit rank."
            )
            lora_rank = requested_rank

    face_crop = _normalize_face_crop(adapter_cfg.get("face_crop", "off"))
    if face_crop == "auto":
        warnings.append(
            "Krea identity face_crop='auto' is reserved for a later release and "
            "currently behaves like 'off'."
        )
        face_crop = "off"  # report the effective value

    return IdentitySnapshot(
        reference_image=str(ref_path),
        ref_boost=ref_boost,
        grounding_px=grounding_px,
        fit_mode=fit_mode,
        lora_scale=lora_scale,
        max_megapixels=max_mp,
        lora_rank=lora_rank,
        face_crop=face_crop,
        warnings=warnings,
    )


# --------------------------------------------------------------------------------------
# Reference resolution (web path -> absolute filesystem path)
# --------------------------------------------------------------------------------------

def resolve_reference_path(value: str, output_base: str | Path | None = None) -> Path:
    """Resolve an identity reference to an absolute, existing file path.

    Mirrors WebbDuck's web-path convention (``outputs/refs/foo.png`` or
    ``/outputs/refs/foo.png`` -> ``<WEBBDUCK_OUTPUT_DIR>/refs/foo.png``) and
    accepts absolute filesystem paths directly, matching the FLUX persona path.
    Raises :class:`KreaIdentityError` when the file cannot be located.
    """
    raw = str(value).strip()
    if not raw:
        raise KreaIdentityError("Identity reference path is empty.")

    if raw.startswith("/outputs/") or raw.startswith("outputs/"):
        rel = raw.split("outputs/", 1)[1]
        base = output_base or os.getenv("WEBBDUCK_OUTPUT_DIR") or "outputs"
        candidate = Path(base).expanduser() / rel.lstrip("/")
    else:
        candidate = Path(raw).expanduser()

    resolved = candidate.resolve()
    if not resolved.is_file():
        raise KreaIdentityError(
            f"Krea identity reference image does not exist: {raw} (resolved {resolved})"
        )
    return resolved


# --------------------------------------------------------------------------------------
# Identity weight resolution
# --------------------------------------------------------------------------------------

@dataclass
class WeightResolution:
    path: str                # absolute local safetensors path
    rank: str
    source: str              # "env-override" | "repo" | "hf-cache"
    approx_bytes: int


def resolve_identity_weight(
    *,
    repo: str | None = None,
    rank: str | None = None,
    hf_hub_download: Callable[..., Path] | None = None,
) -> WeightResolution:
    """Resolve the identity LoRA safetensors to a local path.

    Precedence: ``WEBBDUCK_KREA2_IDENTITY_WEIGHT`` local file override, then the
    repo filename for the chosen rank, downloaded through the HF cache (which
    covers a warm local cache without network). Never vendors weights into the
    repo. A caller-supplied ``hf_hub_download`` (e.g. ``huggingface_hub.hf_hub_download``)
    is used when provided so callers can inject their own token/auth.
    """
    override = identity_weight_override()
    if override is not None:
        return WeightResolution(
            path=str(override),
            rank="local",
            source="env-override",
            approx_bytes=override.stat().st_size,
        )

    spec = _weight_spec_for_rank(rank)
    target_repo = repo or identity_repo()

    if hf_hub_download is None:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover - import env drift
            raise KreaIdentityError(
                f"huggingface_hub is required to resolve the Krea identity LoRA "
                f"({target_repo})."
            ) from exc

    try:
        local_path = hf_hub_download(
            repo_id=target_repo,
            filename=spec["filename"],
        )
    except Exception as exc:
        raise KreaIdentityError(
            f"Unable to download Krea identity LoRA {spec['filename']} from "
            f"{target_repo}: {exc}",
        ) from exc

    return WeightResolution(
        path=str(Path(local_path).resolve()),
        rank=spec["rank"],
        source="repo",
        approx_bytes=int(spec.get("approx_bytes") or 0),
    )


# --------------------------------------------------------------------------------------
# ai-toolkit/ComfyUI -> Diffusers LoRA key conversion
# --------------------------------------------------------------------------------------

# Byte-for-byte the mapping shipped by the upstream app (plus ``txtfusion`` ->
# ``text_fusion`` handled structurally below). Order matters: ``.attn.wo.`` is
# applied before any longer suffix exists, and each entry only matches its own
# dotted token so ``wq``/``wk``/``wv`` never collide.
_ATTN_MLP_MAP = (
    (".attn.wq.", ".attn.to_q."),
    (".attn.wk.", ".attn.to_k."),
    (".attn.wv.", ".attn.to_v."),
    (".attn.wo.", ".attn.to_out.0."),
    (".attn.gate.", ".attn.to_gate."),
    (".mlp.gate.", ".ff.gate."),
    (".mlp.up.", ".ff.up."),
    (".mlp.down.", ".ff.down."),
)

_KREA2_KEY_PREFIXES = ("diffusion_model.", "model.diffusion_model.")

# Suffixes we deliberately tolerate as non-weight metadata.
_IGNORED_SUFFIXES = (
    ".lora_alpha",
    ".lora_te1",
)


def _looks_like_lora_weight(key: str) -> bool:
    lowered = key.lower()
    return (
        ".weight" in lowered
        and (".lora_" in lowered or ".lora" in lowered)
    ) or ("lora" in lowered and lowered.endswith((".lora_a", ".lora_b")))


def analyze_lora_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Classify tensors as map-able, ignorable, or unresolved.

    Returns ``{"mapped": [...], "ignored": [...], "unresolved": [...]}``. A key
    is *ignored* when it is metadata (alpha/LoRA scale) or carries no prefix we
    claim to understand; a key is *unresolved* (report + fail loud) when it
    looks like an actual LoRA weight in an unmapped location.
    """
    mapped: list[str] = []
    ignored: list[str] = []
    unresolved: list[str] = []

    for raw_key in state_dict:
        key = str(raw_key)
        if key.endswith(_IGNORED_SUFFIXES):
            ignored.append(key)
            continue

        stripped = key
        for prefix in _KREA2_KEY_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break

        candidate = stripped
        if candidate.startswith("blocks."):
            candidate = "transformer_blocks." + candidate[len("blocks."):]
        elif candidate.startswith("txtfusion."):
            candidate = "text_fusion." + candidate[len("txtfusion."):]

        for source, target in _ATTN_MLP_MAP:
            if source in candidate:
                candidate = candidate.replace(source, target)

        if candidate != stripped or candidate != key:
            mapped.append(key)
        elif _looks_like_lora_weight(key):
            unresolved.append(key)
        else:
            ignored.append(key)

    return {"mapped": mapped, "ignored": ignored, "unresolved": unresolved}


def split_identity_lora_key(key: str) -> tuple[str | None, str | None]:
    """Return ``(module_path, kind)`` for the suffix of a converted LoRA key.

    ``kind`` is ``"a"`` / ``"b"`` for the low-rank factors and ``"alpha"`` for
    the module alpha scalar; trailing metadata that is not a weight (or an
    unrecognized key) returns ``(None, None)`` so runtime installers can skip
    it without failing loudly.
    """
    for suffix, kind in (
        (".lora_A.weight", "a"),
        (".lora_B.weight", "b"),
        (".lora_alpha", "alpha"),
    ):
        if str(key).endswith(suffix):
            return str(key)[: -len(suffix)], kind
    return None, None


def convert_lora_keys(
    state_dict: dict[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Convert ai-toolkit/ComfyUI Krea identity LoRA keys to Diffusers paths.

    ``strict=True`` (default) raises :class:`KreaIdentityError` on any
    unresolved tensor so a naming drift in the community weight fails loudly
    instead of silently loading a partially-applied adapter. The returned dict
    is the converted state dict.
    """
    report = analyze_lora_keys(state_dict)
    if report["unresolved"] and strict:
        unresolved = "\n  - ".join(report["unresolved"])
        raise KreaIdentityError(
            f"Krea identity LoRA contains {len(report['unresolved'])} unresolved "
            f"tensors that cannot be mapped to Diffusers module paths:\n  - {unresolved}"
        )

    out: dict[str, Any] = {}
    for raw_key, tensor in state_dict.items():
        key = str(raw_key)
        stripped = key
        for prefix in _KREA2_KEY_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break
        candidate = stripped
        if candidate.startswith("blocks."):
            candidate = "transformer_blocks." + candidate[len("blocks."):]
        elif candidate.startswith("txtfusion."):
            candidate = "text_fusion." + candidate[len("txtfusion."):]
        for source, target in _ATTN_MLP_MAP:
            if source in candidate:
                candidate = candidate.replace(source, target)
        out[candidate] = tensor

    return out


# --------------------------------------------------------------------------------------
# Dual-conditioning geometry
# --------------------------------------------------------------------------------------

# The image-grounded instruction template from ComfyUI-Krea2Edit. The system
# prefix is byte-identical to the diffusers Krea 2 text template; the difference
# is the <|vision_start|><|image_pad|><|vision_end|> block inserted before the
# instruction so the VLM grounds the edit on the source image.
GROUNDED_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{}<|im_end|>\n<|im_start|>assistant\n"
)

_TEMPLATE_PREFIX_IDX = 34   # drop the system prefix, matching the text-only path


def grounded_template(instruction: str) -> str:
    return GROUNDED_TEMPLATE.format(instruction or "")


def template_prefix_idx() -> int:
    return _TEMPLATE_PREFIX_IDX


def edit_target_size(
    source_size: tuple[int, int],
    requested_size: tuple[int, int] | None = None,
    *,
    fit_mode: str = DEFAULT_FIT_MODE,
    max_megapixels: float = DEFAULT_MAX_MEGAPIXELS,
    multiple: int = 16,
) -> tuple[int, int]:
    """Choose the (height, width) grid for the edit.

    The card requires the output AR to match the source. ``fit_mode="fit"``
    preserves the request's composition by fitting *within* the requested box
    (contain), then caps at ``max_megapixels`` and snaps each side to
    ``multiple`` (vae_scale_factor * patch_size = 16). ``fit_mode="full"`` uses
    the requested size verbatim (still capped/snapped). When no width/height is
    requested, the source AR is the target AR.
    """
    src_w, src_h = source_size
    if src_w <= 0 or src_h <= 0:
        raise KreaIdentityError("Source reference has invalid dimensions.")

    fit = str(fit_mode or DEFAULT_FIT_MODE).strip().lower()
    if fit not in _VALID_FIT_MODES:
        raise KreaIdentityError(
            f"Krea identity fit_mode {fit!r} is invalid; expected "
            + ", ".join(sorted(_VALID_FIT_MODES))
        )

    if requested_size:
        req_w, req_h = requested_size
        if req_w <= 0 or req_h <= 0:
            raise KreaIdentityError("Requested edit dimensions must be positive.")
        w, h = float(req_w), float(req_h)
        if fit == "fit":
            # Contain the source inside the requested box (composition preserved),
            # then keep the request's AR as the output AR target.
            scale = min(float(req_w) / float(src_w), float(req_h) / float(src_h))
            w = src_w * scale
            h = src_h * scale
    else:
        w, h = float(src_w), float(src_h)

    w = max(1, int(round(w)))
    h = max(1, int(round(h)))

    mp = (w * h) / 1e6
    if mp > max(0.125, float(max_megapixels)):
        actual = max(0.125, float(max_megapixels))
        s = math.sqrt((actual * 1e6) / (w * h))
        w, h = int(round(w * s)), int(round(h * s))

    multiple = max(16, int(multiple))
    w = max(multiple, int((w // multiple) * multiple))
    h = max(multiple, int((h // multiple) * multiple))
    return h, w


def grid_dims(width: int, height: int, *, scale_factor: int = 8, patch_size: int = 2) -> tuple[int, int]:
    """Transformer grid (grid_h, grid_w) for a pixel size."""
    multiple = scale_factor * patch_size
    gh = max(1, int(height // multiple))
    gw = max(1, int(width // multiple))
    return gh, gw


def source_token_count(grid_h: int, grid_w: int) -> int:
    """Packed source-latent token count prepended to the sequence (appearance)."""
    return grid_h * grid_w


def combined_token_count(
    text_seq_len: int,
    grid_h: int,
    grid_w: int,
    *,
    reference_count: int = 1,
) -> int:
    """Total transformer sequence length for the edit forward:
    ``[text | refs*source | target]`` (all three axes for each image grid)."""
    return text_seq_len + (reference_count + 1) * grid_h * grid_w


def edit_position_ids(
    text_seq_len: int,
    grid_h: int,
    grid_w: int,
    n_src: int,
    device: Any,
) -> Any:
    """Build ``(text + n_src*grid + grid, 3)`` rotary coordinates.

    Text sits at ``(0, 0, 0)``; each source block gets ``frame=(i+1)`` with its
    (h, w); the target gets ``frame=0`` with (h, w) — identical axes to the
    upstream ``_edit_position_ids``.
    """
    import torch

    text_ids = torch.zeros(text_seq_len, 3, device=device)

    def _img_ids(frame: int) -> Any:
        ids = torch.zeros(grid_h, grid_w, 3, device=device)
        ids[..., 0] = frame
        ids[..., 1] = torch.arange(grid_h, device=device)[:, None]
        ids[..., 2] = torch.arange(grid_w, device=device)[None, :]
        return ids.reshape(grid_h * grid_w, 3)

    blocks = [text_ids]
    blocks += [_img_ids(i + 1) for i in range(n_src)]
    blocks += [_img_ids(0)]
    return torch.cat(blocks, dim=0)


def ref_boost_bias(
    text_len: int,
    src_len: int,
    tgt_len: int,
    ref_boost: float,
    device: Any,
    dtype: Any,
) -> Any:
    """Additive attention-logit bias for the v1.2 ``ref_boost`` fidelity dial.

    Target queries get ``log(ref_boost)`` added on the source-block keys —
    equivalent to multiplying the refs' post-softmax attention weight before
    renormalization. ``1.0`` returns ``None`` to keep the maskless GQA fast path.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise KreaIdentityError("torch is required to build the ref_boost bias.") from exc

    if ref_boost == 1.0:
        return None
    L = text_len + src_len + tgt_len
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
    rows0 = text_len + src_len
    bias[:, :, rows0:, text_len:rows0] = math.log(max(float(ref_boost), 1e-4))
    return bias


# --------------------------------------------------------------------------------------
# ref_boost-compatible attention processor (GQA-safe)
# --------------------------------------------------------------------------------------

def mask_compat_processor(base_cls: type) -> type:
    """Build a :class:`Krea2AttnProcessor` subclass with dense-mask support.

    Krea attention is GQA (48 query heads / 12 KV heads). A dense float
    attention mask makes the fused SDPA kernels refuse GQA inputs, so when a
    mask is present the K/V heads are ``repeat_interleave``\\ d to the full head
    count and attention dispatches with ``enable_gqa=False``. ``mask=None``
    delegates to the stock processor unchanged (identical to upstream), keeping
    the fast path byte-identical. The diffusers symbols are imported lazily on
    the masked path only, so calling with ``mask=None`` works even before the
    runtime diffusers (>= 0.39) is installed.
    """
    import torch

    class _Krea2IdentityMaskCompatProcessor(base_cls):  # type: ignore[misc,call-arg]
        def __call__(
            self,
            attn,
            hidden_states,
            attention_mask=None,
            image_rotary_emb=None,
        ):
            if attention_mask is None:
                return super().__call__(
                    attn, hidden_states, attention_mask, image_rotary_emb
                )

            from diffusers.models.attention_dispatch import dispatch_attention_fn
            from diffusers.models.embeddings import apply_rotary_emb

            query = attn.to_q(hidden_states).unflatten(
                -1, (attn.num_heads, attn.head_dim)
            )
            key = attn.to_k(hidden_states).unflatten(
                -1, (attn.num_kv_heads, attn.head_dim)
            )
            value = attn.to_v(hidden_states).unflatten(
                -1, (attn.num_kv_heads, attn.head_dim)
            )
            gate = attn.to_gate(hidden_states)
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
            rep = attn.num_heads // attn.num_kv_heads
            if rep > 1:
                key = key.repeat_interleave(rep, dim=2)
                value = value.repeat_interleave(rep, dim=2)
            _probe_mem(
                "attn:qkv ready",
                heads=attn.num_heads,
                q=tuple(query.shape),
                mask=str(attention_mask.shape) if attention_mask is not None else "none",
            )
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                enable_gqa=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
            _probe_mem("attn:dispatch done")
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states * torch.sigmoid(gate)
            return attn.to_out[0](hidden_states)

    return _Krea2IdentityMaskCompatProcessor


# --------------------------------------------------------------------------------------
# Source-preserving transformer forward
# --------------------------------------------------------------------------------------

def edit_transformer_forward(
    transformer: Any,
    latents: Any,
    src_packed: Any,
    prompt_embeds: Any,
    prompt_mask: Any,
    timestep: Any,
    position_ids: Any,
    *,
    ref_boost: float = 1.0,
) -> Any:
    """Run the Krea transformer with the source latent block prepended.

    Reproduces ComfyUI-Krea2Edit's ``krea2_edit_forward`` against the diffusers
    ``Krea2Transformer2DModel`` (img_in == m.first, transformer_blocks ==
    m.blocks, text_fusion/txt_in == m.txtfusion/m.txtmlp, rotary_emb ==
    m.pe_embedder, final_layer == m.last). Only the target tokens are kept as
    the velocity prediction.
    """
    import torch
    import torch.nn.functional as F

    m = transformer
    combined_img = torch.cat([src_packed, latents], dim=1)  # [source | target]

    temb = m.time_embed(timestep, dtype=latents.dtype)
    temb_mod = m.time_mod_proj(F.gelu(temb, approximate="tanh"))
    _probe_mem("time_embed")

    text_attn_mask = (
        prompt_mask[:, None, None, :] if prompt_mask is not None else None
    )
    enc = m.text_fusion(prompt_embeds, attention_mask=text_attn_mask)
    enc = m.txt_in(enc)
    _probe_mem("text_fusion/txt_in closed")

    img = m.img_in(combined_img)
    _probe_mem("img_in")
    hidden = torch.cat([enc, img], dim=1)  # [text | source | target]

    image_rotary_emb = m.rotary_emb(position_ids)

    attention_mask = ref_boost_bias(
        enc.shape[1],
        src_packed.shape[1],
        latents.shape[1],
        ref_boost,
        hidden.device,
        hidden.dtype,
    )

    _probe_mem("forward:embeds/enc/img hidden", enc=enc.shape[1], src=src_packed.shape[1], tgt=latents.shape[1])

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        kernel_ctx = sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        )
    except Exception:
        kernel_ctx = __import__("contextlib").nullcontext()

    with kernel_ctx:
        for block in m.transformer_blocks:
            hidden = block(hidden, temb_mod, image_rotary_emb, attention_mask)

    text_seq_len = enc.shape[1]
    tgt_len = latents.shape[1]
    hidden = hidden[:, text_seq_len:]   # [source | target]
    hidden = hidden[:, -tgt_len:]       # target only
    out = m.final_layer(hidden, temb)
    _probe_mem("final_layer out")
    return out


def edit_transformer_forward_paired(
    transformer: Any,
    latents: Any,
    src_packed: Any,
    prompt_embeds_pos: Any,
    prompt_mask_pos: Any,
    position_ids_pos: Any,
    prompt_embeds_neg: Any,
    prompt_mask_neg: Any,
    position_ids_neg: Any,
    timestep: Any,
    *,
    ref_boost: float = 1.0,
    device: Any = None,
) -> tuple[Any, Any]:
    """Run positive + negative identity edits sharing one block stream per step.

    Block-level group offload streams every transformer block CPU->GPU->CPU per
    forward, so the CFG-positive + CFG-negative pair doubles that churn (28
    steps x 2 rows = 56 full streams). This variant keeps the two rows in lock
    step: each block is loaded once per step, applied to the positive row, then
    the negative row, then offloaded — halving the transfer/Python-dispatch
    overhead while running both rows as fully independent forwards (identical
    math to two ``edit_transformer_forward`` calls, no attention coupling).

    Returns ``(out_pos, out_neg)`` velocity predictions, both target-only.
    """
    import torch
    import torch.nn.functional as F

    m = transformer
    device = torch.device(device if device is not None else latents.device)
    src = src_packed.to(device)
    lats = latents.to(device)

    temb = m.time_embed(timestep.to(device), dtype=lats.dtype)
    temb_mod = m.time_mod_proj(F.gelu(temb, approximate="tanh"))

    def _row(
        prompt_embeds: Any,
        prompt_mask: Any,
        position_ids: Any,
        combined_img: Any,
    ) -> tuple[Any, int, Any, Any]:
        text_attn_mask = (
            prompt_mask[:, None, None, :] if prompt_mask is not None else None
        )
        enc = m.text_fusion(prompt_embeds.to(device), attention_mask=text_attn_mask)
        enc = m.txt_in(enc)
        hidden = torch.cat([enc, m.img_in(combined_img)], dim=1)
        rotary = m.rotary_emb(position_ids.to(device))
        bias = ref_boost_bias(
            enc.shape[1],
            src.shape[1],
            lats.shape[1],
            ref_boost,
            hidden.device,
            hidden.dtype,
        )
        return hidden, enc.shape[1], rotary, bias

    combined_img = torch.cat([src, lats], dim=1)
    hidden_pos, text_len_pos, rotary_pos, bias_pos = _row(
        prompt_embeds_pos, prompt_mask_pos, position_ids_pos, combined_img
    )
    hidden_neg, text_len_neg, rotary_neg, bias_neg = _row(
        prompt_embeds_neg, prompt_mask_neg, position_ids_neg, combined_img
    )

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        kernel_ctx = sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        )
    except Exception:
        kernel_ctx = __import__("contextlib").nullcontext()

    with kernel_ctx:
        for block in m.transformer_blocks:
            block = block.to(device)
            hidden_pos = block(hidden_pos, temb_mod, rotary_pos, bias_pos)
            hidden_neg = block(hidden_neg, temb_mod, rotary_neg, bias_neg)
            block = block.to("cpu")

    tgt_len = lats.shape[1]
    out_pos = m.final_layer(
        hidden_pos[:, text_len_pos:][:, -tgt_len:], temb
    )
    out_neg = m.final_layer(
        hidden_neg[:, text_len_neg:][:, -tgt_len:], temb
    )
    return out_pos, out_neg


# Backward-compatible alias used by tests and early wiring.
_identity_settings = identity_settings_snapshot