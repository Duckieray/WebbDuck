"""WebbDuck-owned metadata store for models, LoRAs, and embeddings.

WebbDuck is the canonical source of rich metadata (tags, families, modes,
pairings, descriptions).  External tools read this data through WebbDuck's
HTTP API instead of maintaining their own copy.

Schema (stored as JSON on disk under ``webbduck_meta/``)::

    webbduck_meta/
    ├── checkpoints.json   # { "checkpoints": { "<name>": { tags, families, … } } }
    ├── loras.json         # { "loras":       { "<name>": { tags, aliases, … } } }
    └── embeddings.json    # { "<name>":                     { file, token, … } }

Every entry is optional — missing files or empty dicts are handled gracefully.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
#  Path resolution
# ---------------------------------------------------------------------------

_DEFAULT_META_DIR_NAME = "webbduck_meta"
_META_DIR_CACHE: Path | None = None


def _resolve_meta_dir() -> Path:
    """Return the WebbDuck-owned metadata directory (always exists on return)."""
    global _META_DIR_CACHE
    if _META_DIR_CACHE is not None:
        return _META_DIR_CACHE

    explicit = os.getenv("WEBBDUCK_META_DIR")
    if explicit:
        _META_DIR_CACHE = Path(explicit).expanduser()
    else:
        _META_DIR_CACHE = Path(__file__).resolve().parent.parent / _DEFAULT_META_DIR_NAME

    _META_DIR_CACHE.mkdir(parents=True, exist_ok=True)
    return _META_DIR_CACHE


# ---------------------------------------------------------------------------
#  Internal meta dataclasses (deserialised from on-disk JSON)
# ---------------------------------------------------------------------------


@dataclass
class _CheckpointMeta:
    tags: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    recommended_mode: str | None = None
    url: str = ""
    description: str = ""


@dataclass
class _LoraMeta:
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)
    weight_hint: float | None = None
    url: str = ""
    trigger: str = ""


@dataclass
class _EmbeddingMeta:
    tags: list[str] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    url: str = ""
    description: str = ""
    disabled: bool = False
    disabled_reason: str = ""


# ---------------------------------------------------------------------------
#  Public asset-info dataclass (the unified view exposed via API)
# ---------------------------------------------------------------------------


@dataclass
class AssetInfo:
    """Unified view of a single asset — merges WebbDuck metadata with
    runtime registry data."""

    name: str
    type: str  # "checkpoint" | "lora" | "embedding"

    # Registry fields
    arch: str | None = None
    source: str | None = None
    path: str | None = None

    # Rich metadata (WebbDuck-owned)
    tags: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    description: str = ""
    url: str = ""
    recommended_mode: str | None = None
    weight_hint: float | None = None

    # Runtime registry extras
    trigger: str | None = None
    token: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False
    disabled_reason: str = ""


# ---------------------------------------------------------------------------
#  Mode synonyms for fuzzy matching
# ---------------------------------------------------------------------------

MODE_SYNONYMS: dict[str, set[str]] = {
    "anime": {"anime", "manga", "hentai", "pony", "animagine", "illustrious"},
    "hentai": {"hentai", "nsfw", "explicit", "erotic"},
    "realistic": {"realistic", "photo", "photorealistic", "realism"},
    "cartoon": {"cartoon", "comic", "toon", "stylized"},
    "furry": {"furry", "anthro", "kemono", "yiff"},
    "fantasy": {"fantasy", "medieval", "magic", "ethereal"},
    "all": {"all", "general", "universal"},
}

# ---------------------------------------------------------------------------
#  The store
# ---------------------------------------------------------------------------


class MetaStore:
    """Unified metadata store owned by WebbDuck.

    Rich metadata lives in ``webbduck_meta/`` on disk.  The store merges it
    with the in-memory runtime registries (``MODEL_REGISTRY``,
    ``LORA_REGISTRY``, ``EMBEDDING_REGISTRY``) so that every API response
    carries both structural info (arch, path, source) and semantic info
    (tags, families, recommendations).

    Usage::

        from models.metastore import meta_store

        assets  = meta_store.all_assets()
        loras   = meta_store.search(asset_type="lora", tag="anime")
        recs    = meta_store.recommended("Juggernaut-XI-v11")
    """

    def __init__(self, meta_dir: Path | None = None):
        self._meta_dir = meta_dir or _resolve_meta_dir()
        self._checkpoints: dict[str, _CheckpointMeta] = {}
        self._loras: dict[str, _LoraMeta] = {}
        self._embeddings: dict[str, _EmbeddingMeta] = {}
        self._ensure_meta_files()
        self._load_all()

    # -- file lifecycle ----------------------------------------------------

    def _ensure_meta_files(self) -> None:
        """Create default metadata files if they don't exist."""
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        self._ensure("checkpoints.json", {"checkpoints": {}})
        self._ensure("loras.json", {"loras": {}})
        self._ensure("embeddings.json", {})

    def _ensure(self, filename: str, default: dict) -> None:
        path = self._meta_dir / filename
        if not path.exists():
            path.write_text(json.dumps(default, indent=2))

    # -- loading -----------------------------------------------------------

    def _load_all(self) -> None:
        self._checkpoints = self._load_json("checkpoints.json", self._parse_checkpoints)
        self._loras = self._load_json("loras.json", self._parse_loras)
        self._embeddings = self._load_json("embeddings.json", self._parse_embeddings)

    def _load_json(self, filename: str, parser) -> dict:
        path = self._meta_dir / filename
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return parser(data)

    @staticmethod
    def _parse_checkpoints(data: dict) -> dict[str, _CheckpointMeta]:
        items = data.get("checkpoints", data)
        result: dict[str, _CheckpointMeta] = {}
        for name, meta in items.items():
            result[name] = _CheckpointMeta(
                tags=list(meta.get("tags", []) or []),
                families=list(meta.get("families", []) or []),
                recommended_mode=meta.get("recommended_mode"),
                url=meta.get("url", ""),
                description=meta.get("description", ""),
            )
        return result

    @staticmethod
    def _parse_loras(data: dict) -> dict[str, _LoraMeta]:
        items = data.get("loras", {})
        result: dict[str, _LoraMeta] = {}
        for name, meta in items.items():
            result[name] = _LoraMeta(
                tags=list(meta.get("tags", []) or []),
                aliases=list(meta.get("aliases", []) or []),
                modes=list(meta.get("modes", []) or []),
                weight_hint=meta.get("weight_hint"),
                url=meta.get("url", ""),
                trigger=meta.get("trigger", ""),
            )
        return result

    @staticmethod
    def _parse_embeddings(data: dict) -> dict[str, _EmbeddingMeta]:
        result: dict[str, _EmbeddingMeta] = {}
        for name, meta in data.items():
            result[name] = _EmbeddingMeta(
                tags=list(meta.get("tags", []) or []),
                modes=list(meta.get("modes", []) or []),
                families=list(meta.get("families", []) or []),
                url=meta.get("url", ""),
                description=meta.get("description", ""),
                disabled=bool(meta.get("disabled", False)),
                disabled_reason=meta.get("disabled_reason", ""),
            )
        return result

    # -- persist -----------------------------------------------------------

    def _save_checkpoints(self) -> None:
        payload = {"checkpoints": {
            name: {k: v for k, v in asdict(m).items() if v}
            for name, m in self._checkpoints.items()
        }}
        (self._meta_dir / "checkpoints.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )

    def _save_loras(self) -> None:
        payload = {"loras": {
            name: {k: v for k, v in asdict(m).items() if v or k == "weight_hint"}
            for name, m in self._loras.items()
        }}
        (self._meta_dir / "loras.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )

    def _save_embeddings(self) -> None:
        payload = {
            name: {k: v for k, v in asdict(m).items() if v or k in ("disabled", "disabled_reason")}
            for name, m in self._embeddings.items()
        }
        (self._meta_dir / "embeddings.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )

    # -- reload -----------------------------------------------------------

    def reload(self) -> None:
        """Re-read metadata files from disk."""
        self._checkpoints.clear()
        self._loras.clear()
        self._embeddings.clear()
        self._load_all()

    # -- query helpers ----------------------------------------------------

    def _merge_checkpoint(self, name: str, registry_info: dict | None = None) -> AssetInfo:
        ph = self._checkpoints.get(name)
        return AssetInfo(
            name=name,
            type="checkpoint",
            arch=registry_info.get("arch") if registry_info else None,
            source=registry_info.get("source") if registry_info else None,
            path=str(registry_info["path"]) if registry_info and "path" in registry_info else None,
            tags=list(ph.tags) if ph else [],
            families=list(ph.families) if ph else [],
            description=ph.description if ph else "",
            url=ph.url if ph else "",
            recommended_mode=ph.recommended_mode if ph else None,
            defaults=dict(registry_info.get("defaults", {})) if registry_info else {},
        )

    def _merge_lora(self, name: str, registry_info: dict | None = None) -> AssetInfo:
        ph = self._loras.get(name)
        return AssetInfo(
            name=name,
            type="lora",
            arch=registry_info.get("arch") if registry_info else None,
            source=registry_info.get("source") if registry_info else None,
            path=str(registry_info["path"]) if registry_info and "path" in registry_info else None,
            tags=list(ph.tags) if ph else [],
            aliases=list(ph.aliases) if ph else [],
            modes=list(ph.modes) if ph else [],
            weight_hint=ph.weight_hint if ph else None,
            url=ph.url if ph else "",
            trigger=ph.trigger if (ph and ph.trigger) else (registry_info.get("trigger") if registry_info else None),
            description=registry_info.get("description", "") if registry_info else "",
        )

    def _merge_embedding(self, name: str, registry_info: dict | None = None) -> AssetInfo:
        ph = self._embeddings.get(name)
        return AssetInfo(
            name=name,
            type="embedding",
            arch=registry_info.get("arch") if registry_info else None,
            source=registry_info.get("source") if registry_info else None,
            path=str(registry_info["path"]) if registry_info and "path" in registry_info else None,
            tags=list(ph.tags) if ph else [],
            modes=list(ph.modes) if ph else [],
            families=list(ph.families) if ph else [],
            description=ph.description if ph else "",
            url=ph.url if ph else "",
            token=registry_info.get("token") if registry_info else None,
            disabled=ph.disabled if ph else False,
            disabled_reason=ph.disabled_reason if ph else "",
        )

    # -- public query API -------------------------------------------------

    def all_assets(self) -> list[AssetInfo]:
        """Return every asset known to WebbDuck, enriched with rich metadata."""
        return self.checkpoints() + self.loras() + self.embeddings()

    def checkpoints(self) -> list[AssetInfo]:
        from models.registry import MODEL_REGISTRY
        return [self._merge_checkpoint(name, info) for name, info in MODEL_REGISTRY.items()]

    def loras(self) -> list[AssetInfo]:
        from models.registry import LORA_REGISTRY
        return [self._merge_lora(name, info) for name, info in LORA_REGISTRY.items()]

    def embeddings(self) -> list[AssetInfo]:
        from models.registry import EMBEDDING_REGISTRY
        return [self._merge_embedding(name, info) for name, info in EMBEDDING_REGISTRY.items()]

    def get(self, name: str, asset_type: str | None = None) -> AssetInfo | None:
        """Look up a single asset by name, optionally scoped to a type."""
        if asset_type is None or asset_type == "checkpoint":
            from models.registry import MODEL_REGISTRY
            if name in MODEL_REGISTRY:
                return self._merge_checkpoint(name, MODEL_REGISTRY[name])
        if asset_type is None or asset_type == "lora":
            from models.registry import LORA_REGISTRY
            if name in LORA_REGISTRY:
                return self._merge_lora(name, LORA_REGISTRY[name])
        if asset_type is None or asset_type == "embedding":
            from models.registry import EMBEDDING_REGISTRY
            if name in EMBEDDING_REGISTRY:
                return self._merge_embedding(name, EMBEDDING_REGISTRY[name])
        return None

    # -- search / filter --------------------------------------------------

    def search(
        self,
        *,
        query: str = "",
        asset_type: str | None = None,
        tag: str | None = None,
        mode: str | None = None,
        family: str | None = None,
        compatible_with: str | None = None,
    ) -> list[AssetInfo]:
        """Flexible search across all assets.

        Parameters
        ----------
        query : str
            Fuzzy match against name, description, aliases, and tags.
        asset_type : str or None
            One of ``"checkpoint"``, ``"lora"``, ``"embedding"``, or None for all.
        tag : str or None
            Assets must have this tag.
        mode : str or None
            Assets must be compatible with this mode (includes synonym matching).
        family : str or None
            Assets must belong to this family.
        compatible_with : str or None
            Name of a checkpoint; return only assets whose arch/families match.
        """
        candidates: list[AssetInfo] = []

        if asset_type in (None, "checkpoint"):
            candidates.extend(self.checkpoints())
        if asset_type in (None, "lora"):
            candidates.extend(self.loras())
        if asset_type in (None, "embedding"):
            candidates.extend(self.embeddings())

        result = candidates

        if query:
            q = query.lower()
            result = [
                a for a in result
                if q in a.name.lower()
                or q in a.description.lower()
                or any(q in alias.lower() for alias in a.aliases)
                or any(q in t.lower() for t in a.tags)
            ]

        if tag:
            t = tag.lower()
            result = [a for a in result if any(t == at.lower() for at in a.tags)]

        if mode:
            synonyms = MODE_SYNONYMS.get(mode.lower(), {mode.lower()})
            result = [
                a for a in result
                if any(m in synonyms for m in [mm.lower() for mm in a.modes])
                or (a.recommended_mode and a.recommended_mode.lower() in synonyms)
            ]

        if family:
            f = family.lower()
            result = [a for a in result if any(f == ff.lower() for ff in a.families)]

        if compatible_with:
            cp = self.get(compatible_with, asset_type="checkpoint")
            if cp:
                arch = cp.arch
                cpf = {ff.lower() for ff in cp.families}
                result = [
                    a for a in result
                    if a.type in ("lora", "embedding")
                    and (
                        (arch and a.arch == arch)
                        or (cpf and cpf & {ff.lower() for ff in a.families})
                    )
                ]

        return result

    def compatible(self, checkpoint_name: str) -> dict[str, list[AssetInfo]]:
        """Return compatible LoRAs and embeddings for the given checkpoint."""
        cp = self.get(checkpoint_name, asset_type="checkpoint")
        if not cp:
            return {"loras": [], "embeddings": []}

        arch = cp.arch
        families = {ff.lower() for ff in cp.families}

        compatible_loras: list[AssetInfo] = []
        compatible_embeddings: list[AssetInfo] = []

        for lora in self.loras():
            if arch and lora.arch == arch:
                compatible_loras.append(lora)
            elif families and families & {ff.lower() for ff in lora.families}:
                compatible_loras.append(lora)

        for emb in self.embeddings():
            if emb.disabled:
                continue
            if arch and emb.families:
                if any(ff.lower() in families or ff.lower() == arch for ff in emb.families):
                    compatible_embeddings.append(emb)
            elif not emb.families:
                compatible_embeddings.append(emb)

        return {"loras": compatible_loras, "embeddings": compatible_embeddings}

    def recommended(self, checkpoint_name: str, mode: str = "") -> dict[str, list[dict]]:
        """Return scored recommendations for a checkpoint + optional mode."""
        cp = self.get(checkpoint_name, asset_type="checkpoint")
        if not cp:
            return {"loras": [], "embeddings": [], "checkpoints": []}

        tags = {t.lower() for t in cp.tags}
        families = {ff.lower() for ff in cp.families}
        recommended_mode = mode or cp.recommended_mode or ""

        # Score LoRAs
        lora_results: list[dict] = []
        for lora in self.loras():
            if lora.arch and cp.arch and lora.arch != cp.arch:
                continue
            score = 0.0
            reasons: list[str] = []

            meta_tags = {t.lower() for t in lora.tags}
            overlap = meta_tags & tags
            if overlap:
                score += len(overlap) * 3.0
                reasons.append(f"tags: {', '.join(sorted(overlap)[:2])}")

            if recommended_mode and lora.modes:
                rml = recommended_mode.lower()
                if rml in [m.lower() for m in lora.modes]:
                    score += 4.0
                    reasons.append(f"mode: {recommended_mode}")
                elif "all" in [m.lower() for m in lora.modes]:
                    score += 0.5

            if score > 0:
                weight = (
                    lora.weight_hint
                    if lora.weight_hint is not None
                    else (float(lora.defaults.get("weight", 1.0)) if "weight" in lora.defaults else 1.0)
                )
                lora_results.append({
                    "name": lora.name,
                    "score": round(score, 1),
                    "reason": ", ".join(reasons[:2]) or "arch match",
                    "weight": weight,
                })

        lora_results.sort(key=lambda x: x["score"], reverse=True)

        # Score embeddings
        emb_results: list[dict] = []
        for emb in self.embeddings():
            if emb.disabled:
                continue
            score = 0.0
            reasons: list[str] = []

            meta_tags = {t.lower() for t in emb.tags}
            overlap = meta_tags & tags
            if overlap:
                score += len(overlap) * 2.0
                reasons.append(f"tags: {', '.join(sorted(overlap)[:2])}")

            if recommended_mode and emb.modes:
                rml = recommended_mode.lower()
                if rml in [m.lower() for m in emb.modes]:
                    score += 3.0
                    reasons.append(f"mode: {recommended_mode}")
                elif "all" in [m.lower() for m in emb.modes]:
                    score += 0.5

            if families and emb.families:
                ef = {ff.lower() for ff in emb.families}
                if families & ef:
                    score += 2.0
                    reasons.append("family match")

            if score > 0:
                emb_results.append({
                    "name": emb.name,
                    "score": round(score, 1),
                    "reason": ", ".join(reasons[:2]) or "compatible",
                })

        emb_results.sort(key=lambda x: x["score"], reverse=True)

        # Score similar checkpoints
        cp_results: list[dict] = []
        for other in self.checkpoints():
            if other.name == checkpoint_name:
                continue
            score = 0.0
            reasons: list[str] = []

            other_families = {ff.lower() for ff in other.families}
            fmatch = families & other_families
            if fmatch:
                score += len(fmatch) * 5.0
                reasons.append(", ".join(sorted(fmatch)[:2]))

            if recommended_mode and other.recommended_mode:
                if other.recommended_mode.lower() == recommended_mode.lower():
                    score += 3.0
                    reasons.append(f"same mode: {recommended_mode}")

            if other.arch and cp.arch and other.arch == cp.arch:
                score += 1.0
                reasons.append(f"arch: {cp.arch}")

            if score > 0:
                cp_results.append({
                    "name": other.name,
                    "score": round(score, 1),
                    "reason": ", ".join(reasons[:2]),
                })

        cp_results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "loras": lora_results[:8],
            "embeddings": emb_results[:8],
            "checkpoints": cp_results[:5],
        }

    def categories(self) -> dict[str, list[str]]:
        """Return aggregate tag/mode/family lists for UI filtering.

        Gathers from the rich metadata store (even for assets not currently
        present in the runtime registries) so that frontend filters show all
        known categories.
        """
        all_tags: set[str] = set()
        all_modes: set[str] = set()
        all_families: set[str] = set()

        # From metadata store (covers everything, even without runtime registry)
        for meta in self._checkpoints.values():
            all_tags.update(meta.tags)
            all_families.update(meta.families)
            if meta.recommended_mode:
                all_modes.add(meta.recommended_mode)
        for meta in self._loras.values():
            all_tags.update(meta.tags)
            all_modes.update(meta.modes)
        for meta in self._embeddings.values():
            all_tags.update(meta.tags)
            all_modes.update(meta.modes)
            all_families.update(meta.families)

        # Also fold in runtime registry assets (adds arch-specific tags)
        for a in self.all_assets():
            all_tags.update(a.tags)
            all_modes.update(a.modes)
            all_families.update(a.families)
            if a.recommended_mode:
                all_modes.add(a.recommended_mode)

        return {
            "tags": sorted(all_tags),
            "modes": sorted(all_modes),
            "families": sorted(all_families),
        }

    # -- write helpers (for future API write endpoints) --------------------

    def update_checkpoint_meta(self, name: str, **kwargs) -> bool:
        """Create or update rich metadata for a checkpoint.

        Accepts the same fields as ``_CheckpointMeta``: ``tags``, ``families``,
        ``recommended_mode``, ``url``, ``description``.
        Returns True if something changed.
        """
        existing = self._checkpoints.get(name)
        new_tags = list(kwargs.get("tags", existing.tags if existing else []))
        new_families = list(kwargs.get("families", existing.families if existing else []))
        changed = (
            existing is None
            or new_tags != existing.tags
            or new_families != existing.families
            or kwargs.get("recommended_mode", existing.recommended_mode) != existing.recommended_mode
            or kwargs.get("url", existing.url) != existing.url
            or kwargs.get("description", existing.description) != existing.description
        )
        if changed:
            self._checkpoints[name] = _CheckpointMeta(
                tags=new_tags,
                families=new_families,
                recommended_mode=kwargs.get("recommended_mode", existing.recommended_mode if existing else None),
                url=kwargs.get("url", existing.url if existing else ""),
                description=kwargs.get("description", existing.description if existing else ""),
            )
            self._save_checkpoints()
        return changed

    def update_lora_meta(self, name: str, **kwargs) -> bool:
        """Create or update rich metadata for a LoRA."""
        existing = self._loras.get(name)
        new_tags = list(kwargs.get("tags", existing.tags if existing else []))
        changed = (
            existing is None
            or new_tags != existing.tags
            or list(kwargs.get("aliases", existing.aliases if existing else [])) != existing.aliases
            or list(kwargs.get("modes", existing.modes if existing else [])) != existing.modes
            or kwargs.get("weight_hint", existing.weight_hint) != existing.weight_hint
            or kwargs.get("url", existing.url) != existing.url
            or kwargs.get("trigger", existing.trigger) != existing.trigger
        )
        if changed:
            self._loras[name] = _LoraMeta(
                tags=new_tags,
                aliases=list(kwargs.get("aliases", existing.aliases if existing else [])),
                modes=list(kwargs.get("modes", existing.modes if existing else [])),
                weight_hint=kwargs.get("weight_hint", existing.weight_hint if existing else None),
                url=kwargs.get("url", existing.url if existing else ""),
                trigger=kwargs.get("trigger", existing.trigger if existing else ""),
            )
            self._save_loras()
        return changed

    def update_embedding_meta(self, name: str, **kwargs) -> bool:
        """Create or update rich metadata for an embedding."""
        existing = self._embeddings.get(name)
        changed = (
            existing is None
            or list(kwargs.get("tags", existing.tags if existing else [])) != existing.tags
            or list(kwargs.get("modes", existing.modes if existing else [])) != existing.modes
            or list(kwargs.get("families", existing.families if existing else [])) != existing.families
            or kwargs.get("url", existing.url) != existing.url
            or kwargs.get("description", existing.description) != existing.description
            or kwargs.get("disabled", existing.disabled) != existing.disabled
        )
        if changed:
            self._embeddings[name] = _EmbeddingMeta(
                tags=list(kwargs.get("tags", existing.tags if existing else [])),
                modes=list(kwargs.get("modes", existing.modes if existing else [])),
                families=list(kwargs.get("families", existing.families if existing else [])),
                url=kwargs.get("url", existing.url if existing else ""),
                description=kwargs.get("description", existing.description if existing else ""),
                disabled=bool(kwargs.get("disabled", existing.disabled if existing else False)),
                disabled_reason=kwargs.get("disabled_reason", existing.disabled_reason if existing else ""),
            )
            self._save_embeddings()
        return changed

    def remove_asset_meta(self, name: str, asset_type: str) -> bool:
        """Remove metadata for a single asset.  Does not touch registry data."""
        store = {
            "checkpoint": self._checkpoints,
            "lora": self._loras,
            "embedding": self._embeddings,
        }.get(asset_type)
        if store and name in store:
            del store[name]
            {
                "checkpoint": self._save_checkpoints,
                "lora": self._save_loras,
                "embedding": self._save_embeddings,
            }[asset_type]()
            return True
        return False

    # -- serialisation ----------------------------------------------------

    def to_serialisable(self, assets: list[AssetInfo] | None = None) -> list[dict]:
        """Convert AssetInfo objects to JSON-safe dicts."""
        if assets is None:
            assets = self.all_assets()
        return [asdict(a) for a in assets]

    def export_json(self, path: str | Path | None = None) -> dict:
        """Export the full store as a JSON-serialisable dict.

        Writes to *path* if provided; always returns the dict.
        """
        payload = {
            "version": 2,
            "meta_dir": str(self._meta_dir),
            "assets": self.to_serialisable(),
            "categories": self.categories(),
        }
        if path:
            Path(path).write_text(json.dumps(payload, indent=2, default=str))
        return payload

    # -- diagnostics ------------------------------------------------------

    def stats(self) -> dict:
        """Return summary statistics about the store."""
        from models.registry import MODEL_REGISTRY, LORA_REGISTRY, EMBEDDING_REGISTRY
        return {
            "meta_dir": str(self._meta_dir),
            "checkpoints": {
                "registry": len(MODEL_REGISTRY),
                "with_metadata": sum(1 for n in MODEL_REGISTRY if n in self._checkpoints),
            },
            "loras": {
                "registry": len(LORA_REGISTRY),
                "with_metadata": sum(1 for n in LORA_REGISTRY if n in self._loras),
            },
            "embeddings": {
                "registry": len(EMBEDDING_REGISTRY),
                "with_metadata": sum(1 for n in EMBEDDING_REGISTRY if n in self._embeddings),
            },
        }


# ---------------------------------------------------------------------------
#  Singleton
# ---------------------------------------------------------------------------

meta_store: MetaStore = MetaStore()


def reload_meta() -> None:
    """Reload metadata files from disk."""
    meta_store.reload()
