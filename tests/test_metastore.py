"""Tests for the unified WebbDuck metadata store (models/metastore.py)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from models.metastore import MetaStore, AssetInfo, MODE_SYNONYMS

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

_CHECKPOINTS = {
    "checkpoints": {
        "Juggernaut-XI-v11": {
            "tags": ["realistic", "photo", "quality"],
            "families": ["realistic"],
            "recommended_mode": "realistic",
            "url": "https://civitai.com/models/1",
            "description": "Photorealistic SDXL checkpoint",
        },
        "animagine-xl-3.1": {
            "tags": ["anime", "illustrious", "stylized"],
            "families": ["illustrious"],
            "recommended_mode": "anime",
            "url": "https://civitai.com/models/2",
            "description": "Anime SDXL checkpoint",
        },
        "noobai-v1.0": {
            "tags": ["anime", "illustrious", "stylized", "quality"],
            "families": ["illustrious", "noobai"],
            "recommended_mode": "anime",
            "url": "https://civitai.com/models/3",
            "description": "NoobAI SDXL checkpoint",
        },
    }
}

_LORAS = {
    "loras": {
        "add-detail-xl": {
            "tags": ["detail", "quality", "enhancer"],
            "aliases": ["add detail", "detail enhancer"],
            "modes": ["all"],
            "weight_hint": 0.6,
        },
        "anime-style-v1": {
            "tags": ["style", "anime", "stylized"],
            "aliases": ["anime style"],
            "modes": ["anime"],
            "weight_hint": 0.8,
        },
        "photorealistic-v2": {
            "tags": ["style", "realistic", "photo"],
            "aliases": ["photo style"],
            "modes": ["realistic"],
            "weight_hint": 0.7,
        },
    }
}

_EMBEDDINGS = {
    "pureerosface_xl": {
        "file": "pureerosface_xl.safetensors",
        "token": "pureerosface_xl",
        "tags": ["style", "nsfw", "erotic"],
        "modes": ["anime", "hentai"],
        "families": ["illustrious", "noobai"],
        "url": "https://civitai.com/models/1013445",
        "description": "Style embedding for NSFW content",
    },
    "SimpleNegativeV3": {
        "file": "SimpleNegativeV3.safetensors",
        "token": "SimpleNegativeV3",
        "tags": ["negative", "quality"],
        "modes": ["all"],
        "families": ["sdxl", "pony", "illustrious", "noobai"],
        "url": "https://civitai.com/models/87243",
        "description": "Negative embedding to reduce artifacts",
    },
    "bad_x": {
        "file": "bad_x.safetensors",
        "token": "badX, bad_x",
        "tags": ["negative", "anatomy", "face", "hands"],
        "modes": ["all"],
        "families": ["sdxl", "illustrious"],
        "url": "https://civitai.com/models/122403",
        "description": "Fixes face and hand mutations",
        "disabled": True,
        "disabled_reason": "Causes generation hang on this setup",
    },
}


@pytest.fixture
def meta_dir(tmp_path: Path) -> Path:
    """Create a WebbDuck-owned metadata directory with sample data."""
    d = tmp_path / "webbduck_meta"
    d.mkdir(parents=True, exist_ok=True)
    (d / "checkpoints.json").write_text(json.dumps(_CHECKPOINTS, indent=2))
    (d / "loras.json").write_text(json.dumps(_LORAS, indent=2))
    (d / "embeddings.json").write_text(json.dumps(_EMBEDDINGS, indent=2))
    return d


@pytest.fixture
def store(meta_dir: Path) -> MetaStore:
    return MetaStore(meta_dir=meta_dir)


@pytest.fixture
def empty_store(tmp_path: Path) -> MetaStore:
    """A store with no pre-existing metadata — tests graceful creation of defaults."""
    return MetaStore(meta_dir=tmp_path / "empty_meta")


# ---------------------------------------------------------------------------
#  Loading & file lifecycle
# ---------------------------------------------------------------------------


class TestFileLifecycle:
    def test_creates_default_files(self, tmp_path: Path):
        d = tmp_path / "fresh_meta"
        MetaStore(meta_dir=d)
        assert (d / "checkpoints.json").exists()
        assert (d / "loras.json").exists()
        assert (d / "embeddings.json").exists()

    def test_loads_all_types(self, store: MetaStore):
        assert len(store._checkpoints) == 3
        assert len(store._loras) == 3
        assert len(store._embeddings) == 3

    def test_empty_store_has_no_crash(self, empty_store: MetaStore):
        assert empty_store._checkpoints == {}
        assert empty_store._loras == {}
        assert empty_store._embeddings == {}

    def test_loaded_checkpoint_meta(self, store: MetaStore):
        cp = store._checkpoints["Juggernaut-XI-v11"]
        assert "realistic" in cp.tags
        assert cp.recommended_mode == "realistic"

    def test_loaded_lora_meta(self, store: MetaStore):
        lora = store._loras["add-detail-xl"]
        assert lora.weight_hint == 0.6
        assert "all" in lora.modes

    def test_loaded_embedding_meta_disabled(self, store: MetaStore):
        emb = store._embeddings["bad_x"]
        assert emb.disabled is True
        assert "generation hang" in emb.disabled_reason

    def test_missing_files_no_crash(self):
        # A non-existent dir inside a temp tree — should create files
        d = Path(tempfile.mkdtemp()) / "nonexistent" / "deep"
        ms = MetaStore(meta_dir=d)
        assert ms._checkpoints == {}
        assert ms._loras == {}
        assert ms._embeddings == {}
        assert d.exists()


# ---------------------------------------------------------------------------
#  Reload
# ---------------------------------------------------------------------------


class TestReload:
    def test_reload_picks_up_new_data(self, meta_dir: Path):
        ms = MetaStore(meta_dir=meta_dir)
        assert len(ms._checkpoints) == 3

        data = json.loads((meta_dir / "checkpoints.json").read_text())
        data["checkpoints"]["new-model"] = {
            "tags": ["test"],
            "families": ["test"],
            "description": "Freshly added",
        }
        (meta_dir / "checkpoints.json").write_text(json.dumps(data, indent=2))

        ms.reload()
        assert "new-model" in ms._checkpoints
        assert len(ms._checkpoints) == 4

    def test_reload_handles_deleted_file(self, meta_dir: Path):
        ms = MetaStore(meta_dir=meta_dir)
        assert len(ms._loras) == 3
        (meta_dir / "loras.json").unlink()
        ms.reload()
        assert ms._loras == {}


# ---------------------------------------------------------------------------
#  Query (metadata layer only — registry needs GPU)
# ---------------------------------------------------------------------------


class TestQuery:
    def test_get_nonexistent(self, store: MetaStore):
        assert store.get("nonexistent") is None

    def test_categories(self, store: MetaStore):
        cats = store.categories()
        assert "tags" in cats
        assert "modes" in cats
        assert "families" in cats
        assert "realistic" in cats["tags"]
        assert "illustrious" in cats["families"]

    def test_search_by_tag_no_registry(self, store: MetaStore):
        # Without registry backing (GPU deps), search returns empty because
        # checkpoints()/loras()/embeddings() need MODEL_REGISTRY etc.
        # This test just ensures no crash.
        result = store.search(tag="anime")
        assert isinstance(result, list)

    def test_search_by_missing_tag(self, store: MetaStore):
        result = store.search(tag="nonexistent_tag_xyz")
        assert result == []

    def test_search_with_query(self, store: MetaStore):
        result = store.search(query="nonexistent")
        assert result == []

    def test_search_mode_filter(self, store: MetaStore):
        result = store.search(mode="anime")
        assert isinstance(result, list)

    def test_search_family_filter(self, store: MetaStore):
        result = store.search(family="illustrious")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
#  Mode synonyms
# ---------------------------------------------------------------------------


class TestModeSynonyms:
    def test_anime_includes_hentai(self):
        assert "hentai" in MODE_SYNONYMS["anime"]

    def test_realistic_includes_photo(self):
        assert "photo" in MODE_SYNONYMS["realistic"]

    def test_unknown_mode_returns_self(self):
        assert MODE_SYNONYMS.get("unknown", {"unknown"}) == {"unknown"}


# ---------------------------------------------------------------------------
#  AssetInfo
# ---------------------------------------------------------------------------


class TestAssetInfo:
    def test_default_fields(self):
        a = AssetInfo(name="test", type="lora")
        assert a.tags == []
        assert a.aliases == []
        assert a.modes == []
        assert a.disabled is False

    def test_type_checkpoint_has_all_fields(self):
        a = AssetInfo(
            name="test",
            type="checkpoint",
            arch="sdxl",
            tags=["anime"],
            families=["illustrious"],
            description="An anime model",
            recommended_mode="anime",
        )
        assert a.arch == "sdxl"
        assert a.recommended_mode == "anime"

    def test_serialisation(self):
        a = AssetInfo(name="test", type="lora", tags=["foo"])
        d = Path(tempfile.mkdtemp()) / "meta"
        result = MetaStore(meta_dir=d).to_serialisable([a])
        assert result[0]["name"] == "test"
        assert result[0]["tags"] == ["foo"]


# ---------------------------------------------------------------------------
#  Metadata write helpers
# ---------------------------------------------------------------------------


class TestWriteHelpers:
    def test_update_checkpoint_meta_create(self, store: MetaStore):
        assert "new-cp" not in store._checkpoints
        changed = store.update_checkpoint_meta("new-cp", tags=["test"], families=["test-family"])
        assert changed is True
        assert "new-cp" in store._checkpoints

    def test_update_checkpoint_meta_update(self, store: MetaStore):
        changed = store.update_checkpoint_meta(
            "Juggernaut-XI-v11", description="Updated description"
        )
        assert changed is True
        assert store._checkpoints["Juggernaut-XI-v11"].description == "Updated description"

    def test_update_checkpoint_meta_noop(self, store: MetaStore):
        cp = store._checkpoints["Juggernaut-XI-v11"]
        changed = store.update_checkpoint_meta(
            "Juggernaut-XI-v11",
            tags=cp.tags,
            families=cp.families,
            recommended_mode=cp.recommended_mode,
            url=cp.url,
            description=cp.description,
        )
        assert changed is False

    def test_update_lora_meta_create(self, store: MetaStore):
        changed = store.update_lora_meta("new-lora", tags=["test"], modes=["all"])
        assert changed is True
        assert store._loras["new-lora"].modes == ["all"]

    def test_update_embedding_meta_disable(self, store: MetaStore):
        changed = store.update_embedding_meta(
            "SimpleNegativeV3", disabled=True, disabled_reason="Testing"
        )
        assert changed is True
        assert store._embeddings["SimpleNegativeV3"].disabled is True

    def test_remove_asset_meta(self, store: MetaStore):
        assert "add-detail-xl" in store._loras
        ok = store.remove_asset_meta("add-detail-xl", "lora")
        assert ok is True
        assert "add-detail-xl" not in store._loras

    def test_remove_nonexistent(self, store: MetaStore):
        ok = store.remove_asset_meta("nonexistent", "lora")
        assert ok is False


# ---------------------------------------------------------------------------
#  Export / serialisation
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_json(self, store: MetaStore):
        export = store.export_json()
        assert export["version"] == 2
        assert export["meta_dir"] is not None
        assert "categories" in export

    def test_export_writes_file(self, store: MetaStore, tmp_path: Path):
        out = tmp_path / "export.json"
        store.export_json(path=out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["version"] == 2


# ---------------------------------------------------------------------------
#  Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_structure(self, store: MetaStore):
        s = store.stats()
        assert "meta_dir" in s
        assert "checkpoints" in s
        assert "loras" in s
        assert "embeddings" in s
        assert s["meta_dir"] is not None
