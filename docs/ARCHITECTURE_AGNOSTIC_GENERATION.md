# Architecture-Agnostic Generation Design

Status: **implementation design / migration target**

This document defines the intended generation architecture for WebbDuck. It does not describe the current implementation as already complete. The current application is still substantially SDXL-shaped internally; this plan exists to guide the refactor without changing the product into an "engine picker."

## Product Contract

The user selects a **model/checkpoint**. WebbDuck figures out everything else.

The normal Studio workflow must not ask the user to select an architecture, model family, pipeline class, runtime backend, or execution environment.

```text
User selects model
      |
      v
Model registry + introspection
      |
      v
Model descriptor
      |
      v
Backend resolver
      |
      v
Capability-aware request normalization
      |
      v
Runtime session / backend adapter
      |
      v
Generated image(s)
```

Examples of internal details that should remain hidden in the normal UI:

- SDXL vs FLUX vs Krea 2 vs a future architecture
- UNet vs diffusion transformer
- CLIP vs Qwen/Gemma/other text encoders
- Diffusers vs an official/reference runtime
- in-process vs isolated subprocess execution
- BF16 vs FP16 vs FP8/INT8 implementation choices

A power/debug view may expose those details for diagnostics, but they are not part of the primary generation workflow.

## Core Principles

1. **The model is the routing key.** The selected checkpoint/model determines the architecture and backend automatically.
2. **Architecture is internal metadata.** It is useful for compatibility and backend resolution, not as a required user choice.
3. **Capabilities drive behavior.** Controls and workflows are enabled from the selected model's capabilities rather than from hard-coded assumptions.
4. **Model-specific defaults win.** Recommended settings can differ between checkpoints using the same architecture.
5. **Architecture-specific code stays behind adapters.** Request handlers, the worker, storage, gallery code, and normal UI orchestration should not branch on `sdxl`, `flux`, `krea2`, etc.
6. **Runtime implementation is replaceable.** A model can move from one loader/runtime to another without changing the public generation contract.
7. **Dependency isolation is allowed and expected.** A new model must not require destabilizing the known-good environment used by existing models.
8. **Existing models remain first-class.** The refactor is not a migration away from SDXL/Pony/Illustrious; it removes their architectural privilege while preserving their workflows.
9. **Backward compatibility matters.** Existing API clients and saved metadata should keep working while neutral aliases are introduced.
10. **Unknown is better than wrong.** If model architecture cannot be identified confidently, report it as unsupported/unknown with diagnostics instead of silently treating it as SDXL.

## Audit: Current WebbDuck Coupling

The current repository already has several reusable generic pieces, but generation is still SDXL-first internally.

### Reusable seams

These should be preserved and generalized rather than replaced:

- `core/worker.py` already provides a queue-first GPU execution boundary.
- `core/gpu_lease.py` already coordinates GPU ownership with plugins.
- `core/runtime.py` already owns device/dtype resolution.
- `server/storage.py` mostly persists arbitrary generation settings and does not require a specific diffusion architecture.
- `models/metastore.py` already separates rich semantic metadata from runtime registry data.
- The `/models` UI workflow already starts from a selected model rather than a separate architecture selector.
- Web plugin isolation already provides a pattern for large optional runtimes.

### SDXL assumptions that must move behind an adapter

#### `models/registry.py`

Current discovery behavior is architecture-biased:

- the preferred checkpoint root resolves through `checkpoint/sdxl` / `checkpoints/sdxl`;
- local Diffusers discovery identifies a directory by finding `unet/config.json`;
- every local single-file `.safetensors` checkpoint is currently classified as SDXL;
- `detect_arch()` only has limited SDXL/FLUX/SD1.5 logic;
- LoRA and embedding compatibility primarily reduces to an `arch` string comparison.

This must become model introspection that can identify transformer-based models, single-file variants, and future formats without assuming a UNet layout.

#### `core/pipeline.py`

This module is currently an SDXL runtime implementation, not a generic pipeline manager. It directly owns:

- `StableDiffusionXLPipeline` / img2img / inpaint pipeline construction;
- `UNet2DConditionModel` loading and caches;
- dual CLIP tokenizer/text-encoder loading;
- the SDXL VAE;
- SDXL scheduler handling;
- textual inversion behavior;
- SDXL LoRA behavior;
- second-pass SDXL pipeline construction;
- `PipelineManager` fields such as `pipe`, `img2img`, `base_img2img`, and `base_inpaint`.

Most of this code can remain, but it should become the implementation of an SDXL backend rather than the shape of WebbDuck itself.

#### `core/generation.py` and `modes/`

`run_generation()` currently asks the global SDXL `pipeline_manager` for several concrete pipelines and then calls `select_mode(settings, pipe, img2img, base_img2img, base_inpaint)`.

The mode interface is therefore coupled to the pipeline arrangement of SDXL. Smart Extend, inpaint, img2img, and two-pass logic are valuable and should be preserved, but they should initially live inside/alongside the SDXL adapter instead of defining the universal backend interface.

#### `server/app.py`

Current API/UI coupling includes:

- `/models` only returning `name`, `type`, and `defaults`;
- `/models/{model}/loras` and `/embeddings` filtering by exact architecture string;
- `/second_pass_models` treating every discovered checkpoint as a candidate;
- `/tokenize` importing the SDXL tokenizer helper and assuming a 77-token CLIP model;
- second-pass validation explicitly requiring SDXL;
- idle unloading and `/health` introspecting the concrete global `pipeline_manager`.

These should become descriptor/capability/runtime-session queries.

#### Catalog watching

The catalog watcher skips a fixed list of known SDXL component directories while creating its filesystem signature. Transformer-oriented checkpoints introduce other large component directories. Catalog signature generation must use model-library-aware boundaries rather than recursively walking arbitrary model components.

#### UI

The UI contains controls that are meaningful for the current SDXL workflows but are not universal model concepts: second pass, CLIP-specific token accounting, embeddings, SDXL scheduler choices, and SDXL-style inpaint/outpaint behavior.

The correct response is not separate architecture pages. The selected model's profile should tell the existing Studio which controls are available, unavailable, constrained, or have different defaults.

## Target Domain Model

### `ModelDescriptor`

Every discovered model should resolve to one normalized descriptor before it can be selected for generation.

Illustrative schema:

```python
@dataclass(frozen=True)
class ModelDescriptor:
    id: str
    name: str
    path_or_repo: str
    source: str                 # local, hf_cache, remote_repo, etc.
    media_type: str             # image
    artifact_format: str        # diffusers, single_file, component_pack, ...

    architecture: str | None    # internal compatibility key
    variant: str | None         # turbo/distilled/base/etc. when detectable
    detection_confidence: str   # exact, strong, heuristic, unknown
    diagnostics: tuple[str, ...]

    capabilities: ModelCapabilities
    defaults: dict
    constraints: dict
    runtime_hints: dict
    compatibility: dict
```

`architecture` is intentionally not the primary API/UI concept. It is a resolver/compatibility key.

The descriptor should be serializable so a safe subset can be returned to the frontend and a snapshot can be written into generation metadata.

### Capability model

Capabilities should describe user-visible behavior rather than implementation details.

Example:

```json
{
  "operations": {
    "text_to_image": true,
    "image_to_image": true,
    "inpaint": false,
    "outpaint": false,
    "multi_reference": true
  },
  "assets": {
    "lora": true,
    "embedding": false,
    "identity_adapter": false,
    "second_pass": false
  },
  "controls": {
    "negative_prompt": false,
    "guidance": true,
    "scheduler": false,
    "clip_skip": false,
    "strength": true,
    "mask": false
  }
}
```

Capabilities should support richer states later (`supported`, `unsupported`, `required`, `experimental`) without forcing a public architecture change.

### Constraints

Capabilities answer "can it do this?" Constraints answer "what values are valid?"

Examples:

- width/height multiples and min/max values;
- maximum pixel area;
- supported/default step ranges;
- frame/reference counts for future multimodal image models;
- allowed dtypes/runtime modes;
- whether negative prompting is meaningful;
- whether guidance is fixed/ignored for a distilled model.

### Defaults

Effective settings should be derived in this order:

1. explicit user request;
2. checkpoint-specific saved/recommended defaults;
3. descriptor/backend defaults for that exact model variant;
4. WebbDuck generic fallback.

When switching models in the UI, WebbDuck should not blindly destroy deliberate user choices. A control should be replaced when it was still using the previous model's derived default, when the new model marks the old value invalid, or when the user explicitly asks to load defaults.

## Model Introspection

Model discovery and model identification are separate concerns.

### Discovery

Discovery finds candidate assets from configured roots/cache locations. It should not need to know how to run them.

Preferred architecture-neutral local layout for new installs:

```text
<models-root>/
  checkpoints/
    image/
      <any nested organization>
```

Legacy roots such as `checkpoint/sdxl` must remain scanned for compatibility. No forced directory migration is required.

### Introspection

An ordered detector chain should inspect candidates and produce a descriptor.

Suggested signals, strongest first:

1. explicit local manifest/override maintained by WebbDuck;
2. Diffusers `model_index.json` pipeline/component classes;
3. component `config.json` architecture/model types;
4. safetensors metadata;
5. safetensors tensor-key signatures;
6. filename/path heuristics only as a low-confidence fallback.

Detection should be extensible:

```python
class ModelDetector(Protocol):
    def inspect(self, candidate: ModelCandidate) -> DetectionResult | None: ...
```

New architectures should normally add a detector/plugin registration rather than edit a monolithic chain of conditionals.

### Unknown models

A candidate that cannot be identified safely may still appear in diagnostics/catalog administration, but should not be presented as a normal runnable model. The descriptor should explain what evidence was missing.

## Backend Resolution

### Backend contract

The central abstraction is a generation backend, not an architecture selector.

Illustrative interface:

```python
class ImageGenerationBackend(Protocol):
    backend_id: str

    def accepts(self, model: ModelDescriptor) -> bool: ...
    def validate(self, model: ModelDescriptor, request: GenerationRequest) -> None: ...
    def load(self, model: ModelDescriptor, runtime: RuntimeProfile) -> RuntimeSession: ...
    def generate(
        self,
        session: RuntimeSession,
        request: GenerationRequest,
        progress,
        cancel_event,
    ) -> GenerationResult: ...
    def unload(self, session: RuntimeSession, *, clear_caches: bool = False) -> None: ...
```

Optional backend services may include token/prompt inspection, adapter compatibility, or model-specific preflight estimation. They must be optional capabilities, not assumptions made by the server.

### Backend registry and resolver

Backends register predicates/priorities. The resolver picks the safest compatible implementation for a descriptor and current runtime.

```text
ModelDescriptor
   |
   +--> compatible backends
            |
            +--> installed?
            +--> runtime compatible?
            +--> required dependencies available?
            +--> local optimized variant available?
            |
            v
        selected backend
```

Do not spread architecture branching into callers. Code such as the following is a design failure outside discovery/resolver/adapter modules:

```python
if model.architecture == "sdxl": ...
elif model.architecture == "flux": ...
elif model.architecture == "krea2": ...
```

### `RuntimeSession`

Replace the assumption of one global SDXL `PipelineManager` shape with an opaque session owned by the selected backend.

```python
@dataclass
class RuntimeSession:
    model_id: str
    backend_id: str
    cache_key: str
    resources: Any
    loaded_at: float
    last_used_at: float
```

`resources` is backend-owned. SDXL can internally keep text2img/img2img/inpaint pipelines while another backend may hold one transformer pipeline or a subprocess handle.

The core worker only needs generic load/generate/unload behavior.

## Execution Modes and Dependency Isolation

A backend may execute:

- in the WebbDuck process;
- in an isolated child process using a separate Python environment;
- eventually through a local service process.

This is an implementation detail selected by the backend resolver.

This matters because new architectures can require versions of Diffusers, Transformers, Python, CUDA extensions, or quantization libraries that conflict with the stable SDXL stack. Supporting a new checkpoint must not require a global dependency upgrade as the first choice.

The GPU lease remains the authority for GPU ownership even when generation happens in a child process. Child processes need progress reporting, cancellation semantics, error normalization, and lease heartbeats comparable to the existing DuckMotion isolation pattern.

## Request and Result Contracts

### `GenerationRequest`

The server should normalize form/API input into a generic request before handing it to a backend.

Core fields:

- selected model ID;
- prompt;
- optional negative prompt;
- dimensions;
- seed;
- requested image count;
- optional source image(s);
- optional mask;
- generic controls (`steps`, `guidance`, `strength`) when supported;
- selected compatible assets;
- operation intent (`text_to_image`, `image_to_image`, `inpaint`, `outpaint`, etc.).

Architecture-specific settings should live in a namespaced `advanced` structure if they cannot be expressed generically. They must not become mandatory fields on every request.

### `GenerationResult`

Backends return a normalized result containing:

- image list;
- resolved seed(s);
- backend/model descriptor snapshot;
- normalized effective settings;
- performance metadata;
- warnings/diagnostics.

Storage should not need to know which backend produced the result.

## API Evolution

### `/models`

Keep the existing endpoint and existing fields for compatibility, but enrich each entry.

Suggested response shape:

```json
{
  "name": "example-model",
  "type": "diffusers",
  "defaults": {},
  "capabilities": {},
  "constraints": {},
  "availability": "ready"
}
```

Architecture/backend details may be present in an optional diagnostics object but are not required by the normal UI.

A dedicated `/models/{name}/profile` endpoint is appropriate if the public catalog payload becomes too large.

### Generation endpoints

Keep accepting `base_model` during migration because current clients use it. Internally normalize it to `model_id`. A neutral `model`/`model_id` alias can be added without breaking old clients.

Request validation should happen against the selected model profile before queue execution whenever possible.

### Assets

`/models/{model}/loras` and `/embeddings` should ask a compatibility service rather than compare one architecture string.

Compatibility may eventually include:

- architecture;
- base model lineage;
- transformer/text-encoder format;
- adapter target modules;
- model family metadata;
- explicit allow/deny overrides.

### Tokenization/prompt diagnostics

Tokenization is not universal. The UI should request prompt diagnostics from the selected model/backend only when supported. Do not emulate a 77-token CLIP limit for models that do not use it.

### Second pass

Second-pass/refiner behavior is a capability of a selected model/backend workflow, not a global assumption. Existing SDXL behavior should continue under the SDXL adapter.

## UI Behavior

There must be no required architecture/engine selector.

When a model changes:

1. load the model profile/capabilities;
2. update derived defaults;
3. enable/disable/hide controls based on capabilities;
4. clear incompatible selected LoRAs/embeddings/adapters with a visible explanation;
5. keep compatible user-entered prompt/settings;
6. show readiness/unsupported diagnostics if the model cannot currently run;
7. generation submission remains model-centric.

Architecture names may appear in an optional model-details/debug view. They are metadata, not workflow navigation.

## Asset Compatibility

Do not equate stylistic family with architecture.

`Pony`, `Illustrious`, `realistic`, and similar terms can remain useful semantic metadata in `MetaStore`. Runtime compatibility should be based on structural adapter compatibility from the model descriptor, optionally refined by metadata overrides.

The existing metastore remains the canonical home for rich human-curated descriptions/tags/recommendations. Structural runtime information should originate from model introspection/registry data and be merged into the public asset view.

## Initial Adapter Validation Targets

These are implementation validation targets, not special cases in the product model.

### Existing SDXL-compatible library

First prove that wrapping the current implementation behind the generic backend produces no behavior regression for the existing SDXL/Pony/Illustrious checkpoints and workflows.

### FLUX

A FLUX adapter should be the first proof that the architecture boundary is real. As of August 2026, `black-forest-labs/FLUX.2-klein-4B` is a useful consumer-hardware validation target because the official release supports text-to-image and image editing and publishes Diffusers-compatible weights. Other FLUX checkpoints/quantized variants should resolve through the same descriptor/backend mechanism rather than new UI modes.

### Krea 2

`krea/Krea-2-Turbo` is a useful second independent validation target. Its official release is a transformer-based text-to-image model and currently requires newer Diffusers support than WebbDuck's pinned stable stack. It is therefore a good test of both capability-driven UI (text-to-image first) and runtime/dependency isolation.

The architecture must not encode these model IDs. They are integration fixtures/acceptance targets.

## Migration Plan

### Phase 0 - Lock current behavior

- Add focused regression fixtures for current model discovery, `/models`, asset filtering, SDXL text2img/img2img/inpaint/outpaint, LoRA loading, embeddings, second pass, seed behavior, and metadata persistence.
- Capture representative registry entries for existing local single-file and Diffusers models.
- Do not begin by upgrading shared ML dependencies.

### Phase 1 - Introduce descriptors without changing execution

- Add `ModelCandidate`, `ModelDescriptor`, capabilities, constraints, and detection diagnostics.
- Build descriptor output alongside the existing `MODEL_REGISTRY` shape.
- Enrich `/models` while preserving existing fields.
- Generalize model roots and catalog signatures while retaining legacy paths.
- Add detection fixtures for SDXL plus non-SDXL model layouts.

Exit criterion: existing UI and generation still behave the same, but every selectable model has a descriptor.

### Phase 2 - Introduce backend registry and wrap SDXL

- Add backend protocol/registry/resolver.
- Move or wrap current `core/pipeline.py` and `modes/` behavior as the SDXL backend implementation.
- Replace direct `pipeline_manager` access in generation/worker/health/idle-unload with generic runtime-session services.
- Keep compatibility imports/shims temporarily if needed.

Exit criterion: current SDXL models run through the generic backend path with no new architecture selector.

### Phase 3 - Capability-driven server and UI

- Add profile/capability validation.
- Make LoRA/embedding/second-pass selectors compatibility-aware.
- Make token diagnostics optional.
- Make Studio sections respond to model capabilities.
- Persist descriptor/backend snapshots into output metadata for debugging/reproducibility.

Exit criterion: the UI can select a model with fewer capabilities without exposing invalid controls.

### Phase 4 - Add FLUX adapter

- Add FLUX detection fixtures.
- Add backend implementation with isolated runtime available where dependency versions require it.
- Support text-to-image first; expose image editing/reference operations only when the selected checkpoint reports them.
- Add compatible LoRA support after base generation is stable.

Exit criterion: switching between an existing SDXL checkpoint and a FLUX checkpoint works from the same model selector and generation endpoint.

### Phase 5 - Add Krea 2 adapter

- Add Krea 2 detection fixtures.
- Implement text-to-image backend and model-specific defaults/constraints.
- Use isolated dependencies rather than upgrading the SDXL environment by default.
- Add adapter/LoRA support only when compatibility is verified.

Exit criterion: Krea 2 appears as another model in the same selector and automatically changes only the controls/defaults it needs.

### Phase 6 - Quantized/single-file hardening

- Add explicit format handlers for supported FP8/INT8/GGUF/other community variants.
- Keep quantization format separate from architecture in descriptors.
- Prefer safe unsupported diagnostics over guessing.
- Add model-specific runtime selection based on installed loaders and hardware.

### Phase 7 - Remove legacy SDXL-first naming internally

After compatibility shims are no longer required:

- neutralize checkpoint-root defaults/documentation;
- remove direct SDXL pipeline assumptions from core/server code;
- update project docs from "SDXL studio" to architecture-agnostic image generation studio;
- keep SDXL-specific docs only inside the SDXL backend area.

## Proposed File Boundaries

Names are illustrative; implementation may adjust them while preserving ownership boundaries.

```text
models/
  descriptors.py
  discovery.py
  introspection/
    base.py
    diffusers.py
    safetensors.py
    overrides.py
  registry.py

core/
  backends/
    base.py
    registry.py
    sdxl.py
    flux.py
    krea2.py
  runtime_session.py
  generation.py
  worker.py
```

Potential migration outcome:

- `core/pipeline.py` becomes an SDXL compatibility shim or is absorbed into `core/backends/sdxl.py`.
- `modes/` remains available to the SDXL backend until generic operation abstractions are genuinely useful.
- `prompt/conditioning.py` remains SDXL-specific rather than being forced onto unrelated models.

Do not move code merely to match this tree. Introduce boundaries when they reduce coupling and are covered by tests.

## Implementation Rules for Coding Agents

- Do not add an engine/architecture dropdown to solve backend routing.
- Do not require users/API clients to know a model's architecture.
- Do not classify unknown single-file checkpoints as SDXL by default.
- Do not make a shared Diffusers upgrade the prerequisite for the refactor.
- Do not remove existing SDXL functionality while extracting it behind an adapter.
- Do not make all architectures implement unsupported features just to satisfy a common interface.
- Do not put model-specific conditionals in `server/app.py`, `core/worker.py`, gallery/storage code, or top-level UI orchestration.
- Do add capability/constraint tests for every new backend.
- Do test model switching and resource cleanup, not only isolated generation.
- Do preserve GPU-lease discipline across in-process and subprocess backends.

## Testing Matrix

At minimum, the implementation should cover:

| Area | Required tests |
| --- | --- |
| Discovery | legacy roots, neutral roots, HF cache, Diffusers folders, single files, unknown candidates |
| Detection | exact model-index classes, component configs, tensor signatures, override precedence |
| Descriptor | serialization, defaults, capabilities, constraints, diagnostics |
| Resolver | correct backend selection, missing backend, incompatible runtime, deterministic priority |
| API | enriched `/models`, profile query, compatibility filtering, unsupported-control validation |
| UI | model switch updates controls/defaults without architecture selector |
| SDXL regression | text2img, img2img, inpaint/outpaint, LoRA, embedding, second pass |
| Multi-backend | SDXL -> FLUX -> SDXL switching and unload/VRAM cleanup |
| Isolation | child error propagation, progress, cancellation, timeout, lease release |
| Storage | selected model, descriptor/backend snapshot, effective settings, reproducible seed |

Krea 2 should be added to the multi-backend switching matrix when its adapter lands.

## Cross-Repo Contract With DuckMotion

WebbDuck and DuckMotion should share the same product philosophy without requiring a shared Python implementation.

WebbDuck owns image generation and image model descriptors. DuckMotion owns video generation and video model descriptors. A handoff sends media + generation context, not architecture choices.

Example handoff:

```json
{
  "source_image": "outputs/.../0.png",
  "prompt": "...",
  "source_metadata": {
    "model": "selected-image-model",
    "seed": 1234
  }
}
```

DuckMotion then uses its currently selected video model and its own capabilities to decide whether image-to-video is valid. WebbDuck must not send `engine=ltx` or `engine=wan` as part of the normal handoff.

See the matching DuckMotion architecture plan for video-specific runtime and audio/result behavior.

## Definition of Done

The architecture-agnostic migration is complete when all of the following are true:

- users select models, not architectures/backends;
- existing SDXL/Pony/Illustrious workflows pass regression tests;
- at least two structurally different non-SDXL image architectures run through the same model-centric API/UI flow;
- model capabilities automatically configure Studio controls;
- incompatible assets are filtered by structural compatibility rather than a simplistic family label;
- model-specific defaults are applied without requiring architecture knowledge;
- runtime/backends can be isolated without changing the generation API;
- model switching unloads/loads resources safely under the shared GPU lease;
- unknown models fail with actionable detection diagnostics instead of being misclassified;
- core/server/storage/UI orchestration contain no architecture-specific routing branches outside designated discovery/resolver/adapter modules.
