# WebbDuck Technical Requirements Specification

## 1. Project Overview
WebbDuck is a modern, high-performance Web UI for Stable Diffusion XL (SDXL) image generation. It mimics the aesthetics and responsiveness of a native application using modern web technologies. This document serves as the authoritative source of truth for all functional and technical requirements.

## 2. Technical Architecture (Hard Requirements)

### 2.1 Core Stack
- **Frontend**: Vanilla JavaScript (ES6 Modules). No frameworks (React/Vue/Svelte) permitted.
- **Styling**: Vanilla CSS with comprehensive CSS Variables (Design Tokens). No libraries (Tailwind/Bootstrap).
- **Backend API**: Python (FastAPI/Flask compatible).
- **Communication**: REST API for actions, WebSocket for real-time progress.

### 2.2 File Structure
- `ui/core/`: Essential utilities (api, state, events, utils).
- `ui/core/`: Essential utilities (api, state, events, utils).
- `ui/modules/`: Feature-specific classes (MaskEditor, LoraManager, LightboxManager, GalleryManager, ProgressManager).
- `ui/styles/`: Organized CSS architecture (tokens, layouts, components).
- `ui/app.js`: Main entry point and initialization logic.
- `ui/styles/`: Organized CSS architecture (tokens, layouts, components).
- `ui/app.js`: Main entry point and initialization logic.

### 2.3 Browser Support
- Must function on Chrome, Firefox, Edge, and Safari (iOS/macOS).
- Must support Mobile Viewports (< 640px) with dedicated layout logic.

---

## 3. Design System & UX Requirements

### 3.1 Visual Language (Glassmorphism)
- **Backgrounds**: Dark mode only. Layers must use semi-transparent rgba values.
  - Surface: `--bg-surface` (#0f1219)
  - Overlay: `--glass-bg` (rgba(15, 18, 25, 0.75))
- **Backdrop Filters**: All overlays (Modals, Panels) must use `backdrop-filter: blur(Npx)`.
- **Typography**:
  - Primary Font: 'Inter' (Google Fonts).
  - Monospace: 'JetBrains Mono' (for seeds/IDs).
- **Colors**:
  - Primary Accent: Indigo (`--accent-500` #6366f1).
  - Semantic Colors: Green (Success), Orange (Warning), Red (Error).

### 3.2 Z-Index Hierarchy (Strict)
To prevent "stacking context wars", the following scale MUST be adhered to:
- **Base Content**: 0-100
- **Sticky Headers**: 200
- **Fixed Overlays**: 300
- **PhotoSwipe Lightbox**: ~100,000 (External Lib)
- **Modals**: 10,000,000 (`--z-modal`)
- **Toasts**: 10,000,300 (`--z-toast`)
*Rationale*: Modals must appear *above* the full-screen lightbox if triggered from within it.

### 3.3 Transitions
- **Standard Timing**: `200ms cubic-bezier(0.16, 1, 0.3, 1)` (ease-out).
- **Interactions**: All buttons and inputs must have `:hover` and `:active` states.

---

## 4. Functional Requirements (Hard)

### 4.1 Generation Engine
- **Text-to-Image**: Standard generation flow.
- **Image-to-Image**:
  - Must accept drag-and-drop or file selection.
  - Must visually preview the input image.
  - Must expose "Denoising Strength" slider (0.0 - 1.0).
- **Inpainting (Studio Mode)**:
  - **Canvas**: Dedicated canvas for drawing masks over input images.
  - **Brush Tools**: Draw / Erase.
  - **Modes**: "Replace" (Inpaint) vs "Keep" (Reverse Inpaint).
  - **Blur**: Adjustable mask blur radius.

### 4.2 Model Manager
- **Base Models**:
  - Dynamic loading from backend.
  - Dropdown interface.
- **LoRA Support**:
  - Loading LoRAs specific to the selected Base Model.
  - Multiple LoRA selection support (UI dependent).
  - Adjustable weight per LoRA.
- **Second Pass**:
  - Option to run a refiner pass.
  - Independent Step and Denoising controls.

### 4.3 Gallery & History
- **Data Persistence**: History is stored on the filesystem as individual run folders.
- **Loading Strategy**:
  - **limit**: Must fetch only the most recent **50** sessions initially to prevent UI freeze.
  - **Pagination**: "Load More" button to fetch older sessions.
- **Display**:
  - Grouped by "Session" (Batch of images).
  - Grid layout.
  - "Newest First" sorting.
- **Run Deletion**:
  - User can delete an entire run folder from the gallery header.
  - Requires confirmation modal.

### 4.4 Lightbox (Viewer)
- **Integration**: PhotoSwipe (v5) based.
- **Deep Integration Features**:
  - **Zoom**: Wheel/Pinch to zoom.
  - **Upscale**: Action to trigger 2x/4x upscale.
  - **Regenerate**: Action to copy metadata to Studio (details below).
  - **Delete**: Action to delete single image.
- **Metadata Panel**:
  - Overlay showing Prompt, Negative, Seed, Model, Steps, CFG.
  - Must be collapsible.

### 4.5 Performance & Optimization
- **Thumbnails**:
  - Backend must generate reduced-resolution sidecars (`.thumb.jpg`) for the gallery grid.
  - Frontend must lazy-load these thumbnails to ensure 60fps scrolling performance.

### 4.6 Deployment Stability
- **Background Execution**:
  - Server must run robustly in background processes (nohup/systemd).
  - Must explicitly suppress `tqdm` progress bars (`HF_HUB_DISABLE_PROGRESS_BARS=1`) to prevent `BrokenPipeError` crashes in non-interactive shells.

### 4.5 Orchestration Features
- **State Persistence**:
  - All form inputs (Prompt, Steps, CFG, Dims) must auto-save to `localStorage`.
  - UI must restore these values on reload.
- **Regenerate Workflow**:
  - Clicking "Regenerate" from Lightbox must:
    1.  Switch view to Studio.
    2.  Populate all fields from image metadata.
    3.  **Clear the Seed** (to allow variation).
    4.  Auto-scroll to top.
    5.  Trigger "Generate" click (optional but preferred).
- **Send to Inpaint**:
  - Workflow to transfer a generated image from Gallery/Preview -> Inpaint Studio.
  - Must load image as "Input Image".
  - Must reset mask.

---

## 5. Behavior Specifications (Detailed)

### 5.1 Deletion (Optimistic UI)
- **Requirement**: "Instant" feedback.
- **Implementation**:
  1.  User confirms delete.
  2.  UI element (Image Card or Session Group) is removed from DOM **immediately**.
  3.  Lightbox closes **immediately** (if open).
  4.  Backend API call is fired in background (no await).
  5.  Error Toast shown *only* if backend fails (rare).
- **Edge Case**: If deleting the last image in a session, the entire session header must be removed.

### 5.2 Resolution Handling
- **Constraint**: SDXL prefers ~1MP resolutions.
- **Feature**: Preset "Chips" for common aspect ratios (1024x1024, 896x1152, etc.).
- **Auto-Sync**: Uploading an image for Img2Img must auto-resize the Width/Height sliders to match the image aspect ratio (snapped to nearest 8px), maxing out at `2048px` on the longest side.

### 5.3 Error Handling Strategy
- **Initialization**:
  - The `app.js` initialization block must be wrapped in `try-catch`.
  - Critical failures (e.g. `api.js` import fail) show a global `alert()`.
  - Non-critical failures (e.g. `setupPreviewToolbar`) log to console but allow rest of UI to load.
- **API Errors**:
  - 4xx/5xx responses must trigger a standardized Red Toast notification.
  - "Failed to fetch" (Network error) must be caught and toasted.

---

## 6. Soft Requirements (Future Roadmap)


### 6.2 Mobile PWA
- Add `manifest.json` for "Add to Home Screen".
- Offline support (Service Worker) for viewing cached gallery.

### 6.3 Advanced Canvas
- "Paint" capabilities (color drawing) in addition to masking.
- Brush size shortcuts (bracket keys `[` `]`).

### 6.4 Keyboard Shortcuts
- `Ctrl+Enter`: Generate.
- `Esc`: Close Modal/Lightbox.
- `Left/Right`: Navigate Gallery (outside lightbox).
