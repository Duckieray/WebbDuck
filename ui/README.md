# WebbDuck UI

The frontend interface for WebbDuck AI Image Generator.

## Structure

The UI is built with vanilla HTML/CSS/JS (ES6 Modules) for maximum performance and zero build steps.

- **index.html**: Main entry point and layout.
- **app.js**: Main orchestrator script. Imports modules and initializes the app.
- **styles.css**: Core application styles.
- **modules/**: Logic split into functional areas.

### Modules

| Module | Description |
|--------|-------------|
| `state.js` | Manages global state (prompt, settings) and LocalStorage persistence. |
| `gallery.js` | Handles history view, incremental loading, and PhotoSwipe integration. |
| `generation.js` | API communication for `/generate`, `/test`, and `/upscale`. |
| `mask-editor.js` | Canvas-based mask drawing logic for inpainting. |
| `preview.js` | Manages the main image preview area, including wipe comparisons. |
| `lora.js` | Handles fetching, displaying, and managing LoRA models. |
| `upload.js` | Image upload handling, drag-and-drop, and captioning. |
| `utils.js` | Shared helpers (token counting, select population). |

## Key Features

- **Zero-Build**: Files are served directly. No webpack/vite required.
- **State Persistence**: Settings are automatically saved to LocalStorage.
- **PhotoSwipe**: Advanced gallery viewer with zoom/pan and metadata support.
- **Mask Editor**: Built-in canvas editor for inpainting masks.
- **Wipe Preview**: "Before/After" slider for upscaled images.
