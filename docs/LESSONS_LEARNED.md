# Lessons Learned 🎓

A retrospective on the architectural decisions and evolution of the WebbDuck project.

## 1. Frontend Architecture: The Monolith Trap

### The Problem
Initially, `app.js` contained all the logic for UI interaction, API calls, and state handling. As features like the Mask Editor and Gallery grew, the file exceeded 1,500 lines, making it:
- Hard to navigate.
- Prone to regression bugs (changing one thing broke another).
- Difficult to test.

### The Solution: Modularization
We refactored the monolith into **Class-based Modules** (`ui/modules/`):
- **Encapsulation**: Each module (`MaskEditor`, `LoraManager`) manages its own DOM elements and event listeners.
- **Dependency Injection**: Modules are initialized in `app.js`, but don't depend on `app.js`.
- **Event Bus**: Communication happens via a decoupled `events.js` system, preventing "spaghetti code" references.

**Lesson**: heavily interactive Vanilla JS apps *must* adopt a modular structure early. Don't wait for the file to be "too big."

## 2. State Management

### The Problem
Early iterations relied on reading DOM elements (`document.getElementById('width').value`) whenever a value was needed. This caused:
- **Desync**: The UI might show one value, but the variable in memory was different.
- **Race Conditions**: Setting values programmatically (e.g., during image upload) didn't always trigger the expected listeners.
- **Persistence Issues**: Values were lost on refresh.

### The Solution: Centralized Store
We implemented `core/state.js` as the "Single Source of Truth":
1.  **State Logic**: Values live in a managed object.
2.  **Sync Functions**: `syncToDOM()` and `syncFromDOM()` handle the boundary.
3.  **Persistence**: Every state change is automatically saved to `localStorage`.

**Lesson**: Separate the *data* from the *view*. The DOM should reflect state, not *be* the state.

## 3. Backend: Async vs. Blocking

### The Problem
Processing an image generation request locks the GPU. If we ran this directly in the FastAPI endpoint function:
- The server would become unresponsive to other requests (like health checks or progress updates).
- Python's Global Interpreter Lock (GIL) and CUDA synchronization would block the event loop.

### The Solution: The Queue Pattern
We implemented `core/worker.py` and an `asyncio.Queue`:
1.  **Endpoints are lightweight**: They just validate input, create a `Future`, and push a job to the queue.
2.  **Worker Loop**: A dedicated background task processes jobs one by one.
3.  **Cancellation**: separation allows us to interrupt or manage jobs independently of the HTTP request lifecycle.

**Lesson**: For heavy compute (AI/ML), never block the API handler. Use a queue.

## 4. Real-time Feedback

### The Problem
Polling (`setInterval`) was used to check generation progress. This was inefficient and led to "jumpy" progress bars or skipped updates.

### The Solution: WebSockets
We integrated a WebSocket pipeline (`server/events.py` -> `ui/core/events.js`):
- **Push vs Pull**: The server pushes updates only when they happen.
- **Rich Data**: We can send detailed objects (VRAM, Step, Stage) instantly.
- **Visuals**: Coupled with `ProgressManager.js`, this makes the app feel "alive" and responsive.

**Lesson**: For operations taking >1 second, use WebSockets over polling for a premium user experience.

## 5. CSS Strategy
We moved from a single `style.css` to a component-based structure (`buttons.css`, `forms.css`, `layout/`).
- **Design Tokens**: Defining colors and spacing in variables (`design-tokens.css`) made "Dark Mode" and theming trivial.
- **Maintainability**: It's much easier to fix a button bug when `buttons.css` exists.

**Lesson**: Structure CSS as strictly as you structure JavaScript.
