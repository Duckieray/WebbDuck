# WebbDuck UI Module

The frontend interface for WebbDuck AI Image Generator.
Built with modern Vanilla JS (ES Modules) and CSS Variables. Zero build steps required.

## 📂 Project Structure

```
ui/
├── index.html          # Main entry point and DOM structure
├── app.js              # Application bootstrapper and orchestrator
├── core/               # Shared utilities
│   ├── api.js          # API client
│   ├── events.js       # Event bus & WebSocket handler
│   ├── state.js        # Global state with persistence
│   └── utils.js        # DOM helpers (byId, listen, etc.)
├── modules/            # Feature-specific Logic
│   ├── LightboxManager.js  # Gallery & PhotoSwipe integration
│   ├── LoraManager.js      # LoRA selection & loading
│   ├── MaskEditor.js       # Inpainting canvas logic
│   └── ProgressManager.js  # WebSocket progress visualization
└── styles/             # CSS Architecture
    ├── main.css        # Entry point
    ├── design-tokens.css # Global variables
    └── components/     # Reusable UI components
```

## 🛠️ Development Guide

### 1. How to Add a New Button

Buttons should receive classes from `styles/components/buttons.css`.

**Example HTML (`index.html`):**
```html
<!-- Primary Action -->
<button class="btn btn-primary" id="btn-save">
  <svg>...</svg>
  Save Project
</button>

<!-- Secondary / Tool -->
<button class="btn btn-secondary btn-icon" id="btn-tool" title="Tool Name">
  <svg>...</svg>
</button>
```

**Common Classes:**
- `.btn`: Base class (required)
- `.btn-primary`: Gradient accent background (Main actions)
- `.btn-secondary`: Glassmorphism background (Tools/Toggles)
- `.btn-ghost`: Transparent (Icon-only buttons)
- `.btn-sm`: Small variant

### 2. How to Add a Handler

Event listeners are standard `addEventListener` calls, but we use the `listen` helper from `core/utils.js` for cleaner code.

**In `app.js` (for global buttons):**
```javascript
import { listen, byId } from './core/utils.js';

// Inside init() or setupHandlers()
listen(byId('btn-save'), 'click', async () => {
    // Your logic here
    console.log('Saved!');
});
```

**In a Module (e.g., `MyModule.js`):**
```javascript
export class MyModule {
    constructor() {
        this.btn = byId('btn-tool');
        this.init();
    }

    init() {
        if (this.btn) {
            // Bind 'this' context !
            listen(this.btn, 'click', this.handleClick.bind(this));
        }
    }

    handleClick() {
        // Logic
    }
}
```

### 3. State Management

All persistent settings (prompt, dimensions, toggles) are managed in `core/state.js`.

**Reading State:**
```javascript
import { getState } from './core/state.js';
const { prompt, width } = getState();
```

**Writing State:**
```javascript
import { setState } from './core/state.js';

// Updates state, saves to localStorage, and triggers subscribers
setState({ width: 512, height: 768 });
```

### 4. Event Bus

Use the Event Bus to communicate between modules without tight coupling.

```javascript
import { emit, on, Events } from './core/events.js';

// Dispatching
emit(Events.GENERATION_START, { mode: 'test' });

// Listening
on(Events.GENERATION_START, (data) => {
    console.log('Generation started:', data.mode);
});
```

### 5. API Interactions

Add new endpoints to `core/api.js`. Use the `postForm` helper for FormData interactions.

```javascript
export async function myNewAction(formData) {
    return postForm('/my_action', formData);
}
```
