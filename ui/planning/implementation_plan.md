# WebbDuck UI Rewrite - Implementation Plan

> Archived planning artifact.
> This document reflects an earlier UI rewrite proposal and is not the authoritative current-state spec.

> **Branch**: `ui_rewrite_claude_opus4.5`  
> **Goal**: Create a stunning, modern UI that makes the current design look outdated

---

## Executive Summary

This document outlines a complete rewrite of the WebbDuck UI from scratch. The new design will feature:

- **Modern Aesthetics**: Glassmorphism, smooth gradients, micro-animations
- **Premium Feel**: Dark theme with vibrant accent colors, professional typography  
- **Enhanced UX**: Intuitive workflows, contextual controls, delightful interactions
- **Responsive Design**: Mobile-first approach with adaptive layouts
- **Performance**: Keep the zero-build vanilla JS architecture

---

## Current UI Analysis

### Existing Structure
```
ui/
├── index.html           (409 lines - monolithic)
├── app.js               (405 lines - orchestrator)
├── styles.css           (686 lines - core styles)
├── styles_gallery.css   (632 lines - gallery/lightbox)
├── styles_mobile.css    (95 lines - mobile tweaks)
├── ws.js                (WebSocket helper)
├── modules/
│   ├── gallery.js       (578 lines - PhotoSwipe integration)
│   ├── generation.js    (121 lines - API calls)
│   ├── lora.js          (105 lines - LoRA management)
│   ├── mask-editor.js   (156 lines - inpainting)
│   ├── preview.js       (272 lines - image preview)
│   ├── state.js         (53 lines - localStorage)
│   ├── upload.js        (167 lines - image upload)
│   └── utils.js         (39 lines - helpers)
└── lib/
    └── PhotoSwipe files
```

### Current Design Issues

| Issue | Description |
|-------|-------------|
| **Dated Appearance** | Plain dark colors, minimal visual interest |
| **Flat Design** | No depth, shadows, or layering effects |
| **Basic Typography** | System fonts, no visual hierarchy |
| **Limited Animations** | Only basic transitions |
| **Cluttered Sidebar** | All controls crammed into one column |
| **Inconsistent Spacing** | Variable padding/margins |
| **No Visual Feedback** | Minimal loading/success states |

---

## New Design Vision

### Design Principles

1. **Depth & Dimension**: Layered glassmorphism panels with blur effects
2. **Vibrant Accents**: Electric blue, purple gradients, neon highlights  
3. **Smooth Motion**: Ease-out transitions, subtle hover effects, loading animations
4. **Clear Hierarchy**: Typography scale, visual weight, grouping
5. **Delightful Details**: Micro-interactions, haptic-style feedback, polish

### Color Palette

```css
:root {
  /* Base */
  --bg-deep: #0a0b0f;
  --bg-base: #0d0e14;
  --bg-surface: #12141c;
  --bg-elevated: #181b26;
  
  /* Glass Effects */
  --glass-bg: rgba(18, 20, 28, 0.7);
  --glass-border: rgba(255, 255, 255, 0.08);
  --glass-glow: rgba(99, 102, 241, 0.1);
  
  /* Accent Colors */
  --accent-primary: #6366f1;       /* Indigo */
  --accent-secondary: #8b5cf6;     /* Purple */
  --accent-gradient: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
  
  /* Status Colors */
  --success: #22c55e;
  --warning: #f59e0b; 
  --danger: #ef4444;
  --info: #3b82f6;
  
  /* Text */
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  
  /* Borders */
  --border-subtle: rgba(255, 255, 255, 0.06);
  --border-default: rgba(255, 255, 255, 0.1);
  --border-accent: rgba(99, 102, 241, 0.4);
}
```

### Typography

```css
/* Modern font stack */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  
  /* Type Scale */
  --text-xs: 0.75rem;    /* 12px */
  --text-sm: 0.875rem;   /* 14px */
  --text-base: 1rem;     /* 16px */
  --text-lg: 1.125rem;   /* 18px */
  --text-xl: 1.25rem;    /* 20px */
  --text-2xl: 1.5rem;    /* 24px */
  --text-3xl: 1.875rem;  /* 30px */
}
```

---

## New Layout Architecture

### Desktop Layout (≥1024px)

```
┌──────────────────────────────────────────────────────────────────────┐
│  [Logo] WebbDuck                    [🎨 Studio] [🖼️ Gallery]  [⚙️]  │  <- Top Nav
├────────────────────────────────────┬─────────────────────────────────┤
│                                    │                                 │
│   ┌────────────────────────┐      │     ┌──────────────────────┐    │
│   │     Control Panel      │      │     │                      │    │
│   │  ┌──────────────────┐  │      │     │                      │    │
│   │  │   Model Select   │  │      │     │    Main Preview      │    │
│   │  └──────────────────┘  │      │     │                      │    │
│   │  ┌──────────────────┐  │      │     │                      │    │
│   │  │     Prompt       │  │      │     └──────────────────────┘    │
│   │  └──────────────────┘  │      │                                 │
│   │  ┌──────────────────┐  │      │     ┌──────────────────────┐    │
│   │  │   Parameters     │  │      │     │   Recent Batch       │    │
│   │  │   (Collapsed)    │  │      │     │   [img][img][img]    │    │
│   │  └──────────────────┘  │      │     └──────────────────────┘    │
│   │                        │      │                                 │
│   │  [🧪 Test] [🚀 Gen]   │      │     ┌──────────────────────┐    │
│   └────────────────────────┘      │     │   Status: Idle       │    │
│                                    │     │   VRAM: 8.2GB        │    │
│   ┌────────────────────────┐      │     └──────────────────────┘    │
│   │    Input Image         │      │                                 │
│   │    (Optional)          │      │                                 │
│   └────────────────────────┘      │                                 │
│                                    │                                 │
├────────────────────────────────────┴─────────────────────────────────┤
│                         Collapsed LoRA Panel                         │
└──────────────────────────────────────────────────────────────────────┘
```

### Mobile Layout (≤768px)

```
┌─────────────────────────────────────┐
│  [☰]  WebbDuck         [Studio ▾]  │  <- Compact nav
├─────────────────────────────────────┤
│                                     │
│   ┌─────────────────────────────┐   │
│   │                             │   │
│   │       Main Preview          │   │
│   │                             │   │
│   └─────────────────────────────┘   │
│                                     │
│   ┌─────────────────────────────┐   │
│   │  [Model ▾]  [Steps] [CFG]  │   │  <- Inline controls
│   └─────────────────────────────┘   │
│                                     │
│   ┌─────────────────────────────┐   │
│   │  Prompt textarea            │   │
│   └─────────────────────────────┘   │
│                                     │
│   ┌─────────────────────────────┐   │
│   │  [🧪 Test]    [🚀 Generate] │   │
│   └─────────────────────────────┘   │
│                                     │
├─────────────────────────────────────┤
│   [🎨]        [🖼️]        [⚙️]     │  <- Bottom tabs
└─────────────────────────────────────┘
```

---

## Component Design System

### 1. Glass Panels

```css
.glass-panel {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  backdrop-filter: blur(20px);
  box-shadow: 
    0 8px 32px rgba(0, 0, 0, 0.4),
    inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
```

### 2. Glowing Buttons

```css
.btn-primary {
  background: var(--accent-gradient);
  border: none;
  border-radius: 12px;
  padding: 12px 24px;
  font-weight: 600;
  color: white;
  position: relative;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}

.btn-primary::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, 
    rgba(255,255,255,0.2), 
    transparent);
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 
    0 8px 24px rgba(99, 102, 241, 0.4),
    0 0 40px rgba(99, 102, 241, 0.2);
}
```

### 3. Float Labels (Inputs)

```css
.input-group {
  position: relative;
}

.input-group input {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  padding: 16px 16px 8px;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.input-group input:focus {
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
}

.input-group label {
  position: absolute;
  top: 50%;
  left: 16px;
  transform: translateY(-50%);
  color: var(--text-muted);
  pointer-events: none;
  transition: all 0.2s;
}

.input-group input:focus + label,
.input-group input:not(:placeholder-shown) + label {
  top: 8px;
  font-size: 0.75rem;
  color: var(--accent-primary);
}
```

### 4. Collapsible Sections

```css
.section-collapsible {
  border-radius: 12px;
  overflow: hidden;
  transition: all 0.3s ease;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  cursor: pointer;
  background: var(--bg-surface);
}

.section-header::after {
  content: '▼';
  font-size: 10px;
  transition: transform 0.3s;
}

.section-collapsible.collapsed .section-header::after {
  transform: rotate(-90deg);
}

.section-body {
  max-height: 1000px;
  padding: 16px;
  transition: max-height 0.3s, padding 0.3s;
}

.section-collapsible.collapsed .section-body {
  max-height: 0;
  padding: 0 16px;
  overflow: hidden;
}
```

---

## New File Structure

```
ui/
├── index.html                  # Minimal shell (loads app)
├── styles/
│   ├── design-tokens.css       # CSS variables & tokens
│   ├── reset.css               # Modern CSS reset
│   ├── base.css                # Typography, global styles
│   ├── components/
│   │   ├── buttons.css         # Button variants
│   │   ├── forms.css           # Inputs, selects, toggles
│   │   ├── panels.css          # Glass panels, cards
│   │   ├── modals.css          # Dialogs, overlays
│   │   └── animations.css      # Keyframes, transitions
│   ├── layouts/
│   │   ├── app-shell.css       # Main layout structure
│   │   ├── studio.css          # Generate view
│   │   ├── gallery.css         # History view
│   │   └── responsive.css      # Breakpoints
│   └── main.css                # Import aggregator
├── components/
│   ├── ModelSelector.js        # Model dropdown with preview
│   ├── PromptEditor.js         # Enhanced textarea
│   ├── ParameterPanel.js       # Collapsible settings
│   ├── LoraManager.js          # Visual LoRA grid
│   ├── ImageUploader.js        # Drag-drop with preview
│   ├── MaskEditor.js           # Canvas overlay
│   ├── GenerationQueue.js      # Progress tracking
│   ├── ImagePreview.js         # Main preview with toolbar
│   ├── Lightbox.js             # Full-screen viewer
│   ├── BatchGallery.js         # Session grid
│   └── Toast.js                # Notifications
├── lib/
│   └── PhotoSwipe files
├── core/
│   ├── api.js                  # Fetch wrappers
│   ├── state.js                # State management
│   ├── events.js               # Event bus
│   └── utils.js                # Helpers
├── app.js                      # Entry point
└── manifest.json               # PWA manifest
```

---

## Key Feature Enhancements

### 1. Prompt Editor
- **Syntax highlighting** for tags, weights, embeddings
- **Auto-complete** for known tokens
- **Token counter** with visual bar
- **Template presets** dropdown

### 2. Model Selector
- **Model cards** with thumbnail previews
- **Quick-switch** between recent models
- **Compatibility indicators** for LoRAs

### 3. Parameter Controls
- **Smart presets** (Quality, Speed, Balanced)
- **Visual sliders** with live preview hints
- **Linked dimensions** (lock aspect ratio)
- **Randomize seed** button

### 4. LoRA Manager
- **Visual grid** with thumbnails
- **Drag-to-reorder** priority
- **Strength curves** (not just linear)
- **Batch enable/disable**

### 5. Generation Progress
- **Animated progress ring**
- **Step counter** (12/30)
- **ETA display**
- **Cancel button**

### 6. Image Preview
- **Smooth zoom/pan** gestures
- **Comparison modes**: Side-by-side, Wipe, Blend
- **Quick actions**: Upscale, Inpaint, Variations
- **Metadata overlay**

### 7. Gallery
- **Masonry layout** with lazy loading
- **Infinite scroll**
- **Multi-select** for batch operations
- **Search/filter** by prompt, model, date

---

## Implementation Phases

### Phase 1: Foundation (Files 1-4)
1. Create new CSS architecture
2. Build design tokens
3. Create base components (buttons, forms)
4. Set up app shell layout

### Phase 2: Core Views (Files 5-8)
1. Build Studio view with panels
2. Create enhanced Prompt Editor
3. Build Parameter controls
4. Create Image Preview component

### Phase 3: Advanced Features (Files 9-12)
1. Implement LoRA Manager
2. Build Mask Editor
3. Create Gallery view
4. Build Lightbox component

### Phase 4: Polish (Files 13-15)
1. Add micro-animations
2. Implement toast notifications
3. Mobile responsive refinements
4. Performance optimization

---

## API Compatibility

The new UI will maintain **100% compatibility** with existing endpoints:

| Endpoint | Method | Usage |
|----------|--------|-------|
| `/models` | GET | Fetch available models |
| `/models/{model}/loras` | GET | Fetch LoRAs for model |
| `/second_pass_models` | GET | Refiner models |
| `/schedulers` | GET | Scheduler list |
| `/generate` | POST | Main generation |
| `/test` | POST | Single test image |
| `/upscale` | POST | 2x upscaling |
| `/caption` | POST | Image captioning |
| `/gallery` | GET | History data |
| `/delete_image` | POST | Remove single image |
| `/delete_run` | POST | Remove batch |
| `/tokenize` | POST | Token counting |
| `/captioners` | GET | Captioner availability |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Visual Appeal | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Load Time | ~1s | <0.5s |
| Lighthouse Score | ~70 | 95+ |
| Mobile UX | ⭐⭐ | ⭐⭐⭐⭐ |
| Accessibility | Basic | WCAG AA |

---

## Next Steps

1. **Review this plan** and provide feedback
2. Begin Phase 1 implementation
3. Iterate based on visual testing
4. Complete all phases

> **IMPORTANT**: This is a complete rewrite. The old UI files will remain untouched initially but will eventually be replaced.
