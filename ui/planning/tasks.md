# WebbDuck UI Rewrite - Task Tracker

## Phase 1: Foundation ✅ In Progress

### CSS Architecture
- [ ] Create `styles/design-tokens.css` - CSS variables & design tokens
- [ ] Create `styles/reset.css` - Modern CSS reset
- [ ] Create `styles/base.css` - Typography & global styles
- [ ] Create `styles/main.css` - Import aggregator

### Component Styles
- [ ] Create `styles/components/buttons.css` - Button variants
- [ ] Create `styles/components/forms.css` - Inputs, selects, toggles
- [ ] Create `styles/components/panels.css` - Glass panels, cards
- [ ] Create `styles/components/modals.css` - Dialogs, overlays
- [ ] Create `styles/components/animations.css` - Keyframes, transitions

### Layout Styles
- [ ] Create `styles/layouts/app-shell.css` - Main layout structure
- [ ] Create `styles/layouts/studio.css` - Generate view
- [ ] Create `styles/layouts/gallery.css` - History view  
- [ ] Create `styles/layouts/responsive.css` - Breakpoints

### New HTML Shell
- [ ] Create new `index.html` with modern structure

---

## Phase 2: Core Views

### Core JS Modules
- [ ] Create `core/api.js` - Fetch wrappers
- [ ] Create `core/state.js` - State management
- [ ] Create `core/events.js` - Event bus
- [ ] Create `core/utils.js` - Helpers

### Components
- [ ] Create `components/ModelSelector.js`
- [ ] Create `components/PromptEditor.js`
- [ ] Create `components/ParameterPanel.js`
- [ ] Create `components/ImagePreview.js`

### Entry Point
- [ ] Create new `app.js` - Main orchestrator

---

## Phase 3: Advanced Features

### Components
- [ ] Create `components/LoraManager.js`
- [ ] Create `components/ImageUploader.js`
- [ ] Create `components/MaskEditor.js`
- [ ] Create `components/GenerationQueue.js`

### Gallery
- [ ] Create `components/BatchGallery.js`
- [ ] Create `components/Lightbox.js`

---

## Phase 4: Polish

### Enhancements
- [ ] Add micro-animations throughout
- [ ] Create `components/Toast.js` - Notifications
- [ ] Mobile responsive testing & fixes
- [ ] Performance optimization
- [ ] Accessibility audit

### Cleanup
- [ ] Remove old CSS files
- [ ] Remove old module files
- [ ] Update README.md
- [ ] Final testing

---

## Notes

- Keep PhotoSwipe library files
- Maintain 100% API compatibility
- Test on Chrome, Firefox, Safari
- Test on iOS and Android
