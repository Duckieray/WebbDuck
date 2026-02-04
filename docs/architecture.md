# Architecture Overview

WebbDuck is composed of small, replaceable layers.

---

## High-Level Flow

1. UI collects generation request
2. Core validates configuration
3. Prompt system prepares conditioning
4. Pipeline executes generation
5. Outputs are returned

---

## Core Components

### Pipeline Manager
- Owns model lifecycle
- Manages second-pass model attachment
- Ensures VRAM cleanup
- Prevents duplicate loading
- Distinguishes true SDXL refiners from generic img2img models
- Enforces correct conditioning paths based on UNet configuration
- Prevents misusing non-refiner checkpoints as refiners

### Prompt System
- Token-aware
- Explicit truncation handling
- Optional experimental routing
- Never rewrites without consent

### Experimental Router
- Decides whether to use:
  - standard generation
  - two-pass generation
  - prompt decomposition
- Entirely bypassed when disabled
    
### Resource Management
- **Smart Offloading**: Pipelines are automatically moved to CPU when not in use to free up VRAM for other tasks (like captioning).
- **VRAM Monitoring**: Real-time tracking of memory usage.


---

## What WebbDuck Avoids

- Persistent global state
- Hidden caches
- Implicit prompt mutation
- Unbounded background workers