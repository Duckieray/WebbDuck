# WebbDuck — Vision

Modern diffusion tooling has two extremes:

- High-level UIs that hide behavior and impose guardrails
- Low-level node graphs that expose everything but obscure intent

WebbDuck exists to occupy the space in between.

---

## The Core Idea

**Diffusion workflows should be declarative, not procedural.**

Users should describe:
- what they want
- how strongly they want it
- when details should matter

Not:
- how tensors flow between nodes
- which internal pipeline step happens first
- how many times a prompt is re-encoded

WebbDuck treats prompt conditioning as a *first-class problem*, not a side effect.

---

## Why Diffusers?

Diffusers is:
- explicit
- modular
- well-maintained
- easy to reason about

WebbDuck does not wrap Diffusers to hide it.
It exposes Diffusers cleanly.

---

## What Makes WebbDuck Different

- No permanent background processes
- No implicit prompt rewriting
- No silent truncation without warning
- No hidden VRAM usage
- No forced “one true workflow”

If WebbDuck does something unusual, it logs it.

---

## Long-Term Direction

WebbDuck aims to become:
- a reference implementation for modern diffusion workflows
- a testbed for prompt conditioning research
- a clean base for experimentation without UI bloat
