# Design Philosophy

WebbDuck follows a few strict principles.

---

## 1. Opt-in over implicit behavior

If a feature:
- changes prompt meaning
- adds compute cost
- alters model behavior

…it must be explicitly enabled.

---

## 2. Experimental means experimental

Experimental features:
- are disabled by default
- are clearly logged
- may fail loudly
- may be removed entirely

Stability is a feature — experimentation is optional.

---

## 3. No silent failures

If something is truncated, skipped, or approximated:
- the user is told
- logs explain why
- defaults remain predictable

---

## 4. Respect user hardware

WebbDuck should never:
- preload unnecessary models
- run background LLMs
- consume VRAM unless generating

Users running on tight systems should feel safe using it.

---

## 5. Model-agnostic by design

WebbDuck does not assume:
- Realism
- Anime
- SDXL vs SD1.5

Models define behavior. WebbDuck routes intent.
