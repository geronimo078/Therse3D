# AGENTS.md — Project Instructions for AI Coding Agents

This repository implements **therse3d**, an AI-native WebGPU 3D engine in TypeScript.

## Mandatory Reading Before Changes
1. `Architecture.md` — Full specification, directory structure, ECS layout, and 6-step implementation plan.
2. `AI_DEV_GUIDE.md` — Mandatory guide for AI-assisted development and agent workflows.

## Hard Constraints
- **Zero-Dependency Core:** Never add `three`, `babylonjs`, `gl-matrix`, or any runtime dependencies.
- **Zero-Allocation Render Loop:** No `new` allocations per frame in `Engine` tick, `WorldStore.updateMatrices()`, or `Renderer.renderFrame()`.
- **Deterministic GPU Memory:** Track every `GPUBuffer`/`GPUTexture` via `BufferManager` and dispose cleanly with `.destroy()`.
- **Auto-Healing Over Crashes:** `app.sync()` and the public API must recover invalid inputs to safe defaults and report `warnings[]`.
- **API Parity:** Imperative API (`Box(...)`) and declarative JSON (`app.sync({ entities: [...] })`) must support identical options.

## Verification After Every Task
Always run after code changes:
- `npm run typecheck`
- `npm test`
- `npm run build`
