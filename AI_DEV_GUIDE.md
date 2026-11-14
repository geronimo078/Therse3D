# AI-Assisted Development Guide (`therse3d`)

This document defines the mandatory workflow for **AI coding agents** (e.g., Antigravity, Cursor, Claude Code, Codex) and **local LLMs/VLMs** when developing and using `therse3d`.

---

## 1. Mandatory Context Before Any Code Change

Before implementing or modifying any code, always consult:
1. **[Architecture.md](Architecture.md)** — Master architecture, ECS memory layout, public API syntax, and the 6-step implementation plan.
2. **This Guide (`AI_DEV_GUIDE.md`)** — Development workflow, hard constraints, and verification requirements.

---

## 2. The 5 Non-Negotiable Coding Rules

All generated code — whether written by a human or an AI — **must** follow these 5 rules:

1. **No External 3D Libraries:**
   - Never add `three`, `babylonjs`, `gl-matrix`, or similar packages as runtime dependencies.
   - Keep all math helpers (`vec3`, `mat4`, `quat`, `aabb`) zero-allocation inside `src/core/Math.ts`.
2. **Zero Garbage-Collection Pressure in the Render Loop:**
   - Inside `onTick()`, `Renderer.renderFrame()`, and `WorldStore.updateMatrices()`, **never** allocate new objects (`{}`), arrays (`[]`), or TypedArrays (`new Float32Array`) per frame.
   - Always reuse pre-allocated scratch buffers at class or module scope.
3. **Strict WebGPU Resource Tracking:**
   - Register every `GPUBuffer` and `GPUTexture` through `BufferManager`.
   - When removing entities (`app.remove(id)` or `app.sync()`), deterministically free associated GPU resources via `.destroy()`.
4. **Fault Tolerance Over Exceptions (Auto-Healing):**
   - Public API methods — especially `app.sync()` — must never crash on invalid AI inputs (`NaN`, `Infinity`, malformed hex colors, unknown `on` references, or unknown material names).
   - Clamp or fall back to safe defaults and return structured entries in `warnings[]`.
5. **API Parity (KISS):**
   - Every new feature must be **identically** configurable through both the imperative JS API (`Box(...)`) and the declarative JSON schema (`{ type: 'box' }`).

---

## 3. Development Workflow Along the 6-Step Plan

The engine is built incrementally following **Section 6 of `Architecture.md`**:

1. **Step 1:** Foundation, WebGPU Device & ECS TypedArray Bridge (`Math.ts`, `Device.ts`, `WorldStore.ts`, `EntityHandle.ts`, `Engine.ts`)
2. **Step 2:** WebGPU Instanced PBR Renderer & Semantic Materials (`BufferManager.ts`, `Primitives.ts`, `Materials.ts`, `pbr.wgsl`, `Renderer.ts`, `OrbitControls.ts`)
3. **Step 3:** Spatial Layout Engine (`SpatialLayout.ts` — `on`, `align`, `offset`, `lookAt`)
4. **Step 4:** Declarative `app.sync()` & Auto-Healing State Engine (`SceneSync.ts`)
5. **Step 5:** Bidirectional AI Perception (`Inspector.ts`, `id_depth.wgsl`, `Snapshot.ts`)
6. **Step 6:** LLM Schema, Prompt Generator, MCP Bridge & glTF Loader (`Schema.ts`, `McpBridge.ts`, `GltfLoader.ts`)

### Rules Per Step for AI Agents:
- **Complete one step cleanly at a time**, verifying its acceptance criterion from `Architecture.md`.
- **Write unit tests** for pure logic modules (`Math.ts`, `WorldStore.ts`, `SpatialLayout.ts`, `SceneSync.ts`, `Inspector.ts`, `Schema.ts`).
- **Verify after every change:**
  ```bash
  npm run typecheck
  npm test
  npm run build
  ```
- **Keep the core zero-dependency:** `"dependencies"` in `package.json` must remain empty.

---

## 4. Runtime AI Agent Loop

When an LLM or VLM builds or refines a 3D scene with `therse3d`, it follows this closed feedback loop:

1. **Load Context:** `Engine.getPromptContext(app)` provides the compact system prompt including available primitives, material presets, and active entity IDs.
2. **Sync Scene (`sync`):** The agent sends a declarative JSON state to `app.sync(state)`. Prefer relative placement (`on`, `align`) over manual Y-coordinate math.
3. **Inspect Warnings & Spatial Graph (`describe`):** Check `SyncReport.warnings` and `app.describe()` to detect off-screen (`offscreen-*`) or intersecting (`collidingWith`) objects.
4. **Visually Verify (`snapshot`):** When using a VLM, call `await app.snapshot({ layers: ['rgb', 'annotated', 'depth', 'ids'] })` and adjust based on the annotated bounding boxes.

---

## 5. Useful Commands

| Command | Purpose |
| :--- | :--- |
| `npm run dev` | Starts Vite dev server (examples at `/examples/basic.html`, `/examples/ai-sync.html`, `/examples/vlm-vision.html`) |
| `npm run typecheck` | Strict TypeScript check (`tsc --noEmit`) |
| `npm test` | Runs Vitest unit tests |
| `npm run build` | Typecheck + production bundle in `dist/` |
