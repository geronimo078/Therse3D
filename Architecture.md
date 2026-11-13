# therse3d — AI-Native WebGPU 3D Engine (Master Architecture & Spec)

> **Project Goal:** A high-performance, data-oriented WebGPU 3D engine in TypeScript that is as easy to use as Three.js (`10 lines to first render`), yet optimized from the ground up for **AI code generation (LLMs)**, **AI agents (MCP/tool-use)**, and **Vision-Language Models (VLMs)**.

---

## 1. Core Design Principles

1. **Dual-Layer Architecture:**
   * **Public API (Outer):** Declarative, flat, fault-tolerant, and extremely token-efficient (minimal boilerplate for humans and LLMs).
   * **Engine Core (Inner):** Data-oriented Entity-Component-System (ECS) backed by flat `TypedArrays` (`Float32Array`, `Uint32Array`) with zero-copy uploads to WebGPU storage buffers.
2. **WebGPU-First (No WebGL Legacy Bloat):**
   * Exclusive use of WebGPU & WGSL.
   * Automatic GPU instancing for identical geometries (1 draw call for tens of thousands of objects) plus GPU-driven culling.
3. **Zero-Boilerplate ("Batteries Included"):**
   * `Engine.create()` automatically handles canvas setup, `ResizeObserver`, `devicePixelRatio`, render loop, tone mapping, contact shadows, and studio lighting.
4. **Spatial Intelligence & Auto-Healing:**
   * Relative placement (`on: 'table'`, `align: 'front-right'`) instead of error-prone manual coordinate math by LLMs.
   * Fault-tolerant `app.sync(json)` with automatic diffing, smooth tweening, and structured warnings instead of runtime crashes.
5. **Bidirectional AI Perception:**
   * `app.describe()` returns a compact, token-efficient spatial graph including visibility and collision warnings.
   * `app.snapshot()` renders RGB, annotated bounding boxes, depth map, and object ID masks simultaneously via Multiple Render Targets (MRT).

---

## 2. Directory Structure

```text
therse3d/
├── ARCHITECTURE.md           # This specification (master context for local AI agents)
├── AI_DEV_GUIDE.md           # Mandatory guide for AI-assisted development
├── AGENTS.md                 # Entry-point instructions for AI coding agents
├── package.json              # Zero-dependency core (TypeScript + @webgpu/types)
├── tsconfig.json             # Strict TypeScript configuration (ESNext, bundler)
├── src/
│   ├── index.ts              # Public API exports (Engine, Box, Sphere, Cylinder, Plane, Model, OrbitControls)
│   ├── core/
│   │   ├── Engine.ts         # Entry point, lifecycle, presets, ticker, plugin system
│   │   ├── Math.ts           # Zero-allocation math helpers (vec3, mat4, quat, aabb)
│   │   ├── WorldStore.ts     # ECS data layer (flat TypedArrays + dirty flags)
│   │   ├── SpatialLayout.ts  # Bounding-box computation & relative placement (on, align, margin)
│   │   └── SceneSync.ts      # Idempotent JSON state sync with diffing, auto-tweening & auto-healing
│   ├── gpu/
│   │   ├── Device.ts         # WebGPU adapter/device initialization & feature checks
│   │   ├── Renderer.ts       # Render pipeline, auto-instancing, Multiple Render Targets (MRT)
│   │   ├── BufferManager.ts  # GPU storage buffers & deterministic memory disposal
│   │   └── shaders/
│   │       ├── pbr.wgsl      # Standard PBR shader + environment lighting + instancing
│   │       ├── id_depth.wgsl # Multi-pass shader for object ID picking & depth maps
│   │       └── splat.wgsl    # Compute sort & quad rendering for 3D Gaussian Splatting
│   ├── entities/
│   │   ├── EntityHandle.ts   # Lightweight proxy/wrapper over ECS index (OOP ergonomics)
│   │   ├── Primitives.ts     # Factory functions: Box(), Sphere(), Cylinder(), Plane(), Line()
│   │   ├── Materials.ts      # Semantic presets ('oak-wood', 'brushed-aluminum', 'frosted-glass', etc.)
│   │   └── GltfLoader.ts     # Lean glTF 2.0 (.glb) parser with automatic instancing
│   ├── controls/
│   │   └── OrbitControls.ts  # Camera controls with smooth damping & camera.fitAll()
│   └── ai/
│       ├── Inspector.ts      # app.describe() -> spatial JSON & frustum warnings for LLMs
│       ├── Snapshot.ts       # app.snapshot() -> RGB, annotated overlay, depth & ID buffers
│       ├── Schema.ts         # JSON Schema & Engine.getPromptContext() for LLM tool calling
│       └── McpBridge.ts      # Optional WebSocket/PostMessage bridge for local AI agents
└── examples/
    ├── basic.html            # 10-line "Hello Cube" example
    ├── ai-sync.html          # Live scene control via JSON state updates
    └── vlm-vision.html       # Multi-pass snapshot & spatial inspector demo
```

---

## 3. Public API Design (Target Syntax)

### A) Imperative API (Three.js Ergonomics Without Boilerplate)
```typescript
import { Engine, Box, Sphere, OrbitControls } from 'therse3d';

const app = await Engine.create('#canvas', {
  preset: 'studio',     // 'studio' | 'outdoor' | 'dark-ui' | 'cad-blueprint'
  autoResize: true,
  antialias: true
});

const table = app.add(Box({
  id: 'table',
  size: [2.4, 0.08, 1.2],
  position: [0, 0.75, 0],
  material: 'oak-wood'
}));

// Relative placement: sits flush on 'table' without manual Y calculation
const orb = app.add(Sphere({
  id: 'orb',
  radius: 0.15,
  on: 'table',
  align: 'center',
  color: '#00aaff',
  metalness: 0.6
}));

app.use(OrbitControls());
app.camera.fitAll({ padding: 1.2 });

app.onTick((dt, elapsed) => {
  orb.position[1] += Math.sin(elapsed * 3) * 0.002;
});
```

### B) Declarative AI-Sync API (`app.sync`)
```typescript
const report = app.sync({
  preset: 'studio',
  camera: { target: 'orb', angle: 'front-elevated', distance: 3.5 },
  entities: [
    { id: 'table', type: 'box', size: [2.4, 0.08, 1.2], position: [0, 0.75, 0], material: 'oak-wood' },
    { id: 'orb', type: 'sphere', radius: 0.2, on: 'table', align: 'front-right', color: '#ff3366' }
  ]
});
// Return value for the AI agent: { applied: 2, removed: 0, warnings: [] }
```

### C) AI Inspection & Vision Feedback (`app.describe` & `app.snapshot`)
```typescript
const spatialState = app.describe();
const vision = await app.snapshot({
  layers: ['rgb', 'annotated', 'depth', 'ids'],
  resolution: [1024, 768]
});
```

---

## 4. Data Model & ECS Memory Layout (`WorldStore.ts`)

All transform and material data live in contiguous `TypedArrays` to avoid CPU cache misses and enable direct `device.queue.writeBuffer()` uploads:

| Buffer / Array | Type | Stride / Size | Description |
| :--- | :--- | :--- | :--- |
| `positions` | `Float32Array` | `3 * MAX` | `[x, y, z]` world coordinates |
| `rotations` | `Float32Array` | `4 * MAX` | `[x, y, z, w]` quaternions |
| `scales` | `Float32Array` | `3 * MAX` | `[sx, sy, sz]` scale |
| `modelMatrices` | `Float32Array` | `16 * MAX` | Precomputed 4x4 matrices (direct GPU upload) |
| `colors` | `Float32Array` | `4 * MAX` | `[r, g, b, a]` linear color space |
| `pbrParams` | `Float32Array` | `4 * MAX` | `[metalness, roughness, emissive, materialId]` |
| `boundsMin` / `Max` | `Float32Array` | `3 * MAX` | AABB bounding boxes for `SpatialLayout` & frustum checks |
| `dirtyFlags` | `Uint8Array` | `1 * MAX` | Marks modified entities for selective GPU updates |

---

## 5. Rules for AI-Assisted Code Generation (Coding Guidelines)

1. **No External 3D Libraries:** Never introduce `three`, `babylonjs`, or `gl-matrix` as dependencies. Math helpers (`vec3`, `mat4`, `quat`) are kept lean and zero-allocation in `src/core/Math.ts`.
2. **Zero Garbage-Collection Pressure in the Render Loop:** In `onTick`, `Renderer.renderFrame()`, and `WorldStore.updateMatrices()`, never allocate new arrays or objects (`new Float32Array`, `{}`) per frame. Use pre-allocated scratch buffers.
3. **Strict WebGPU Resource Tracking:** Every `GPUBuffer` and `GPUTexture` is registered via `BufferManager` and deterministically released with `.destroy()` when an entity is removed (`app.remove(id)` or `app.sync()`).
4. **Fault Tolerance Over Exceptions:** Public API methods (especially `app.sync()`) must not throw fatal errors on malformed AI inputs (e.g., `NaN`, invalid hex colors, unknown `on` references); instead, they fall back to safe defaults and log structured entries in `warnings[]`.
5. **KISS Parity for Primitives & Shaders:** New features must be identically configurable via both the imperative JS API (`Box(...)`) and the declarative JSON schema (`{ type: 'box' }`).

---

## 6. Detailed 6-Step Implementation Plan

### Step 1: Foundation, WebGPU Device & ECS TypedArray Bridge
**Goal:** Establish the minimal skeleton; a canvas is initialized, WebGPU runs stably, and entities can be created and managed in memory via a Three.js-like API.

* **Files:** `package.json`, `tsconfig.json`, `src/core/Math.ts`, `src/gpu/Device.ts`, `src/core/WorldStore.ts`, `src/entities/EntityHandle.ts`, `src/core/Engine.ts`
* **Details:**
  * `Math.ts`: Zero-allocation `mat4`, `vec3`, `quat`.
  * `Device.ts`: `navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })`, `requestDevice()`, `device.lost` handling.
  * `WorldStore.ts`: `TypedArrays` (initial capacity `10_000` entities, dynamically growing), free-list for entity IDs (`allocate()` / `free(id)`), `updateMatrices()` for `dirtyFlags`.
  * `EntityHandle.ts`: Getters/setters (`position`, `rotation`, `scale`, `color`, `visible`) writing directly into `WorldStore`.
  * `Engine.ts`: `Engine.create(selectorOrCanvas, options)` with `ResizeObserver`, `devicePixelRatio`, `requestAnimationFrame` loop (`onTick`), and `destroy()`.
* **Acceptance Criterion:** `Engine.create('#canvas')` starts without errors; `app.add(Box(...))` allocates an ID and `cube.position = [1, 2, 3]` updates the 4x4 matrix in `Float32Array`.

---

### Step 2: WebGPU Instanced PBR Renderer & Semantic Materials
**Goal:** All base primitives render in high-quality PBR look via a single instanced draw call per geometry type.

* **Files:** `src/gpu/BufferManager.ts`, `src/entities/Primitives.ts`, `src/entities/Materials.ts`, `src/gpu/shaders/pbr.wgsl`, `src/gpu/Renderer.ts`, `src/controls/OrbitControls.ts`
* **Details:**
  * `BufferManager.ts`: Manages `GPUBuffer` & `GPUTexture`, uploads modified ranges (`writeBuffer`), and deterministically destroys unused resources.
  * `Primitives.ts`: Procedural vertex/index generators (position, normal, UV) for `Box`, `Sphere`, `Cylinder`, and `Plane`.
  * `Materials.ts`: Translates semantic names (`'oak-wood'`, `'brushed-aluminum'`, `'matte-plastic'`, `'frosted-glass'`, `'emissive-neon'`) into `[metalness, roughness, emissive, baseTint]`.
  * `pbr.wgsl`: Reads `modelMatrices`, `colors`, `pbrParams` via `@instance_index` from storage buffers. Implements Cook-Torrance PBR shading, 3-point studio lighting, hemisphere ambient, and ACES tone mapping.
  * `Renderer.ts`: Depth texture (`depth24plus`), MSAA (`sampleCount: 4`), auto-instancing per geometry type, and scene presets (`studio`, `outdoor`, `dark-ui`, `cad-blueprint`).
  * `OrbitControls.ts`: Orbit, pan, zoom with smooth damping plus `camera.fitAll({ padding })`.
* **Acceptance Criterion:** `examples/basic.html` renders a lit 3D scene in < 10 lines of code; 10,000 rotating boxes run at 60/120 FPS.

---

### Step 3: Spatial Layout Engine (Relative Placement Without Coordinate Math)
**Goal:** LLMs do not need to compute exact world coordinates to place objects on top of, next to, or aligned with each other.

* **Files:** `src/core/SpatialLayout.ts`
* **Details:**
  * Computes world-space AABB (`boundsMin`, `boundsMax`) from geometry, scale, and position.
  * Topological sorting for nested dependencies (`mug` on `table` on `floor`).
  * Supported layout directives:
    * `on: string` (entity ID or `'floor'`): Sets `minY` flush with `maxY` of the target object (`margin` optional).
    * `align: 'center' | 'front' | 'back' | 'left' | 'right' | 'front-left' | 'front-right' | 'back-left' | 'back-right'`: Aligns X/Z relative to the target bounding box.
    * `offset: [dx, dy, dz]`: Fine adjustment after automatic alignment.
    * `lookAt: string | [x, y, z]`: Automatically rotates the entity toward the target.
* **Acceptance Criterion:** An LLM can place a table, laptop, and mug using only `on` and `align`, without floating or intersecting objects.

---

### Step 4: Declarative `app.sync()` & Auto-Healing State Engine
**Goal:** Idempotent interface for AI agents that accepts JSON scenes, computes diffs, animates changes, and catches AI mistakes.

* **Files:** `src/core/SceneSync.ts`
* **Details:**
  * **Diff Algorithm:** Compares incoming `entities[]` by `id` with `WorldStore` (create new, update changed properties, delete removed IDs + free GPU memory).
  * **Auto-Tweening:** Smoothly interpolates position, rotation, and color changes over `transition` (default: `0.25s`).
  * **Auto-Healing & Validation:** Replaces `NaN`/`Infinity` with safe defaults, normalizes colors (`#f00`, `[1,0,0]`, CSS names), catches invalid `on` references or unknown material names, and reports them cleanly in `SyncReport.warnings[]`.
* **Acceptance Criterion:** Repeated `app.sync()` calls smoothly morph the scene without memory leaks and return clear warnings for self-correction on invalid fields.

---

### Step 5: Bidirectional AI Perception (`app.describe()` & `app.snapshot()`)
**Goal:** Give the AI "eyes" and spatial awareness of the current scene to eliminate hallucinations in agent loops.

* **Files:** `src/ai/Inspector.ts`, `src/gpu/shaders/id_depth.wgsl`, `src/ai/Snapshot.ts`
* **Details:**
  * **`Inspector.ts` (`app.describe()`):**
    * Projects bounding boxes of all active entities through the current camera view-projection matrix.
    * Determines per object: `visible` (in frustum?), `screenRegion` (`'center'`, `'top-left'`, `'offscreen-right'`, ...), `screenCoverage` (% of viewport), and `relations` (`on`, `near`, `collidingWith`).
    * Emits automatic warnings (`"Entity 'lamp' is outside camera view"`, `"Entity 'chair' intersects with 'table'"`).
  * **`id_depth.wgsl` & `Snapshot.ts` (`app.snapshot()`):**
    * Uses WebGPU Multiple Render Targets (MRT) for an on-demand capture pass:
      * `rgb`: Standard image (`ImageBitmap` / Base64 PNG/WebP).
      * `ids`: `r32uint` render target with the exact numeric `entityId` per pixel (no raycasting).
      * `depth`: Linearized depth map (`0..1` between `near` and `far`).
      * `annotated`: Draws crisp 2D bounding boxes and ID tags (`#table`, `#orb`) onto an offscreen 2D canvas over the RGB snapshot for Vision-Language Models.
* **Acceptance Criterion:** `await app.snapshot({ layers: ['rgb', 'annotated', 'depth', 'ids'] })` returns all 4 layers in < 15ms; `app.describe()` reliably detects occluded or off-screen objects.

---

### Step 6: LLM Schema, Prompt Generator, MCP Bridge & glTF Loader
**Goal:** Plug-and-play integration into local AI workflows (Ollama, LM Studio, MCP servers, Vercel AI SDK) without manual prompt engineering.

* **Files:** `src/ai/Schema.ts`, `src/ai/McpBridge.ts`, `src/entities/GltfLoader.ts`
* **Details:**
  * **`Schema.ts`:**
    * Exports the complete JSON Schema (`THERSE3D_SCENE_SCHEMA`) for Structured Outputs / Tool Calling.
    * Implements `Engine.getPromptContext(app?)`: Generates a compact system prompt (~250–350 tokens) with all primitives, material presets, layout directives, and active scene IDs.
  * **`McpBridge.ts`:**
    * Adapter layer (WebSocket / BroadcastChannel / HTTP poll) making a running browser `therse3d` instance controllable by external AI agents.
    * Provides 4 standardized agent tools:
      1. `therse_sync_scene` → calls `app.sync()` and returns `SyncReport` + `app.describe()`.
      2. `therse_inspect` → calls `app.describe()`.
      3. `therse_snapshot` → returns the `annotated` RGB image for multimodal VLM self-verification.
      4. `therse_camera_focus` → focuses the camera on an `entityId` or view (`'top'`, `'front'`, `'isometric'`).
  * **`GltfLoader.ts`:**
    * Minimal `.glb` (Binary glTF 2.0) loader that loads meshes and PBR materials directly into `WorldStore` & `BufferManager`, allowing complex 3D assets via `{ type: 'model', src: 'robot.glb', on: 'floor' }` in `app.sync()`.
* **Acceptance Criterion:** A local LLM (e.g., Qwen or Gemma via Ollama) can autonomously build an error-free 3D scene in a closed loop (`sync` → `describe` → `snapshot` → fix) using the exported tool schema.
