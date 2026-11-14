# therse3d

**An AI-Native WebGPU 3D Engine in TypeScript.**

`therse3d` combines a flat, Three.js-style imperative API and a declarative JSON state-sync API with a data-oriented Entity-Component-System (ECS) backed by contiguous `TypedArrays` and WebGPU storage buffers. It is designed from the ground up for **LLM code generation**, **AI agents (MCP / tool-calling)**, and **Vision-Language Models (VLMs)**.

---

## Features

- **Zero-Dependency Core:** Pure TypeScript + WGSL with zero runtime dependencies.
- **WebGPU-First Instanced PBR Renderer:** Cook-Torrance PBR shading, 3-point studio lighting, hemisphere ambient, 2048px directional shadow mapping, alpha-blended transparent pass, `texture_2d_array` albedo textures (`app.loadTexture()` / `app.registerTexture()`), ACES filmic tone mapping, 4x MSAA, and automatic GPU instancing per geometry type (`Box`, `Sphere`, `Cylinder`, `Plane`, `Line`, `Cone`, `Torus`, `Model`, `Group`, `Light`).
- **Data-Oriented ECS (`WorldStore`):** Contiguous `Float32Array` / `Uint8Array` buffers (`positions`, `rotations`, `scales`, `modelMatrices`, `colors`, `pbrParams`, `boundsMin`, `boundsMax`, `dirtyFlags`) with free-list ID recycling and zero-allocation render-loop updates.
- **Relative Spatial Layout (`SpatialLayout`):** Place objects relative to each other (`on`, `align`, `margin`, `offset`, `lookAt`) with topological sorting instead of manual coordinate math.
- **Declarative `app.sync()` & Auto-Healing:** Idempotent JSON scene synchronization with diffing, smoothstep auto-tweening, and structured `warnings[]` instead of runtime crashes.
- **Bidirectional AI Perception:**
  - `app.describe()` — Token-efficient spatial graph with frustum visibility, `screenRegion`, `screenCoverage`, spatial relations (`on`, `near`, `collidingWith`), and automatic warnings.
  - `app.snapshot()` — On-demand Multiple Render Targets (MRT) capture for `rgb`, `annotated` (2D bounding boxes + `#id` tags), `depth` (`r32float`), and `ids` (`r32uint`).
- **LLM & Agent Integration:** `THERSE3D_SCENE_SCHEMA`, `THERSE3D_TOOL_DEFINITIONS`, `Engine.getPromptContext()`, `McpBridge`, and a binary glTF 2.0 (`.glb`) loader (`GltfLoader`).

---

## Quick Start

### Imperative API

```typescript
import { Engine, Box, Sphere, Plane, OrbitControls } from 'therse3d';

const app = await Engine.create('#canvas', { preset: 'studio' });

app.add(Plane({ id: 'floor', size: [10, 1, 10], color: '#222834' }));
app.add(Box({ id: 'table', size: [2.4, 0.12, 1.2], position: [0, 0.75, 0], material: 'oak-wood' }));
const orb = app.add(Sphere({
  id: 'orb',
  radius: 0.2,
  on: 'table',
  align: 'center',
  color: '#00aaff',
  metalness: 0.75,
  roughness: 0.15,
}));

app.use(OrbitControls());
app.camera.fitAll({ padding: 1.3 });

app.onTick((_dt, elapsed) => {
  orb.position[1] = 1.01 + Math.sin(elapsed * 2.5) * 0.08;
});
```

### Declarative AI-Sync API

```typescript
const report = app.sync({
  preset: 'studio',
  transition: 0.3,
  camera: { target: 'orb', angle: 'front-elevated', distance: 3.5 },
  entities: [
    { id: 'table', type: 'box', size: [2.4, 0.12, 1.2], position: [0, 0.75, 0], material: 'oak-wood' },
    { id: 'orb', type: 'sphere', radius: 0.2, on: 'table', align: 'front-right', color: '#ff3366' },
  ],
});
// => { applied: 2, added: 2, updated: 0, removed: 0, warnings: [] }
```

### AI Inspection & Snapshot

```typescript
const spatialState = app.describe();
const vision = await app.snapshot({
  layers: ['rgb', 'annotated', 'depth', 'ids'],
  resolution: [1024, 768],
});
```

---

## Development

### Prerequisites

- **Node.js** 20+
- A **WebGPU-capable browser** (e.g., Chrome or Edge 113+) for running the interactive samples

### Commands

```bash
npm install
npm run dev        # Start Vite dev server
npm run typecheck  # Strict TypeScript check (tsc --noEmit)
npm test           # Run Vitest unit tests
npm run build      # Typecheck + build ES/CJS bundle in dist/
```

### Samples & Examples

With `npm run dev` running, open:

- `http://localhost:5173/sample/index.html` — Interactive WebGPU 3D sample with preset switcher, `sync()`, `describe()`, and `snapshot()` preview
- `http://localhost:5173/examples/basic.html` — Minimal scene setup
- `http://localhost:5173/examples/ai-sync.html` — Declarative JSON scene sync demo
- `http://localhost:5173/examples/vlm-vision.html` — Spatial inspector & snapshot demo
- `http://localhost:5173/examples/ai-architect.html` — Interactive AI Architect with live JSON `sync()`, `describe()`, and `snapshot()`
- `http://localhost:5173/examples/earth-globe.html` — Living Earth planet with 3D `.glb` terrain & tree assets, textured Moon, atmosphere/auroras, dynamic sun & 2048px shadows

---

## Project Structure

```text
therse3d/
├── Architecture.md         # Master architecture & 6-step specification
├── AI_DEV_GUIDE.md         # Guidelines for AI-assisted development
├── AGENTS.md               # Entry-point instructions for AI coding agents
├── sample/                 # Interactive 3D sample app
├── examples/               # Focused HTML examples
└── src/
    ├── index.ts            # Public API exports
    ├── core/               # Engine, Math, WorldStore, SpatialLayout, SceneSync
    ├── gpu/                # Device, Renderer, BufferManager & WGSL shaders
    ├── entities/           # EntityHandle, Primitives, Materials, GltfLoader
    ├── controls/           # Camera & OrbitControls
    └── ai/                 # Inspector, Snapshot, Schema, McpBridge
```

---

## Documentation

- [Architecture.md](Architecture.md) — Full specification, ECS memory layout, and coding rules
- [AI_DEV_GUIDE.md](AI_DEV_GUIDE.md) — AI-assisted development workflow and runtime agent loop
- [AGENTS.md](AGENTS.md) — Hard constraints and verification checklist for coding agents

## License

MIT

