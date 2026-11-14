# AI-Supervised Smart Factory (`examples/smart-factory`)

An interactive WebGPU demonstration showcasing **therse3d**'s AI-native architecture—specifically declarative scene composition, spatial inspection, auto-healing error recovery, and structured MCP tool workflows compared to manual Three.js boilerplate.

## Files

- `index.html` — Viewport, HUD badges (`Entities`, `AGV Fleet`, `Warnings`), and the AI Supervisor sidebar.
- `factory.js` — Declarative factory definition, AGV patrol loop, click-to-pick/hazard injection, and `McpBridge` tool pipeline.

## Key Engine Features Demonstrated

1. **Declarative Scene & Spatial Layout (`app.sync` + `SpatialLayout`)**
   - Builds 6 warehouse racks via `{ kind: 'grid', columns: 3, spacing: 3.6 }`, cargo crates placed `on` racks, a packing bench placed `beside` the sorting conveyor, a quarantine pad, and two autonomous AGV bots (`agv-1`, `agv-2`) in a single JSON payload.
2. **Closed-Loop AI Supervision (`McpBridge`)**
   - All supervisor actions dispatch through `McpBridge.handleToolCall(...)`:
     - `therse_inspect` — Returns `Inspector.describe()` output with entity positions, `screenRegion`, and spatial `relations` (`on`, `near`, `collidingWith`).
     - `therse_camera_focus` — Smoothly focuses the camera on the detected hazard.
     - `therse_patch_scene` — Relocates `hazard-1` onto `quarantine-pad` using `avoidOverlap: true` and `transition: 0.55`.
     - `therse_snapshot` — Captures an annotated VLM snapshot (`layers: ['rgb', 'annotated']`) and displays it in the sidebar.
     - `therse_undo` — Rolls back the last scene patch.
3. **Auto-Healing & Structured `warnings[]`**
   - **Test Broken AI JSON** sends intentionally malformed patch data (negative/`NaN` scale, unknown material `'unobtanium-9000'`, missing `on` target `'nonexistent-rack-99'`) to show how `therse3d` recovers safely and reports structured `warnings[]` instead of crashing.
4. **Zero-Allocation AGV Loop (`app.onTick`)**
   - `agv-1` and `agv-2` patrol along warehouse lanes via `app.get(...)` and `.setPosition(...)`. When `hazard-1` enters a lane within `1.85m`, the fleet pauses and flashes red beacons until the AI resolves the obstruction.

## Controls

- **Orbit / Zoom:** Drag mouse to orbit, scroll wheel to zoom.
- **Inspect Entity:** Click any 3D object to run `app.pick()` and view its `Inspector.describe()` entry.
- **Place Hazard:** Toggle **Click Mode: Place Hazard** or **Shift + Click** anywhere on the floor, or press **Drop Lane Hazard**.
- **AI Diagnose & Reroute:** Runs the full `therse_inspect` → `therse_camera_focus` → `therse_patch_scene` → `therse_snapshot` pipeline.

## Running

Start the Vite dev server from the repository root (`npm run dev`) and open `/examples/smart-factory/` in a WebGPU-capable browser.

