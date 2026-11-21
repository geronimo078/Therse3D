# Tile-Mesh Engine Plugin (`examples/tile-mesh`)

Interactive WebGPU demonstration of **therse3d**'s optional `TileMeshPlugin` for georeferenced **spherical** and **planar** terrain tiles.

## Files

- `index.html` — Viewport, live tile/vertex/triangle counters, and interactive controls.
- `tile-demo.js` — Closed spherical (`6×4`) and planar (`4×4`) georeferenced tile grids, camera-distance LOD monitoring, `tiles.pick()`, `sampleSurface()`, `deformRegion()` sculpting, and tile lifecycle controls.

## Features Demonstrated

1. **Dual Projection Modes (`spherical` & `planar`)**
   - Switch between a closed `6×4` tiled planet (`planetRadius: 2.0`, `lat [-90°..90°]`) and a flat `4×4` georeferenced terrain grid (`originGeo`, `metersPerDegree`).
2. **Camera-Distance LOD (`updateLOD`)**
   - Orbit or zoom the camera in and out to watch tiles dynamically switch between `24×24`, `12×12`, `6×6`, and `4×4` subdivisions.
3. **Seamless Tile Borders & Skirts**
   - Cross-border finite-difference normals prevent lighting seams across adjacent tiles, and perimeter skirts (`skirtDepth`) hide T-junction cracks between differing LOD levels.
4. **Exact Surface Queries (`sampleSurface`) & Picking (`tiles.pick`)**
   - Click any tile to select it and inspect its `revision`, LOD resolution, and live `sampleSurface(lon, lat)` readout (`elevation`, `slopeDeg`, `land`/`water`).
5. **Local Terrain Deformation (`deformRegion`)**
   - Toggle **Click: Inspect → Raise Hill (+) → Sculpt Crater (-)** to deform terrain heights and `materialMask` live across tile boundaries with in-place GPU vertex buffer patching. Sculpted deformations are preserved automatically when zooming in or out across LOD levels.
6. **Deterministic GPU Buffer Lifecycle**
   - **Replace Selected** (`setTile`), **Remove Selected** (`removeTile`), and **Async Stream Tile** (`loadTile`) immediately unload superseded GPU vertex/index buffers via `engine.unloadMesh()`.

## Running

Start the Vite dev server from the repository root (`npm run dev`) and open `/examples/tile-mesh/index.html` in a WebGPU-capable browser.
