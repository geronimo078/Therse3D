# Tile-Mesh Engine Plugin (`src/plugins/TileMeshPlugin.ts`)

An optional **therse3d** `EnginePlugin` for georeferenced **spherical** and **planar** terrain tiles with camera-distance LOD, seamless boundaries, exact surface sampling, in-place terrain deformation, and deterministic GPU buffer lifecycle management.

## Quick Start

```typescript
import { Engine, OrbitControls, TileMeshPlugin } from 'therse3d';

const app = await Engine.create('#canvas', { preset: 'outdoor' });
app.use(OrbitControls({ distance: 6.5 }));

const tiles = TileMeshPlugin(app, {
  mode: 'spherical',
  planetRadius: 2.0,
  elevationScale: 0.32,
  waterLevel: 0.0,
  defaultHeightSampler: (lon, lat) => Math.sin(lon * 0.2) * Math.cos(lat * 0.2) * 0.2,
});

tiles.setTile({
  id: 'sector-0-0',
  coord: { z: 2, x: 2, y: 1 }, // or explicit GeoBounds: { minLon, maxLon, minLat, maxLat }
});
```

## Core API (`TileMeshManager`)

- **Tile Lifecycle:**
  - `setTile(input: TileInput): TileRecord` — Creates or replaces a tile (`coord` or `bounds`) and immediately unloads any previous GPU vertex/index buffer (`engine.unloadMesh`).
  - `loadTile(id, asyncProvider): Promise<TileRecord | null>` — Asynchronously streams a tile while ignoring stale completions if the tile is removed or superseded.
  - `removeTile(id): boolean` / `clearTiles(): void` — Removes tile entities and destroys their GPU buffers.
  - `getTile(id)` / `listTiles()` — Inspects active tiles (`revision`, `segments`, `bounds`, `vertexCount`, `triangleCount`).
- **Camera-Distance LOD & Seamless Boundaries:**
  - `updateLOD(cameraPosition?)` — Automatically runs via `engine.onTick` when `autoLod: true` and rebuilds tiles only when crossing `lodLevels` thresholds.
  - Cross-border central finite-difference normals + perimeter `skirtDepth` prevent lighting seams and T-junction cracks between differing LODs.
- **Surface Queries (`sampleSurface`):**
  - `sampleSurface(lon, lat): SurfaceSample | null` — Barycentrically interpolates the **actual rendered triangle** at `(lon, lat)`, returning `elevation`, `worldPosition`, `normal`, `slope`, `slopeDeg`, `surfaceType` (`'land' | 'water'`), `isWater`, `materialMask`, `tileId`, and `revision`.
- **In-Place Terrain Deformation (`deformRegion`):**
  - `deformRegion(options: TileDeformOptions): TileDeformResult` — Locally modifies heights (`'add' | 'flatten' | 'smooth'`) and `materialMask` across all intersecting tiles, recomputes cross-border normals and skirt edges, patches the existing GPU vertex buffer in-place (`engine.updateMeshVertices`), updates world bounds, and increments `revision`. Recorded deformations are automatically replayed across `updateLOD()` resolution transitions so edits are never lost when zooming in or out.
- **Picking & Cleanup:**
  - `pick(x, y): TilePickHit | null` — Ray-casts against each tile's true world AABB.
  - `dispose(): void` — Automatically invoked when `app.dispose()` runs, freeing all GPU buffers and `onTick` listeners.
