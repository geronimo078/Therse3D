import { mat4 } from '../core/Math';
import { DirtyFlag } from '../core/WorldStore';
import type { Engine, EnginePlugin } from '../core/Engine';
import type { EntityHandle } from '../entities/EntityHandle';
import {
  parseColor,
  type MaterialDefinition,
  type MaterialPresetName,
} from '../entities/Materials';
import { Model, type GeometryMeshData, type PrimitiveOptions } from '../entities/Primitives';

export type TileProjectionMode = 'planar' | 'spherical';
export type SurfaceType = 'land' | 'water';

export interface SurfaceSample {
  tileId: string;
  revision: number;
  lon: number;
  lat: number;
  elevation: number;
  worldPosition: [number, number, number];
  normal: [number, number, number];
  slope: number;
  slopeDeg: number;
  surfaceType: SurfaceType;
  isWater: boolean;
  materialMask: number;
}

export interface TileDeformOptions {
  lon: number;
  lat: number;
  radiusDeg: number;
  deltaElevation?: number;
  targetElevation?: number;
  mode?: 'add' | 'flatten' | 'smooth';
  falloff?: 'smooth' | 'linear';
  materialMaskValue?: number;
  materialMaskMode?: 'set' | 'max' | 'add';
}

export interface TileDeformResult {
  updatedTiles: string[];
  patchedVertices: number;
}

export interface TileCoord {
  z: number;
  x: number;
  y: number;
}

export interface GeoBounds {
  minLon: number;
  maxLon: number;
  minLat: number;
  maxLat: number;
}

export interface TileLodLevel {
  maxDistance: number;
  segments: number;
}

export type TileHeightSampler = (lonDeg: number, latDeg: number) => number;

export interface TileInput {
  id: string;
  coord?: TileCoord;
  bounds?: GeoBounds;
  heightSampler?: TileHeightSampler;
  segments?: number;
  color?: string | [number, number, number] | [number, number, number, number];
  material?: MaterialPresetName | Partial<MaterialDefinition>;
  texture?: string | number;
  roughness?: number;
  metalness?: number;
  skirtDepth?: number;
  visible?: boolean;
  waterLevel?: number;
}

export interface TileRecord {
  id: string;
  revision: number;
  bounds: GeoBounds;
  centerWorld: [number, number, number];
  segments: number;
  meshId: number;
  entity: EntityHandle;
  vertexCount: number;
  triangleCount: number;
}

export interface TileMeshPluginOptions {
  mode?: TileProjectionMode;
  planetRadius?: number;
  planetCenter?: readonly [number, number, number];
  originGeo?: readonly [number, number];
  metersPerDegree?: number;
  elevationScale?: number;
  defaultSegments?: number;
  skirtDepth?: number;
  waterLevel?: number;
  autoLod?: boolean;
  lodLevels?: readonly TileLodLevel[];
  defaultHeightSampler?: TileHeightSampler;
}

export interface BuildTileMeshOptions {
  bounds: GeoBounds;
  segments: number;
  mode: TileProjectionMode;
  planetRadius: number;
  planetCenter: readonly [number, number, number];
  originGeo: readonly [number, number];
  metersPerDegree: number;
  elevationScale: number;
  skirtDepth: number;
  heightSampler: TileHeightSampler;
}

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

export const DEFAULT_TILE_LOD_LEVELS: readonly TileLodLevel[] = [
  { maxDistance: 6, segments: 32 },
  { maxDistance: 14, segments: 16 },
  { maxDistance: 28, segments: 8 },
  { maxDistance: Infinity, segments: 4 },
];

/**
 * Converts a Slippy-Map z/x/y tile coordinate into geographic bounds (degrees).
 */
export function tileCoordToGeoBounds(coord: TileCoord): GeoBounds {
  const z = Math.max(0, Math.min(22, Math.floor(coord.z)));
  const n = Math.pow(2, z);
  const x = Math.max(0, Math.min(n - 1, Math.floor(coord.x)));
  const y = Math.max(0, Math.min(n - 1, Math.floor(coord.y)));

  const minLon = (x / n) * 360 - 180;
  const maxLon = ((x + 1) / n) * 360 - 180;

  const latRadTop = Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n)));
  const latRadBot = Math.atan(Math.sinh(Math.PI * (1 - (2 * (y + 1)) / n)));

  return {
    minLon,
    maxLon,
    minLat: latRadBot * RAD_TO_DEG,
    maxLat: latRadTop * RAD_TO_DEG,
  };
}

/**
 * Projects geographic (lonDeg, latDeg, elevation) into 3D world space for planar or spherical mode.
 */
export function geoToWorldPoint(
  lonDeg: number,
  latDeg: number,
  elevation: number,
  options: Pick<
    BuildTileMeshOptions,
    'mode' | 'planetRadius' | 'planetCenter' | 'originGeo' | 'metersPerDegree' | 'elevationScale'
  >,
  out: [number, number, number] = [0, 0, 0],
): [number, number, number] {
  const elev = elevation * options.elevationScale;
  if (options.mode === 'spherical') {
    const lon = lonDeg * DEG_TO_RAD;
    const lat = latDeg * DEG_TO_RAD;
    const cosLat = Math.cos(lat);
    const r = Math.max(0.01, options.planetRadius + elev);
    out[0] = (options.planetCenter[0] ?? 0) + r * cosLat * Math.cos(lon);
    out[1] = (options.planetCenter[1] ?? 0) + r * Math.sin(lat);
    out[2] = (options.planetCenter[2] ?? 0) - r * cosLat * Math.sin(lon);
    return out;
  }

  const lon0 = options.originGeo[0] ?? 0;
  const lat0 = options.originGeo[1] ?? 0;
  const cosLat0 = Math.max(0.1, Math.cos(lat0 * DEG_TO_RAD));
  out[0] = (lonDeg - lon0) * options.metersPerDegree * cosLat0;
  out[1] = elev;
  out[2] = -(latDeg - lat0) * options.metersPerDegree;
  return out;
}

/**
 * Builds a seamless georeferenced terrain tile mesh with cross-border finite-difference normals
 * and perimeter skirts to prevent LOD T-junction cracks.
 */
export function buildTileMeshData(opts: BuildTileMeshOptions): GeometryMeshData {
  const seg = Math.max(1, Math.min(128, Math.round(opts.segments)));
  const { minLon, maxLon, minLat, maxLat } = opts.bounds;
  const lonSpan = maxLon - minLon;
  const latSpan = maxLat - minLat;

  const gridCount = (seg + 1) * (seg + 1);
  // 4 perimeter edges * (seg + 1) skirt vertices when skirtDepth > 0
  const hasSkirt = opts.skirtDepth > 0;
  const edgeLen = seg + 1;
  const totalVerts = hasSkirt ? gridCount + edgeLen * 4 : gridCount;
  const gridTris = seg * seg * 2;
  const skirtTris = hasSkirt ? seg * 2 * 4 : 0;

  const vertices = new Float32Array(totalVerts * 8);
  const indices = new Uint32Array((gridTris + skirtTris) * 3);

  const epsLon = Math.max(1e-6, Math.abs(lonSpan) / (seg * 2));
  const epsLat = Math.max(1e-6, Math.abs(latSpan) / (seg * 2));
  const pL: [number, number, number] = [0, 0, 0];
  const pR: [number, number, number] = [0, 0, 0];
  const pD: [number, number, number] = [0, 0, 0];
  const pU: [number, number, number] = [0, 0, 0];
  const pC: [number, number, number] = [0, 0, 0];

  const writeVertex = (
    vIndex: number,
    px: number, py: number, pz: number,
    nx: number, ny: number, nz: number,
    u: number, v: number,
  ) => {
    const o = vIndex * 8;
    vertices[o] = px;
    vertices[o + 1] = py;
    vertices[o + 2] = pz;
    vertices[o + 3] = nx;
    vertices[o + 4] = ny;
    vertices[o + 5] = nz;
    vertices[o + 6] = u;
    vertices[o + 7] = v;
  };

  for (let iy = 0; iy <= seg; iy++) {
    const v = iy / seg;
    // iy=0 is north (maxLat), iy=seg is south (minLat) for standard CCW winding
    const lat = maxLat - v * latSpan;
    for (let ix = 0; ix <= seg; ix++) {
      const u = ix / seg;
      const lon = minLon + u * lonSpan;

      const h = opts.heightSampler(lon, lat);
      geoToWorldPoint(lon, lat, h, opts, pC);

      // Seamless cross-tile normals via central finite differences in geo space
      const hL = opts.heightSampler(lon - epsLon, lat);
      const hR = opts.heightSampler(lon + epsLon, lat);
      const hD = opts.heightSampler(lon, lat - epsLat);
      const hU = opts.heightSampler(lon, lat + epsLat);
      geoToWorldPoint(lon - epsLon, lat, hL, opts, pL);
      geoToWorldPoint(lon + epsLon, lat, hR, opts, pR);
      geoToWorldPoint(lon, lat - epsLat, hD, opts, pD);
      geoToWorldPoint(lon, lat + epsLat, hU, opts, pU);

      // Tangent U (+lon = +X planar, east spherical), Tangent V (+lat = -Z planar, north spherical)
      const tux = pR[0] - pL[0], tuy = pR[1] - pL[1], tuz = pR[2] - pL[2];
      const tvx = pU[0] - pD[0], tvy = pU[1] - pD[1], tvz = pU[2] - pD[2];

      // normal = cross(tu, tv) so planar (+X) x (-Z) = +Y, spherical (-Z) x (+Y) = +X (outward)
      let nx = tuy * tvz - tuz * tvy;
      let ny = tuz * tvx - tux * tvz;
      let nz = tux * tvy - tuy * tvx;
      const nLen = Math.hypot(nx, ny, nz);
      if (nLen > 1e-5) {
        nx /= nLen;
        ny /= nLen;
        nz /= nLen;
      } else if (opts.mode === 'spherical') {
        const rx = pC[0] - (opts.planetCenter[0] ?? 0);
        const ry = pC[1] - (opts.planetCenter[1] ?? 0);
        const rz = pC[2] - (opts.planetCenter[2] ?? 0);
        const rLen = Math.hypot(rx, ry, rz) || 1;
        nx = rx / rLen;
        ny = ry / rLen;
        nz = rz / rLen;
      } else {
        nx = 0; ny = 1; nz = 0;
      }

      writeVertex(iy * edgeLen + ix, pC[0], pC[1], pC[2], nx, ny, nz, u, 1 - v);
    }
  }

  let idx = 0;
  for (let iy = 0; iy < seg; iy++) {
    for (let ix = 0; ix < seg; ix++) {
      const a = iy * edgeLen + ix;
      const b = a + edgeLen;
      const c = a + 1;
      const d = b + 1;
      // CCW outward
      indices[idx++] = a; indices[idx++] = b; indices[idx++] = c;
      indices[idx++] = c; indices[idx++] = b; indices[idx++] = d;
    }
  }

  if (hasSkirt) {
    const drop = opts.skirtDepth;
    let skirtBase = gridCount;

    const addSkirtEdge = (getGridIndex: (k: number) => number, flipWinding: boolean) => {
      const start = skirtBase;
      for (let k = 0; k < edgeLen; k++) {
        const gi = getGridIndex(k);
        const go = gi * 8;
        const px = vertices[go]!;
        const py = vertices[go + 1]!;
        const pz = vertices[go + 2]!;
        const nx = vertices[go + 3]!;
        const ny = vertices[go + 4]!;
        const nz = vertices[go + 5]!;
        const u = vertices[go + 6]!;
        const v = vertices[go + 7]!;

        let sx = px, sy = py - drop, sz = pz;
        if (opts.mode === 'spherical') {
          const cx = opts.planetCenter[0] ?? 0;
          const cy = opts.planetCenter[1] ?? 0;
          const cz = opts.planetCenter[2] ?? 0;
          const rx = px - cx, ry = py - cy, rz = pz - cz;
          const rLen = Math.hypot(rx, ry, rz) || 1;
          sx = px - (rx / rLen) * drop;
          sy = py - (ry / rLen) * drop;
          sz = pz - (rz / rLen) * drop;
        }
        writeVertex(skirtBase++, sx, sy, sz, nx, ny, nz, u, v);
      }

      for (let k = 0; k < seg; k++) {
        const g0 = getGridIndex(k);
        const g1 = getGridIndex(k + 1);
        const s0 = start + k;
        const s1 = start + k + 1;
        if (!flipWinding) {
          indices[idx++] = g0; indices[idx++] = s0; indices[idx++] = g1;
          indices[idx++] = g1; indices[idx++] = s0; indices[idx++] = s1;
        } else {
          indices[idx++] = g0; indices[idx++] = g1; indices[idx++] = s0;
          indices[idx++] = g1; indices[idx++] = s1; indices[idx++] = s0;
        }
      }
    };

    // North (iy=0, left->right), South (iy=seg, left->right), West (ix=0, top->bottom), East (ix=seg, top->bottom)
    addSkirtEdge((k) => k, true);
    addSkirtEdge((k) => seg * edgeLen + k, false);
    addSkirtEdge((k) => k * edgeLen, false);
    addSkirtEdge((k) => k * edgeLen + seg, true);
  }

  return { vertices, indices };
}

interface InternalTileState {
  input: TileInput;
  revision: number;
  bounds: GeoBounds;
  centerWorld: [number, number, number];
  worldMin: [number, number, number];
  worldMax: [number, number, number];
  segments: number;
  fixedSegments: boolean;
  meshId: number;
  meshData: GeometryMeshData;
  elevations: Float32Array;
  materialMask: Float32Array;
  entity: EntityHandle;
  vertexCount: number;
  triangleCount: number;
}

export interface TilePickHit {
  id: string;
  distance: number;
  point: [number, number, number];
}

/**
 * Optional Engine plugin managing georeferenced planar/spherical terrain tiles,
 * camera-distance LOD, seamless boundaries, and deterministic GPU buffer lifecycle.
 */
export class TileMeshManager {
  private readonly engine: Engine;
  readonly mode: TileProjectionMode;
  readonly planetRadius: number;
  readonly planetCenter: [number, number, number];
  readonly originGeo: [number, number];
  readonly metersPerDegree: number;
  readonly elevationScale: number;
  readonly defaultSegments: number;
  readonly skirtDepth: number;
  readonly waterLevel: number;
  readonly lodLevels: TileLodLevel[];
  private readonly defaultHeightSampler: TileHeightSampler;

  private readonly tiles = new Map<string, InternalTileState>();
  private readonly pendingLoads = new Map<string, number>();
  private nextLoadToken = 1;
  private autoLod: boolean;
  private unsubTick: (() => void) | null = null;
  private disposed = false;

  constructor(engine: Engine, options: TileMeshPluginOptions = {}) {
    this.engine = engine;
    this.mode = options.mode ?? 'planar';
    this.planetRadius = Math.max(0.1, options.planetRadius ?? 2.0);
    this.planetCenter = [
      options.planetCenter?.[0] ?? 0,
      options.planetCenter?.[1] ?? 0,
      options.planetCenter?.[2] ?? 0,
    ];
    this.originGeo = [options.originGeo?.[0] ?? 0, options.originGeo?.[1] ?? 0];
    this.metersPerDegree = Math.max(0.01, options.metersPerDegree ?? 1.0);
    this.elevationScale = options.elevationScale ?? 1.0;
    this.defaultSegments = Math.max(1, Math.round(options.defaultSegments ?? 16));
    this.skirtDepth = Math.max(0, options.skirtDepth ?? 0.08);
    this.waterLevel = options.waterLevel ?? 0;
    this.autoLod = options.autoLod ?? true;
    this.lodLevels = [...(options.lodLevels ?? DEFAULT_TILE_LOD_LEVELS)].sort(
      (a, b) => a.maxDistance - b.maxDistance,
    );
    this.defaultHeightSampler = options.defaultHeightSampler ?? (() => 0);

    if (this.autoLod) {
      this.unsubTick = this.engine.onTick(() => {
        this.updateLOD();
      });
    }
  }

  private resolveBounds(input: TileInput): GeoBounds {
    if (input.bounds) return { ...input.bounds };
    if (input.coord) return tileCoordToGeoBounds(input.coord);
    throw new Error(`Tile "${input.id}" requires either "coord" or "bounds".`);
  }

  private selectSegmentsForDistance(dist: number): number {
    for (let i = 0; i < this.lodLevels.length; i++) {
      const level = this.lodLevels[i]!;
      if (dist <= level.maxDistance) {
        return Math.max(1, Math.round(level.segments));
      }
    }
    return this.defaultSegments;
  }

  private buildMeshForTile(bounds: GeoBounds, segments: number, input: TileInput): GeometryMeshData {
    return buildTileMeshData({
      bounds,
      segments,
      mode: this.mode,
      planetRadius: this.planetRadius,
      planetCenter: this.planetCenter,
      originGeo: this.originGeo,
      metersPerDegree: this.metersPerDegree,
      elevationScale: this.elevationScale,
      skirtDepth: input.skirtDepth ?? this.skirtDepth,
      heightSampler: input.heightSampler ?? this.defaultHeightSampler,
    });
  }

  /**
   * Creates or replaces a georeferenced tile and deterministically frees any previous GPU mesh buffer.
   */
  setTile(input: TileInput): TileRecord {
    if (this.disposed) {
      throw new Error('TileMeshManager has been disposed.');
    }

    // Cancel any in-flight async load for this tile ID
    this.pendingLoads.delete(input.id);

    const bounds = this.resolveBounds(input);
    const sampler = input.heightSampler ?? this.defaultHeightSampler;
    const centerLon = (bounds.minLon + bounds.maxLon) * 0.5;
    const centerLat = (bounds.minLat + bounds.maxLat) * 0.5;
    const centerElev = sampler(centerLon, centerLat);
    const centerWorld = geoToWorldPoint(centerLon, centerLat, centerElev, {
      mode: this.mode,
      planetRadius: this.planetRadius,
      planetCenter: this.planetCenter,
      originGeo: this.originGeo,
      metersPerDegree: this.metersPerDegree,
      elevationScale: this.elevationScale,
    });

    const camPos = this.engine.camera.position;
    const dist = Math.hypot(
      (camPos[0] ?? 0) - centerWorld[0],
      (camPos[1] ?? 0) - centerWorld[1],
      (camPos[2] ?? 0) - centerWorld[2],
    );
    const fixedSegments = input.segments !== undefined;
    const segments = fixedSegments
      ? Math.max(1, Math.round(input.segments!))
      : this.autoLod
        ? this.selectSegmentsForDistance(dist)
        : this.defaultSegments;

    const meshData = this.buildMeshForTile(bounds, segments, input);
    const { elevations, materialMask } = this.sampleGridArrays(bounds, segments, input.heightSampler ?? this.defaultHeightSampler);
    const { min: worldMin, max: worldMax } = computeMeshBounds(meshData.vertices);
    const newMeshId = this.engine.registerMesh(meshData, `tile-${input.id}-lod${segments}`);
    const vertexCount = meshData.vertices.length / 8;
    const triangleCount = meshData.indices.length / 3;

    const existing = this.tiles.get(input.id);
    if (existing) {
      const oldMeshId = existing.meshId;
      this.engine.setEntityMesh(existing.entity, newMeshId);
      this.engine.unloadMesh(oldMeshId);

      if (input.color !== undefined) existing.entity.color = parseColor(input.color);
      if (input.visible !== undefined) existing.entity.visible = input.visible;
      if (this.engine.store && (input.roughness !== undefined || input.metalness !== undefined)) {
        const pOff = existing.entity.id * 4;
        if (input.metalness !== undefined) this.engine.store.pbrParams[pOff] = input.metalness;
        if (input.roughness !== undefined) this.engine.store.pbrParams[pOff + 1] = input.roughness;
        this.engine.store.markDirty(existing.entity.id, DirtyFlag.Material);
      }

      existing.input = input;
      existing.revision += 1;
      existing.bounds = bounds;
      existing.centerWorld = centerWorld;
      existing.worldMin = worldMin;
      existing.worldMax = worldMax;
      existing.segments = segments;
      existing.fixedSegments = fixedSegments;
      existing.meshId = newMeshId;
      existing.meshData = meshData;
      existing.elevations = elevations;
      existing.materialMask = materialMask;
      existing.vertexCount = vertexCount;
      existing.triangleCount = triangleCount;
      return this.toRecord(existing);
    }

    const modelOpts: PrimitiveOptions = {
      id: `tile:${input.id}`,
      position: [0, 0, 0],
    };
    if (input.color !== undefined) modelOpts.color = input.color;
    if (input.material !== undefined) modelOpts.material = input.material;
    if (input.texture !== undefined) modelOpts.texture = input.texture;
    if (input.roughness !== undefined) modelOpts.roughness = input.roughness;
    if (input.metalness !== undefined) modelOpts.metalness = input.metalness;
    if (input.visible !== undefined) modelOpts.visible = input.visible;

    const entity = this.engine.add(Model(modelOpts));
    this.engine.setEntityMesh(entity, newMeshId);

    const state: InternalTileState = {
      input,
      revision: 1,
      bounds,
      centerWorld,
      worldMin,
      worldMax,
      segments,
      fixedSegments,
      meshId: newMeshId,
      meshData,
      elevations,
      materialMask,
      entity,
      vertexCount,
      triangleCount,
    };
    this.tiles.set(input.id, state);
    return this.toRecord(state);
  }

  /**
   * Asynchronously loads and applies a tile from a provider, safely ignoring stale completions
   * if the tile is removed or superseded while loading.
   */
  async loadTile(
    id: string,
    provider: (id: string) => Promise<Omit<TileInput, 'id'> | TileInput>,
  ): Promise<TileRecord | null> {
    if (this.disposed) return null;
    const token = this.nextLoadToken++;
    this.pendingLoads.set(id, token);

    const data = await provider(id);
    if (this.disposed || this.pendingLoads.get(id) !== token) {
      return null;
    }
    this.pendingLoads.delete(id);
    return this.setTile({ ...data, id });
  }

  /**
   * Removes a tile entity and immediately destroys its GPU vertex/index buffers.
   */
  removeTile(id: string): boolean {
    this.pendingLoads.delete(id);
    const state = this.tiles.get(id);
    if (!state) return false;
    this.tiles.delete(id);
    this.engine.remove(state.entity);
    this.engine.unloadMesh(state.meshId);
    return true;
  }

  clearTiles(): void {
    this.pendingLoads.clear();
    for (const id of Array.from(this.tiles.keys())) {
      this.removeTile(id);
    }
  }

  getTile(id: string): TileRecord | undefined {
    const state = this.tiles.get(id);
    return state ? this.toRecord(state) : undefined;
  }

  listTiles(): TileRecord[] {
    return Array.from(this.tiles.values()).map((s) => this.toRecord(s));
  }

  /**
   * Re-evaluates camera distance for all dynamic-LOD tiles and swaps GPU meshes only when resolution changes.
   */
  updateLOD(cameraPosition?: readonly [number, number, number]): number {
    if (this.disposed || this.tiles.size === 0) return 0;
    const camPos = cameraPosition ?? this.engine.camera.position;
    const cx = camPos[0] ?? 0;
    const cy = camPos[1] ?? 0;
    const cz = camPos[2] ?? 0;

    let updated = 0;
    for (const state of this.tiles.values()) {
      if (state.fixedSegments) continue;
      const dist = Math.hypot(
        cx - state.centerWorld[0],
        cy - state.centerWorld[1],
        cz - state.centerWorld[2],
      );
      const targetSegments = this.selectSegmentsForDistance(dist);
      if (targetSegments === state.segments) continue;

      const meshData = this.buildMeshForTile(state.bounds, targetSegments, state.input);
      const { elevations, materialMask } = this.sampleGridArrays(
        state.bounds,
        targetSegments,
        state.input.heightSampler ?? this.defaultHeightSampler,
      );
      const { min: worldMin, max: worldMax } = computeMeshBounds(meshData.vertices);
      const newMeshId = this.engine.registerMesh(meshData, `tile-${state.input.id}-lod${targetSegments}`);
      const oldMeshId = state.meshId;
      this.engine.setEntityMesh(state.entity, newMeshId);
      this.engine.unloadMesh(oldMeshId);

      state.revision += 1;
      state.segments = targetSegments;
      state.meshId = newMeshId;
      state.meshData = meshData;
      state.elevations = elevations;
      state.materialMask = materialMask;
      state.worldMin = worldMin;
      state.worldMax = worldMax;
      state.vertexCount = meshData.vertices.length / 8;
      state.triangleCount = meshData.indices.length / 3;
      updated++;
    }
    return updated;
  }

  /**
   * Samples the exact rendered tile triangle at (lon, lat) using barycentric interpolation,
   * returning elevation, surface normal, slope, land/water classification, and tile revision.
   */
  sampleSurface(lon: number, lat: number): SurfaceSample | null {
    if (this.disposed || this.tiles.size === 0) return null;

    let match: InternalTileState | null = null;
    for (const state of this.tiles.values()) {
      const b = state.bounds;
      if (lon >= b.minLon && lon <= b.maxLon && lat >= b.minLat && lat <= b.maxLat) {
        if (!match || state.segments > match.segments) {
          match = state;
        }
      }
    }
    if (!match) return null;

    const seg = match.segments;
    const edgeLen = seg + 1;
    const { minLon, maxLon, minLat, maxLat } = match.bounds;
    const u = Math.max(0, Math.min(1, (lon - minLon) / Math.max(1e-9, maxLon - minLon)));
    const v = Math.max(0, Math.min(1, (maxLat - lat) / Math.max(1e-9, maxLat - minLat)));

    const fx = u * seg;
    const fy = v * seg;
    const ix = Math.min(seg - 1, Math.floor(fx));
    const iy = Math.min(seg - 1, Math.floor(fy));
    const tx = fx - ix;
    const ty = fy - iy;

    const a = iy * edgeLen + ix;
    const b = a + edgeLen;
    const c = a + 1;
    const d = b + 1;

    let i0: number, i1: number, i2: number;
    let w0: number, w1: number, w2: number;
    if (tx + ty <= 1) {
      i0 = a; i1 = b; i2 = c;
      w0 = 1 - tx - ty;
      w1 = ty;
      w2 = tx;
    } else {
      i0 = c; i1 = b; i2 = d;
      w0 = 1 - ty;
      w1 = 1 - tx;
      w2 = tx + ty - 1;
    }

    const verts = match.meshData.vertices;
    const o0 = i0 * 8;
    const o1 = i1 * 8;
    const o2 = i2 * 8;

    const p0x = verts[o0] ?? 0, p0y = verts[o0 + 1] ?? 0, p0z = verts[o0 + 2] ?? 0;
    const p1x = verts[o1] ?? 0, p1y = verts[o1 + 1] ?? 0, p1z = verts[o1 + 2] ?? 0;
    const p2x = verts[o2] ?? 0, p2y = verts[o2 + 1] ?? 0, p2z = verts[o2 + 2] ?? 0;

    const px = p0x * w0 + p1x * w1 + p2x * w2;
    const py = p0y * w0 + p1y * w1 + p2y * w2;
    const pz = p0z * w0 + p1z * w1 + p2z * w2;

    let nx = (verts[o0 + 3] ?? 0) * w0 + (verts[o1 + 3] ?? 0) * w1 + (verts[o2 + 3] ?? 0) * w2;
    let ny = (verts[o0 + 4] ?? 0) * w0 + (verts[o1 + 4] ?? 0) * w1 + (verts[o2 + 4] ?? 0) * w2;
    let nz = (verts[o0 + 5] ?? 0) * w0 + (verts[o1 + 5] ?? 0) * w1 + (verts[o2 + 5] ?? 0) * w2;
    const nLen = Math.hypot(nx, ny, nz) || 1;
    nx /= nLen;
    ny /= nLen;
    nz /= nLen;

    // Exact geometric normal of the rendered triangle for true surface slope
    const e1x = p1x - p0x, e1y = p1y - p0y, e1z = p1z - p0z;
    const e2x = p2x - p0x, e2y = p2y - p0y, e2z = p2z - p0z;
    let fnx = e1y * e2z - e1z * e2y;
    let fny = e1z * e2x - e1x * e2z;
    let fnz = e1x * e2y - e1y * e2x;
    const fnLen = Math.hypot(fnx, fny, fnz) || 1;
    fnx /= fnLen;
    fny /= fnLen;
    fnz /= fnLen;

    let elevation: number;
    let upX = 0, upY = 1, upZ = 0;
    if (this.mode === 'spherical') {
      const rx = px - this.planetCenter[0];
      const ry = py - this.planetCenter[1];
      const rz = pz - this.planetCenter[2];
      const rLen = Math.hypot(rx, ry, rz) || 1;
      elevation = rLen - this.planetRadius;
      upX = rx / rLen;
      upY = ry / rLen;
      upZ = rz / rLen;
    } else {
      elevation = py;
    }

    const cosSlope = Math.max(-1, Math.min(1, Math.abs(fnx * upX + fny * upY + fnz * upZ)));
    const slopeRad = Math.acos(cosSlope);
    const slopeDeg = slopeRad * RAD_TO_DEG;
    const slope = Math.sin(slopeRad);

    const tileWaterLevel = match.input.waterLevel ?? this.waterLevel;
    const isWater = elevation <= tileWaterLevel;
    const maskVal =
      (match.materialMask[i0] ?? 0) * w0 +
      (match.materialMask[i1] ?? 0) * w1 +
      (match.materialMask[i2] ?? 0) * w2;

    return {
      tileId: match.input.id,
      revision: match.revision,
      lon,
      lat,
      elevation,
      worldPosition: [px, py, pz],
      normal: [nx, ny, nz],
      slope,
      slopeDeg,
      surfaceType: isWater ? 'water' : 'land',
      isWater,
      materialMask: maskVal,
    };
  }

  private sampleGridArrays(
    bounds: GeoBounds,
    segments: number,
    heightSampler: TileHeightSampler,
  ): { elevations: Float32Array; materialMask: Float32Array } {
    const edgeLen = segments + 1;
    const count = edgeLen * edgeLen;
    const elevations = new Float32Array(count);
    const materialMask = new Float32Array(count);
    const lonSpan = bounds.maxLon - bounds.minLon;
    const latSpan = bounds.maxLat - bounds.minLat;

    for (let iy = 0; iy <= segments; iy++) {
      const lat = bounds.maxLat - (iy / segments) * latSpan;
      for (let ix = 0; ix <= segments; ix++) {
        const lon = bounds.minLon + (ix / segments) * lonSpan;
        elevations[iy * edgeLen + ix] = heightSampler(lon, lat);
      }
    }
    return { elevations, materialMask };
  }

  private sampleElevationAtGeo(lon: number, lat: number, fallback: TileHeightSampler): number {
    for (const state of this.tiles.values()) {
      const b = state.bounds;
      if (lon >= b.minLon && lon <= b.maxLon && lat >= b.minLat && lat <= b.maxLat) {
        const seg = state.segments;
        const edgeLen = seg + 1;
        const u = Math.max(0, Math.min(1, (lon - b.minLon) / Math.max(1e-9, b.maxLon - b.minLon)));
        const v = Math.max(0, Math.min(1, (b.maxLat - lat) / Math.max(1e-9, b.maxLat - b.minLat)));
        const fx = u * seg;
        const fy = v * seg;
        const ix0 = Math.min(seg - 1, Math.floor(fx));
        const iy0 = Math.min(seg - 1, Math.floor(fy));
        const tx = fx - ix0;
        const ty = fy - iy0;
        const e00 = state.elevations[iy0 * edgeLen + ix0] ?? 0;
        const e10 = state.elevations[iy0 * edgeLen + ix0 + 1] ?? 0;
        const e01 = state.elevations[(iy0 + 1) * edgeLen + ix0] ?? 0;
        const e11 = state.elevations[(iy0 + 1) * edgeLen + ix0 + 1] ?? 0;
        return (e00 * (1 - tx) + e10 * tx) * (1 - ty) + (e01 * (1 - tx) + e11 * tx) * ty;
      }
    }
    return fallback(lon, lat);
  }

  /**
   * Deforms a localized geographic region across all affected tiles, updating heights,
   * material masks, cross-border normals, perimeter skirts, GPU vertex buffers in-place,
   * world bounds, and tile revisions.
   */
  deformRegion(options: TileDeformOptions): TileDeformResult {
    if (this.disposed || this.tiles.size === 0) {
      return { updatedTiles: [], patchedVertices: 0 };
    }

    const r = Math.max(1e-5, options.radiusDeg);
    const mode = options.mode ?? 'add';
    const falloffMode = options.falloff ?? 'smooth';
    const delta = options.deltaElevation ?? 0;
    const targetElev = options.targetElevation ?? 0;
    const maskVal = options.materialMaskValue;
    const maskMode = options.materialMaskMode ?? 'set';

    const updatedTiles: string[] = [];
    let patchedVertices = 0;

    // Pass 1: Update heights & material masks on all intersecting tiles
    const dirtyTiles: Array<{
      state: InternalTileState;
      minIx: number;
      maxIx: number;
      minIy: number;
      maxIy: number;
    }> = [];

    for (const state of this.tiles.values()) {
      const b = state.bounds;
      if (
        options.lon + r < b.minLon ||
        options.lon - r > b.maxLon ||
        options.lat + r < b.minLat ||
        options.lat - r > b.maxLat
      ) {
        continue;
      }

      const seg = state.segments;
      const edgeLen = seg + 1;
      const lonSpan = b.maxLon - b.minLon;
      const latSpan = b.maxLat - b.minLat;

      let minIx = seg, maxIx = 0, minIy = seg, maxIy = 0;
      let anyHit = false;

      for (let iy = 0; iy <= seg; iy++) {
        const lat = b.maxLat - (iy / seg) * latSpan;
        for (let ix = 0; ix <= seg; ix++) {
          const lon = b.minLon + (ix / seg) * lonSpan;
          const dist = Math.hypot(lon - options.lon, lat - options.lat);
          if (dist > r) continue;

          anyHit = true;
          if (ix < minIx) minIx = ix;
          if (ix > maxIx) maxIx = ix;
          if (iy < minIy) minIy = iy;
          if (iy > maxIy) maxIy = iy;

          const t = 1 - dist / r;
          const w = falloffMode === 'linear' ? t : t * t * (3 - 2 * t);
          const idx = iy * edgeLen + ix;
          const curH = state.elevations[idx] ?? 0;

          if (mode === 'flatten') {
            state.elevations[idx] = curH + (targetElev - curH) * w;
          } else if (mode === 'smooth') {
            const hL = state.elevations[iy * edgeLen + Math.max(0, ix - 1)] ?? curH;
            const hR = state.elevations[iy * edgeLen + Math.min(seg, ix + 1)] ?? curH;
            const hU = state.elevations[Math.max(0, iy - 1) * edgeLen + ix] ?? curH;
            const hD = state.elevations[Math.min(seg, iy + 1) * edgeLen + ix] ?? curH;
            const avg = (hL + hR + hU + hD) * 0.25;
            state.elevations[idx] = curH + (avg - curH) * w;
          } else {
            state.elevations[idx] = curH + delta * w;
          }

          if (maskVal !== undefined) {
            const curM = state.materialMask[idx] ?? 0;
            if (maskMode === 'max') {
              state.materialMask[idx] = Math.max(curM, maskVal * w);
            } else if (maskMode === 'add') {
              state.materialMask[idx] = Math.max(0, Math.min(1, curM + maskVal * w));
            } else {
              state.materialMask[idx] = curM + (maskVal - curM) * w;
            }
          }
        }
      }

      if (anyHit) {
        dirtyTiles.push({ state, minIx, maxIx, minIy, maxIy });
      }
    }

    // Pass 2: Recompute positions, cross-border normals, skirts, and patch GPU buffers
    const pC: [number, number, number] = [0, 0, 0];
    const pL: [number, number, number] = [0, 0, 0];
    const pR: [number, number, number] = [0, 0, 0];
    const pD: [number, number, number] = [0, 0, 0];
    const pU: [number, number, number] = [0, 0, 0];

    const projOpts = {
      mode: this.mode,
      planetRadius: this.planetRadius,
      planetCenter: this.planetCenter,
      originGeo: this.originGeo,
      metersPerDegree: this.metersPerDegree,
      elevationScale: this.elevationScale,
    };

    for (const item of dirtyTiles) {
      const { state } = item;
      const seg = state.segments;
      const edgeLen = seg + 1;
      const b = state.bounds;
      const lonSpan = b.maxLon - b.minLon;
      const latSpan = b.maxLat - b.minLat;
      const epsLon = Math.max(1e-6, Math.abs(lonSpan) / (seg * 2));
      const epsLat = Math.max(1e-6, Math.abs(latSpan) / (seg * 2));
      const fallback = state.input.heightSampler ?? this.defaultHeightSampler;

      const nMinIx = Math.max(0, item.minIx - 1);
      const nMaxIx = Math.min(seg, item.maxIx + 1);
      const nMinIy = Math.max(0, item.minIy - 1);
      const nMaxIy = Math.min(seg, item.maxIy + 1);

      const verts = state.meshData.vertices;
      for (let iy = nMinIy; iy <= nMaxIy; iy++) {
        const lat = b.maxLat - (iy / seg) * latSpan;
        for (let ix = nMinIx; ix <= nMaxIx; ix++) {
          const lon = b.minLon + (ix / seg) * lonSpan;
          const gIdx = iy * edgeLen + ix;
          const h = state.elevations[gIdx] ?? 0;
          geoToWorldPoint(lon, lat, h, projOpts, pC);

          const hL = ix > 0 ? (state.elevations[iy * edgeLen + ix - 1] ?? h) : this.sampleElevationAtGeo(lon - epsLon, lat, fallback);
          const hR = ix < seg ? (state.elevations[iy * edgeLen + ix + 1] ?? h) : this.sampleElevationAtGeo(lon + epsLon, lat, fallback);
          const hU = iy > 0 ? (state.elevations[(iy - 1) * edgeLen + ix] ?? h) : this.sampleElevationAtGeo(lon, lat + epsLat, fallback);
          const hD = iy < seg ? (state.elevations[(iy + 1) * edgeLen + ix] ?? h) : this.sampleElevationAtGeo(lon, lat - epsLat, fallback);

          geoToWorldPoint(lon - epsLon, lat, hL, projOpts, pL);
          geoToWorldPoint(lon + epsLon, lat, hR, projOpts, pR);
          geoToWorldPoint(lon, lat - epsLat, hD, projOpts, pD);
          geoToWorldPoint(lon, lat + epsLat, hU, projOpts, pU);

          const tux = pR[0] - pL[0], tuy = pR[1] - pL[1], tuz = pR[2] - pL[2];
          const tvx = pU[0] - pD[0], tvy = pU[1] - pD[1], tvz = pU[2] - pD[2];
          let nx = tuy * tvz - tuz * tvy;
          let ny = tuz * tvx - tux * tvz;
          let nz = tux * tvy - tuy * tvx;
          const nLen = Math.hypot(nx, ny, nz);
          if (nLen > 1e-5) {
            nx /= nLen; ny /= nLen; nz /= nLen;
          } else if (this.mode === 'spherical') {
            const rx = pC[0] - this.planetCenter[0];
            const ry = pC[1] - this.planetCenter[1];
            const rz = pC[2] - this.planetCenter[2];
            const rLen = Math.hypot(rx, ry, rz) || 1;
            nx = rx / rLen; ny = ry / rLen; nz = rz / rLen;
          } else {
            nx = 0; ny = 1; nz = 0;
          }

          const o = gIdx * 8;
          verts[o] = pC[0];
          verts[o + 1] = pC[1];
          verts[o + 2] = pC[2];
          verts[o + 3] = nx;
          verts[o + 4] = ny;
          verts[o + 5] = nz;
          patchedVertices++;
        }
      }

      // Patch perimeter skirt vertices if skirtDepth > 0 and dirty box touches an edge
      const drop = state.input.skirtDepth ?? this.skirtDepth;
      const gridCount = edgeLen * edgeLen;
      if (drop > 0 && verts.length / 8 >= gridCount + edgeLen * 4) {
        const updateSkirtEdge = (skirtStart: number, getGridIndex: (k: number) => number) => {
          for (let k = 0; k < edgeLen; k++) {
            const gi = getGridIndex(k);
            const go = gi * 8;
            const so = (skirtStart + k) * 8;
            const px = verts[go]!, py = verts[go + 1]!, pz = verts[go + 2]!;
            let sx = px, sy = py - drop, sz = pz;
            if (this.mode === 'spherical') {
              const rx = px - this.planetCenter[0];
              const ry = py - this.planetCenter[1];
              const rz = pz - this.planetCenter[2];
              const rLen = Math.hypot(rx, ry, rz) || 1;
              sx = px - (rx / rLen) * drop;
              sy = py - (ry / rLen) * drop;
              sz = pz - (rz / rLen) * drop;
            }
            verts[so] = sx; verts[so + 1] = sy; verts[so + 2] = sz;
            verts[so + 3] = verts[go + 3]!; verts[so + 4] = verts[go + 4]!; verts[so + 5] = verts[go + 5]!;
          }
        };
        if (nMinIy === 0) updateSkirtEdge(gridCount, (k) => k);
        if (nMaxIy === seg) updateSkirtEdge(gridCount + edgeLen, (k) => seg * edgeLen + k);
        if (nMinIx === 0) updateSkirtEdge(gridCount + edgeLen * 2, (k) => k * edgeLen);
        if (nMaxIx === seg) updateSkirtEdge(gridCount + edgeLen * 3, (k) => k * edgeLen + seg);
      }

      // Upload patched vertices in-place to the existing GPU vertex buffer
      const startRow = nMinIy * edgeLen;
      const rowCount = (nMaxIy - nMinIy + 1) * edgeLen;
      this.engine.updateMeshVertices(state.meshId, verts, startRow, rowCount);
      if (drop > 0 && (nMinIy === 0 || nMaxIy === seg || nMinIx === 0 || nMaxIx === seg)) {
        this.engine.updateMeshVertices(state.meshId, verts, gridCount, edgeLen * 4);
      }

      const { min, max } = computeMeshBounds(verts);
      state.worldMin = min;
      state.worldMax = max;
      state.revision += 1;
      updatedTiles.push(state.input.id);
    }

    return { updatedTiles, patchedVertices };
  }

  /**
   * Picks the closest tile under the given canvas pixel coordinates using each tile's true world bounds.
   */
  pick(x: number, y: number): TilePickHit | null {
    if (this.disposed || this.tiles.size === 0) return null;
    const canvas = this.engine.gpu.canvas;
    const w = Math.max(1, canvas.clientWidth || canvas.width);
    const h = Math.max(1, canvas.clientHeight || canvas.height);
    const nx = (x / w) * 2 - 1;
    const ny = 1 - (y / h) * 2;
    const vp = this.engine.camera.updateMatrices(Math.max(1, canvas.width) / Math.max(1, canvas.height));
    const invVp = new Float32Array(16);
    if (!mat4.invert(invVp, vp)) return null;

    const unproject = (ndcZ: number): [number, number, number] => {
      const cx = (invVp[0] ?? 0) * nx + (invVp[4] ?? 0) * ny + (invVp[8] ?? 0) * ndcZ + (invVp[12] ?? 0);
      const cy = (invVp[1] ?? 0) * nx + (invVp[5] ?? 0) * ny + (invVp[9] ?? 0) * ndcZ + (invVp[13] ?? 0);
      const cz = (invVp[2] ?? 0) * nx + (invVp[6] ?? 0) * ny + (invVp[10] ?? 0) * ndcZ + (invVp[14] ?? 0);
      const cw = (invVp[3] ?? 0) * nx + (invVp[7] ?? 0) * ny + (invVp[11] ?? 0) * ndcZ + (invVp[15] ?? 1);
      const invW = Math.abs(cw) > 1e-6 ? 1 / cw : 1;
      return [cx * invW, cy * invW, cz * invW];
    };

    const nearPt = unproject(0);
    const farPt = unproject(1);
    let dx = farPt[0] - nearPt[0];
    let dy = farPt[1] - nearPt[1];
    let dz = farPt[2] - nearPt[2];
    const len = Math.hypot(dx, dy, dz) || 1;
    dx /= len; dy /= len; dz /= len;

    let bestHit: TilePickHit | null = null;
    for (const state of this.tiles.values()) {
      const t = intersectRayAabb(nearPt, [dx, dy, dz], state.worldMin, state.worldMax);
      if (t !== null && (!bestHit || t < bestHit.distance)) {
        bestHit = {
          id: state.input.id,
          distance: t,
          point: [nearPt[0] + dx * t, nearPt[1] + dy * t, nearPt[2] + dz * t],
        };
      }
    }
    return bestHit;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.unsubTick?.();
    this.unsubTick = null;
    this.clearTiles();
  }

  private toRecord(state: InternalTileState): TileRecord {
    return {
      id: state.input.id,
      revision: state.revision,
      bounds: { ...state.bounds },
      centerWorld: [state.centerWorld[0], state.centerWorld[1], state.centerWorld[2]],
      segments: state.segments,
      meshId: state.meshId,
      entity: state.entity,
      vertexCount: state.vertexCount,
      triangleCount: state.triangleCount,
    };
  }
}

function computeMeshBounds(vertices: Float32Array): {
  min: [number, number, number];
  max: [number, number, number];
} {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < vertices.length; i += 8) {
    const x = vertices[i] ?? 0;
    const y = vertices[i + 1] ?? 0;
    const z = vertices[i + 2] ?? 0;
    if (x < min[0]) min[0] = x;
    if (y < min[1]) min[1] = y;
    if (z < min[2]) min[2] = z;
    if (x > max[0]) max[0] = x;
    if (y > max[1]) max[1] = y;
    if (z > max[2]) max[2] = z;
  }
  return { min, max };
}

function intersectRayAabb(
  origin: readonly [number, number, number],
  dir: readonly [number, number, number],
  min: readonly [number, number, number],
  max: readonly [number, number, number],
): number | null {
  let tMin = 0;
  let tMax = Infinity;
  for (let i = 0; i < 3; i++) {
    const o = origin[i] ?? 0;
    const d = dir[i] ?? 0;
    const bMin = min[i] ?? 0;
    const bMax = max[i] ?? 0;
    if (Math.abs(d) < 1e-8) {
      if (o < bMin || o > bMax) return null;
    } else {
      let t1 = (bMin - o) / d;
      let t2 = (bMax - o) / d;
      if (t1 > t2) {
        const tmp = t1;
        t1 = t2;
        t2 = tmp;
      }
      if (t1 > tMin) tMin = t1;
      if (t2 < tMax) tMax = t2;
      if (tMin > tMax) return null;
    }
  }
  return tMin >= 0 ? tMin : tMax >= 0 ? tMax : null;
}

export function TileMeshPlugin(options?: TileMeshPluginOptions): EnginePlugin;
export function TileMeshPlugin(engine: Engine, options?: TileMeshPluginOptions): TileMeshManager;
export function TileMeshPlugin(
  engineOrOptions?: Engine | TileMeshPluginOptions,
  maybeOptions?: TileMeshPluginOptions,
): EnginePlugin | TileMeshManager {
  if (engineOrOptions && typeof (engineOrOptions as Engine).onTick === 'function') {
    const eng = engineOrOptions as Engine;
    const manager = new TileMeshManager(eng, maybeOptions);
    eng.use(() => () => manager.dispose());
    return manager;
  }
  const opts = (engineOrOptions as TileMeshPluginOptions | undefined) ?? {};
  return (engine: Engine) => {
    const manager = new TileMeshManager(engine, opts);
    return () => manager.dispose();
  };
}
