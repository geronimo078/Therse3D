import { describe, expect, it, vi } from 'vitest';
import {
  TileMeshManager,
  TileMeshPlugin,
  buildTileMeshData,
  geoToWorldPoint,
  tileCoordToGeoBounds,
} from './TileMeshPlugin';
import type { Engine, TickCallback } from '../core/Engine';
import type { PrimitiveDescriptor } from '../entities/Primitives';

function createMockEngine() {
  let nextMeshId = 1;
  let nextEntityId = 1;
  const unloadedMeshIds: number[] = [];
  const registeredMeshes: number[] = [];
  const removedEntities: unknown[] = [];
  const tickCallbacks = new Set<TickCallback>();
  const cleanups: Array<() => void> = [];

  const camera = {
    position: new Float32Array([0, 2, 0]),
  };

  const engine = {
    camera,
    registerMesh: vi.fn((_data, _label) => {
      const id = nextMeshId++;
      registeredMeshes.push(id);
      return id;
    }),
    unloadMesh: vi.fn((meshId: number) => {
      unloadedMeshIds.push(meshId);
    }),
    updateMeshVertices: vi.fn(() => true),
    setEntityMesh: vi.fn(() => true),
    add: vi.fn((desc: PrimitiveDescriptor) => ({
      id: nextEntityId++,
      name: desc.id ?? 'tile',
      color: [1, 1, 1, 1],
      visible: true,
      roughness: 0.5,
      metalness: 0.0,
    })),
    remove: vi.fn((entity: unknown) => {
      removedEntities.push(entity);
      return true;
    }),
    onTick: vi.fn((cb: TickCallback) => {
      tickCallbacks.add(cb);
      return () => tickCallbacks.delete(cb);
    }),
    use: vi.fn((plugin: (e: Engine) => void | (() => void)) => {
      const c = plugin(engine as unknown as Engine);
      if (typeof c === 'function') cleanups.push(c);
      return engine;
    }),
  };

  return {
    engine: engine as unknown as Engine,
    mocks: {
      engine,
      registeredMeshes,
      unloadedMeshIds,
      removedEntities,
      tickCallbacks,
      cleanups,
    },
  };
}

describe('TileMeshPlugin', () => {
  it('converts slippy-map coords to geo bounds and projects planar & spherical points', () => {
    const bounds = tileCoordToGeoBounds({ z: 1, x: 1, y: 0 });
    expect(bounds.minLon).toBeCloseTo(0, 5);
    expect(bounds.maxLon).toBeCloseTo(180, 5);
    expect(bounds.minLat).toBeCloseTo(0, 5);
    expect(bounds.maxLat).toBeGreaterThan(80);

    const planar = geoToWorldPoint(2, -1, 0.5, {
      mode: 'planar',
      planetRadius: 2,
      planetCenter: [0, 0, 0],
      originGeo: [0, 0],
      metersPerDegree: 1,
      elevationScale: 1,
    });
    expect(planar).toEqual([2, 0.5, 1]);

    const spherical = geoToWorldPoint(0, 0, 0, {
      mode: 'spherical',
      planetRadius: 4,
      planetCenter: [0, 0, 0],
      originGeo: [0, 0],
      metersPerDegree: 1,
      elevationScale: 1,
    });
    expect(spherical[0]).toBeCloseTo(4, 5);
    expect(spherical[1]).toBeCloseTo(0, 5);
    expect(spherical[2]).toBeCloseTo(0, 5);
  });

  it('produces identical boundary positions & normals across adjacent tiles and adds skirts', () => {
    const sampler = (lon: number, lat: number) => Math.sin(lon * 0.4) * Math.cos(lat * 0.4) * 0.3;
    const common = {
      mode: 'spherical' as const,
      planetRadius: 2,
      planetCenter: [0, 0, 0] as const,
      originGeo: [0, 0] as const,
      metersPerDegree: 1,
      elevationScale: 1,
      skirtDepth: 0.05,
      heightSampler: sampler,
    };

    const westTile = buildTileMeshData({
      ...common,
      bounds: { minLon: -10, maxLon: 0, minLat: 0, maxLat: 10 },
      segments: 8,
    });
    const eastTile = buildTileMeshData({
      ...common,
      bounds: { minLon: 0, maxLon: 10, minLat: 0, maxLat: 10 },
      segments: 8,
    });

    // Compare shared corner (lon=0, lat=10): west iy=0, ix=8 vs east iy=0, ix=0
    const westIdx = 8 * 8;
    const eastIdx = 0;
    for (let k = 0; k < 6; k++) {
      expect(westTile.vertices[westIdx + k]).toBeCloseTo(eastTile.vertices[eastIdx + k]!, 4);
    }

    // Verify skirt vertices & indices are appended: (8+1)^2 + 4*(8+1) = 81 + 36 = 117 verts
    expect(westTile.vertices.length / 8).toBe(117);

    // Verify outward vertex normals & CCW triangle winding on spherical tile
    expect(eastTile.vertices[3]!).toBeGreaterThan(0); // +X outward near lon=0, lat=10
    const i0 = eastTile.indices[0]! * 8;
    const i1 = eastTile.indices[1]! * 8;
    const i2 = eastTile.indices[2]! * 8;
    const abx = eastTile.vertices[i1]! - eastTile.vertices[i0]!;
    const aby = eastTile.vertices[i1 + 1]! - eastTile.vertices[i0 + 1]!;
    const abz = eastTile.vertices[i1 + 2]! - eastTile.vertices[i0 + 2]!;
    const acx = eastTile.vertices[i2]! - eastTile.vertices[i0]!;
    const acy = eastTile.vertices[i2 + 1]! - eastTile.vertices[i0 + 1]!;
    const acz = eastTile.vertices[i2 + 2]! - eastTile.vertices[i0 + 2]!;
    const fnx = aby * acz - abz * acy;
    const fny = abz * acx - abx * acz;
    const fnz = abx * acy - aby * acx;
    const dotOut =
      fnx * eastTile.vertices[i0]! +
      fny * eastTile.vertices[i0 + 1]! +
      fnz * eastTile.vertices[i0 + 2]!;
    expect(dotOut).toBeGreaterThan(0);
  });

  it('manages GPU buffer lifecycle on replace, remove, LOD change, and dispose', () => {
    const { engine, mocks } = createMockEngine();
    const tiles = TileMeshPlugin(engine, {
      mode: 'planar',
      originGeo: [0, 0],
      metersPerDegree: 1,
      lodLevels: [
        { maxDistance: 5, segments: 16 },
        { maxDistance: Infinity, segments: 4 },
      ],
    });

    // Camera starts at [0, 2, 0] -> distance to [0, 0, 0] is 2 <= 5 -> 16 segments
    const first = tiles.setTile({
      id: 't1',
      bounds: { minLon: -1, maxLon: 1, minLat: -1, maxLat: 1 },
    });
    expect(first.segments).toBe(16);
    expect(first.meshId).toBe(1);

    // Replace same tile -> old meshId 1 must be unloaded
    const replaced = tiles.setTile({
      id: 't1',
      bounds: { minLon: -1, maxLon: 1, minLat: -1, maxLat: 1 },
    });
    expect(replaced.meshId).toBe(2);
    expect(mocks.unloadedMeshIds).toContain(1);

    // Move camera far away -> LOD drops to 4 segments and old meshId 2 is unloaded
    const changed = tiles.updateLOD([0, 30, 0]);
    expect(changed).toBe(1);
    expect(tiles.getTile('t1')?.segments).toBe(4);
    expect(mocks.unloadedMeshIds).toContain(2);

    // Dispose cleans up remaining meshId 3 and removes entity
    mocks.cleanups.forEach((fn) => fn());
    expect(mocks.unloadedMeshIds).toContain(3);
    expect(mocks.removedEntities.length).toBe(1);
    expect(tiles.listTiles().length).toBe(0);
  });

  it('ignores stale async loadTile completions after removal', async () => {
    const { engine } = createMockEngine();
    const manager = new TileMeshManager(engine, { autoLod: false });

    let resolveProvider!: (v: { bounds: { minLon: number; maxLon: number; minLat: number; maxLat: number } }) => void;
    const pending = manager.loadTile(
      'async-tile',
      () =>
        new Promise((res) => {
          resolveProvider = res;
        }),
    );

    manager.removeTile('async-tile');
    resolveProvider({ bounds: { minLon: 0, maxLon: 1, minLat: 0, maxLat: 1 } });

    const result = await pending;
    expect(result).toBeNull();
    expect(manager.listTiles().length).toBe(0);
    manager.dispose();
  });

  it('samples exact rendered mesh surface (height, normal, slope, land/water, revision)', () => {
    const { engine } = createMockEngine();
    const tiles = TileMeshPlugin(engine, {
      mode: 'planar',
      originGeo: [0, 0],
      metersPerDegree: 1,
      waterLevel: 0.05,
      autoLod: false,
    });

    // Tile with 2x2 segments; height is 0 at corners and 1.0 at center (0, 0) [vertex d of cell (0,0)].
    // Cell (0,0) splits along diagonal b-c (tx+ty=1), so (-0.25, 0.25) has tx=0.75, ty=0.75 -> wD=0.5 -> elev=0.5,
    // even though heightSampler returns 0 at (-0.25, 0.25) itself!
    tiles.setTile({
      id: 'terrain-1',
      bounds: { minLon: -1, maxLon: 1, minLat: -1, maxLat: 1 },
      segments: 2,
      heightSampler: (lon, lat) => (Math.abs(lon) < 0.01 && Math.abs(lat) < 0.01 ? 1.0 : 0.0),
    });

    const midSample = tiles.sampleSurface(-0.25, 0.25)!;
    expect(midSample).not.toBeNull();
    expect(midSample.tileId).toBe('terrain-1');
    expect(midSample.revision).toBe(1);
    expect(midSample.elevation).toBeCloseTo(0.5, 4);
    expect(midSample.worldPosition[1]).toBeCloseTo(0.5, 4);
    expect(midSample.surfaceType).toBe('land');
    expect(midSample.isWater).toBe(false);
    expect(midSample.slopeDeg).toBeGreaterThan(0);

    // Corner at (-1, 1) has elevation 0 <= waterLevel (0.05) -> water
    const waterSample = tiles.sampleSurface(-1, 1)!;
    expect(waterSample.elevation).toBeCloseTo(0, 4);
    expect(waterSample.surfaceType).toBe('water');
    expect(waterSample.isWater).toBe(true);

    // Outside bounds returns null
    expect(tiles.sampleSurface(5, 5)).toBeNull();

    // Replacing tile increments revision to 2
    const updated = tiles.setTile({
      id: 'terrain-1',
      bounds: { minLon: -1, maxLon: 1, minLat: -1, maxLat: 1 },
      segments: 2,
      heightSampler: () => 0,
    });
    expect(updated.revision).toBe(2);
    expect(tiles.sampleSurface(0, 0)!.revision).toBe(2);
    expect(tiles.sampleSurface(0, 0)!.slopeDeg).toBeCloseTo(0, 3);
  });

  it('deforms terrain locally across tile borders, patches GPU vertices in-place, and updates sampleSurface', () => {
    const { engine, mocks } = createMockEngine();
    const tiles = TileMeshPlugin(engine, {
      mode: 'planar',
      originGeo: [0, 0],
      metersPerDegree: 1,
      waterLevel: -0.05,
      autoLod: false,
    });

    tiles.setTile({
      id: 'west',
      bounds: { minLon: -2, maxLon: 0, minLat: -1, maxLat: 1 },
      segments: 8,
    });
    tiles.setTile({
      id: 'east',
      bounds: { minLon: 0, maxLon: 2, minLat: -1, maxLat: 1 },
      segments: 8,
    });

    const beforeSample = tiles.sampleSurface(0, 0)!;
    expect(beforeSample.elevation).toBeCloseTo(0, 4);
    expect(beforeSample.materialMask).toBe(0);

    // Deform right on the shared border (lon=0, lat=0)
    const deformRes = tiles.deformRegion({
      lon: 0,
      lat: 0,
      radiusDeg: 0.75,
      deltaElevation: 0.6,
      materialMaskValue: 0.85,
      mode: 'add',
    });

    expect(deformRes.updatedTiles.sort()).toEqual(['east', 'west']);
    expect(deformRes.patchedVertices).toBeGreaterThan(0);
    expect(mocks.engine.updateMeshVertices).toHaveBeenCalled();

    // Revision bumped on both tiles without destroying/recreating meshIds
    expect(tiles.getTile('west')!.revision).toBe(2);
    expect(tiles.getTile('east')!.revision).toBe(2);

    // Querying either side of the border gives updated elevation, materialMask, and matching height
    const westEdge = tiles.sampleSurface(-0.001, 0)!;
    const eastEdge = tiles.sampleSurface(0.001, 0)!;
    expect(westEdge.elevation).toBeCloseTo(0.6, 2);
    expect(eastEdge.elevation).toBeCloseTo(0.6, 2);
    expect(westEdge.materialMask).toBeCloseTo(0.85, 2);
    expect(eastEdge.materialMask).toBeCloseTo(0.85, 2);
  });
});
