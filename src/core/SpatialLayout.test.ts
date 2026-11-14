import { describe, expect, it } from 'vitest';
import { SpatialLayout } from './SpatialLayout';
import { GeometryTypeId, WorldStore } from './WorldStore';

describe('SpatialLayout (Schritt 3)', () => {
  it('topologically sorts nested dependencies and detects cycles safely', () => {
    const layout = new SpatialLayout();
    const sorted = layout.topologicalSort([
      { id: 'mug', on: 'table' },
      { id: 'table', on: 'floor' },
      { id: 'floor' },
    ]);
    expect(sorted.map((n) => n.id)).toEqual(['floor', 'table', 'mug']);

    const warnings: string[] = [];
    layout.topologicalSort(
      [
        { id: 'a', on: 'b' },
        { id: 'b', on: 'a' },
      ],
      warnings,
    );
    expect(warnings.length).toBeGreaterThan(0);
  });

  it('places table on floor, laptop and mug on table without floating or intersection', () => {
    const store = new WorldStore(16);
    const layout = new SpatialLayout();

    const floorId = store.allocate();
    store.geometryTypes[floorId] = GeometryTypeId.Plane;
    store.scales.set([10, 1, 10], floorId * 3);

    const tableId = store.allocate();
    store.geometryTypes[tableId] = GeometryTypeId.Box;
    store.scales.set([2.4, 0.8, 1.2], tableId * 3);

    const laptopId = store.allocate();
    store.geometryTypes[laptopId] = GeometryTypeId.Box;
    store.scales.set([0.4, 0.04, 0.3], laptopId * 3);

    const mugId = store.allocate();
    store.geometryTypes[mugId] = GeometryTypeId.Cylinder;
    store.scales.set([0.1, 0.12, 0.1], mugId * 3);

    store.updateMatrices();

    const lookup = (name: string) =>
      ({ floor: floorId, table: tableId, laptop: laptopId, mug: mugId })[name];

    layout.applyPlacement(store, tableId, { on: 'floor', align: 'center' }, lookup);
    // Table height 0.8 on floor (maxY=0) => table center Y = 0.4, maxY = 0.8
    expect(store.positions[tableId * 3 + 1]).toBeCloseTo(0.4);
    expect(store.boundsMax[tableId * 3 + 1]).toBeCloseTo(0.8);

    layout.applyPlacement(store, laptopId, { on: 'table', align: 'center' }, lookup);
    // Laptop height 0.04 on table (maxY=0.8) => laptop minY = 0.8, center Y = 0.82
    expect(store.boundsMin[laptopId * 3 + 1]).toBeCloseTo(0.8);
    expect(store.positions[laptopId * 3 + 1]).toBeCloseTo(0.82);

    layout.applyPlacement(
      store,
      mugId,
      { on: 'table', align: 'front-right', margin: 0.05, offset: [0, 0, -0.02] },
      lookup,
    );
    // Mug minY = 0.8 (flush on table), X = 1.2 - 0.05 - 0.05 = 1.1, Z = 0.6 - 0.05 - 0.05 - 0.02 = 0.48
    expect(store.boundsMin[mugId * 3 + 1]).toBeCloseTo(0.8);
    expect(store.positions[mugId * 3]).toBeCloseTo(1.1);
    expect(store.positions[mugId * 3 + 2]).toBeCloseTo(0.48);
  });

  it('auto-heals unknown "on" target to floor and rotates via lookAt', () => {
    const store = new WorldStore(8);
    const layout = new SpatialLayout();
    const id = store.allocate();
    store.updateMatrices();

    const warnings: string[] = [];
    layout.applyPlacement(
      store,
      id,
      { on: 'nonexistent', lookAt: [10, 0, 0] },
      () => undefined,
      warnings,
    );

    expect(warnings.length).toBe(1);
    expect(store.positions[id * 3 + 1]).toBeCloseTo(0.5);
    // Quaternion rotated around Y toward +X (not identity [0,0,0,1])
    expect(Math.abs(store.rotations[id * 4 + 1] ?? 0)).toBeGreaterThan(0.5);
  });

  it('places entities beside, inside, and resolves overlaps with warnings', () => {
    const store = new WorldStore(16);
    const layout = new SpatialLayout();

    const shelfId = store.allocate();
    store.scales.set([2, 2, 1], shelfId * 3);
    store.positions.set([0, 1, 0], shelfId * 3);

    const chairId = store.allocate();
    store.scales.set([0.6, 0.8, 0.6], chairId * 3);

    const bookId = store.allocate();
    store.scales.set([0.2, 0.3, 0.2], bookId * 3);

    const extraId = store.allocate();
    store.scales.set([0.6, 0.8, 0.6], extraId * 3);

    store.updateMatrices();
    const lookup = (name: string) => ({ shelf: shelfId, chair: chairId, book: bookId, extra: extraId })[name];

    layout.applyPlacement(store, chairId, { beside: 'shelf', side: 'right', margin: 0.2 }, lookup);
    // shelf maxX = 1.0, chair halfX = 0.3, margin = 0.2 => chair x = 1.5, y = 0.4
    expect(store.positions[chairId * 3]).toBeCloseTo(1.5);
    expect(store.positions[chairId * 3 + 1]).toBeCloseTo(0.4);

    layout.applyPlacement(store, bookId, { inside: 'shelf', align: 'center', margin: 0.1 }, lookup);
    // shelf minY = 0, book halfY = 0.15, margin = 0.1 => book y = 0.25
    expect(store.positions[bookId * 3 + 1]).toBeCloseTo(0.25);

    const warnings: string[] = [];
    store.positions.set([1.5, 0.4, 0], extraId * 3);
    store.markDirty(extraId, 1);
    store.updateMatrices();
    layout.applyPlacement(store, extraId, { avoidOverlap: true }, lookup, warnings);
    expect(warnings.length).toBeGreaterThanOrEqual(1);
    expect(Math.abs((store.positions[extraId * 3] ?? 0) - 1.5)).toBeGreaterThan(0.5);
  });

  it('arranges entity groups in ring and grid layouts', () => {
    const store = new WorldStore(16);
    const layout = new SpatialLayout();
    const ids = [store.allocate(), store.allocate(), store.allocate(), store.allocate()];
    store.updateMatrices();
    const lookup = (name: string) => ids[Number(name.slice(1))];

    layout.arrangeGroup(store, { kind: 'ring', entities: ['e0', 'e1', 'e2', 'e3'], radius: 2, lookAtCenter: true }, lookup);
    expect(store.positions[ids[0]! * 3 + 2]).toBeCloseTo(2);
    expect(store.positions[ids[1]! * 3]).toBeCloseTo(2);

    layout.arrangeGroup(store, { kind: 'grid', entities: ['e0', 'e1', 'e2', 'e3'], columns: 2, spacing: 2 }, lookup);
    expect(store.positions[ids[0]! * 3]).toBeCloseTo(-1);
    expect(store.positions[ids[0]! * 3 + 2]).toBeCloseTo(-1);
    expect(store.positions[ids[3]! * 3]).toBeCloseTo(1);
    expect(store.positions[ids[3]! * 3 + 2]).toBeCloseTo(1);
  });
});

