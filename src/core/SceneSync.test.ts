import { describe, expect, it } from 'vitest';
import { SceneSync } from './SceneSync';
import { WorldStore } from './WorldStore';

describe('SceneSync (Step 4)', () => {
  it('diffs added, updated, and removed entities in WorldStore without leaks', () => {
    const store = new WorldStore(16);
    const sync = new SceneSync(store);

    const r1 = sync.apply({
      entities: [
        { id: 'table', type: 'box', size: [2.4, 0.08, 1.2], position: [0, 0.75, 0], material: 'oak-wood' },
        { id: 'orb', type: 'sphere', radius: 0.2, on: 'table', align: 'front-right', color: '#ff3366' },
      ],
    });
    expect(r1).toEqual({ applied: 2, added: 2, updated: 0, removed: 0, warnings: [] });
    expect(store.count).toBe(2);

    const orbId = sync.getEntityId('orb')!;
    // Table maxY = 0.75 + 0.04 = 0.79; orb radius 0.2 => center Y = 0.79 + 0.2 = 0.99
    expect(store.positions[orbId * 3 + 1]).toBeCloseTo(0.99);

    const r2 = sync.apply({
      entities: [
        { id: 'table', type: 'box', size: [2.4, 0.08, 1.2], position: [0, 0.75, 0], material: 'oak-wood' },
      ],
    });
    expect(r2).toEqual({ applied: 1, added: 0, updated: 1, removed: 1, warnings: [] });
    expect(store.count).toBe(1);
    expect(sync.getEntityId('orb')).toBeUndefined();
  });

  it('smoothly tweens position and color changes over transition duration', () => {
    const store = new WorldStore(8);
    const sync = new SceneSync(store);

    sync.apply({
      entities: [{ id: 'cube', type: 'box', position: [0, 0, 0], color: [0, 0, 0, 1] }],
    });

    const id = sync.getEntityId('cube')!;
    sync.apply({
      transition: 0.5,
      entities: [{ id: 'cube', type: 'box', position: [10, 0, 0], color: [1, 1, 1, 1] }],
    });

    // At t = 0, still at start position
    expect(store.positions[id * 3]).toBeCloseTo(0);

    // At t = 0.25 (halfway -> smoothstep(0.5) = 0.5)
    sync.updateTweens(0.25);
    expect(store.positions[id * 3]).toBeCloseTo(5);
    expect(store.colors[id * 4]).toBeCloseTo(0.5);

    // At t = 0.5 (complete)
    sync.updateTweens(0.25);
    expect(store.positions[id * 3]).toBeCloseTo(10);
    expect(store.colors[id * 4]).toBeCloseTo(1);
  });

  it('auto-heals NaN, invalid colors, unknown types/materials/targets, and duplicates with warnings', () => {
    const store = new WorldStore(8);
    const sync = new SceneSync(store);

    const report = sync.apply({
      entities: [
        {
          id: 'broken',
          type: 'unknown-shape',
          position: [Number.NaN, 1, Number.POSITIVE_INFINITY],
          color: 'not-a-color',
          material: 'unobtanium',
          on: 'missing-parent',
        },
        {
          id: 'broken',
          type: 'box',
        },
      ],
    });

    expect(report.added).toBe(1);
    expect(report.warnings.length).toBeGreaterThanOrEqual(5);
    const id = sync.getEntityId('broken')!;
    expect(Number.isFinite(store.positions[id * 3])).toBe(true);
    expect(Number.isFinite(store.positions[id * 3 + 2])).toBe(true);
  });

  it('supports partial patch(), undo(), parent hierarchy, and declarative lights', () => {
    const store = new WorldStore(16);
    const sync = new SceneSync(store);

    sync.apply({
      transition: 0,
      entities: [
        { id: 'rig', type: 'group', position: [10, 0, 0] },
        { id: 'lamp', type: 'light', kind: 'point', parent: 'rig', position: [0, 2, 0], color: '#ffaa44', intensity: 3.5 },
        { id: 'box', type: 'box', position: [0, 0.5, 0] },
      ],
    });

    const lampId = sync.getEntityId('lamp')!;
    expect(store.modelMatrices[lampId * 16 + 12]).toBeCloseTo(10);
    expect(store.modelMatrices[lampId * 16 + 13]).toBeCloseTo(2);
    expect(store.lightParams[lampId * 4 + 1]).toBeCloseTo(3.5);

    // Partial patch: update rig position & remove box without resending lamp
    const patchRes = sync.patch({
      transition: 0,
      upsert: [{ id: 'rig', type: 'group', position: [15, 0, 0] }],
      remove: ['box'],
    });
    expect(patchRes.removed).toBe(1);
    expect(sync.getEntityId('box')).toBeUndefined();
    expect(store.modelMatrices[lampId * 16 + 12]).toBeCloseTo(15);

    // Undo reverts to state before patch
    expect(sync.canUndo).toBe(true);
    const undoRes = sync.undo();
    expect(undoRes.warnings).toEqual([]);
    expect(sync.getEntityId('box')).toBeDefined();
    expect(store.modelMatrices[sync.getEntityId('lamp')! * 16 + 12]).toBeCloseTo(10);
  });
});

