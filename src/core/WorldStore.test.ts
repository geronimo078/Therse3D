import { describe, expect, it } from 'vitest';
import { WorldStore, DirtyFlag } from './WorldStore';
import { SceneSync } from './SceneSync';
import { EntityHandle } from '../entities/EntityHandle';
import { Box } from '../entities/Primitives';

describe('WorldStore (Schritt 1)', () => {
  it('allocates, recycles via free-list, and updates 4x4 modelMatrices', () => {
    const store = new WorldStore(4);
    const id0 = store.allocate();
    const id1 = store.allocate();

    expect(id0).toBe(0);
    expect(id1).toBe(1);
    expect(store.count).toBe(2);

    store.free(id0);
    expect(store.alive[id0]).toBe(0);
    expect(store.count).toBe(1);

    const recycled = store.allocate();
    expect(recycled).toBe(id0);
    expect(store.count).toBe(2);

    const updated = store.updateMatrices();
    expect(updated).toBe(2);
    expect((store.dirtyFlags[id0] ?? 0) & DirtyFlag.Transform).toBe(0);
  });

  it('grows dynamically when capacity is exceeded', () => {
    const store = new WorldStore(2);
    const a = store.allocate();
    const b = store.allocate();
    expect(store.capacity).toBe(2);

    const c = store.allocate();
    expect(c).toBe(2);
    expect(store.capacity).toBe(4);
    expect(store.generation).toBe(1);
    expect(store.alive[a]).toBe(1);
    expect(store.alive[b]).toBe(1);
    expect(store.alive[c]).toBe(1);
  });
});

describe('EntityHandle (Schritt 1)', () => {
  it('reads/writes position, scale, rotation, color, visible directly in WorldStore and updates modelMatrices', () => {
    const store = new WorldStore(2);
    const id = store.allocate();
    const cube = new EntityHandle(store, id, 'cube');

    cube.position = [1, 2, 3];
    cube.scale = [2, 3, 4];
    cube.color = [0.2, 0.4, 0.6, 1];
    cube.visible = false;

    expect(store.visible[id]).toBe(0);
    expect(store.colors[id * 4 + 1]).toBeCloseTo(0.4);

    store.updateMatrices();
    const mOffset = id * 16;
    expect(store.modelMatrices[mOffset + 0]).toBeCloseTo(2);
    expect(store.modelMatrices[mOffset + 5]).toBeCloseTo(3);
    expect(store.modelMatrices[mOffset + 10]).toBeCloseTo(4);
    expect(store.modelMatrices[mOffset + 12]).toBeCloseTo(1);
    expect(store.modelMatrices[mOffset + 13]).toBeCloseTo(2);
    expect(store.modelMatrices[mOffset + 14]).toBeCloseTo(3);

    // In-place mutation via getter + WorldStore growth
    store.allocate();
    store.allocate(); // triggers grow()
    cube.position[1] = (cube.position[1] ?? 0) + 5;
    store.updateMatrices();
    expect(store.modelMatrices[mOffset + 13]).toBeCloseTo(7);
  });

  it('propagates parent transforms to child entities in WorldStore', () => {
    const store = new WorldStore(8);
    const parentId = store.allocate();
    const childId = store.allocate();
    store.setParent(childId, parentId);

    store.positions.set([10, 0, 0], parentId * 3);
    store.positions.set([2, 3, 0], childId * 3);
    store.markDirty(parentId, DirtyFlag.Transform);
    store.updateMatrices();

    expect(store.modelMatrices[childId * 16 + 12]).toBeCloseTo(12);
    expect(store.modelMatrices[childId * 16 + 13]).toBeCloseTo(3);

    // Move parent again -> child updates automatically
    store.positions.set([20, 1, 0], parentId * 3);
    store.markDirty(parentId, DirtyFlag.Transform);
    store.updateMatrices();
    expect(store.modelMatrices[childId * 16 + 12]).toBeCloseTo(22);
    expect(store.modelMatrices[childId * 16 + 13]).toBeCloseTo(4);
  });

  it('creates PrimitiveDescriptor via Box() without Engine instance', () => {
    const desc = Box({ id: 'table', size: [2.4, 0.08, 1.2], position: [0, 0.75, 0] });
    expect(desc.__descriptor).toBe('primitive-descriptor');
    expect(desc.type).toBe('box');
    expect(desc.id).toBe('table');
  });
});

describe('SceneSync', () => {
  it('diffs added, updated, and removed entities', () => {
    const sync = new SceneSync();
    const first = sync.apply({
      entities: [
        { id: 'a', type: 'box' },
        { id: 'b', type: 'sphere' },
      ],
    });
    expect(first).toEqual({ applied: 2, added: 2, updated: 0, removed: 0, warnings: [] });

    const second = sync.apply({
      entities: [
        { id: 'b', type: 'sphere' },
        { id: 'c', type: 'cylinder' },
      ],
    });
    expect(second).toEqual({ applied: 2, added: 1, updated: 1, removed: 1, warnings: [] });
  });
});
