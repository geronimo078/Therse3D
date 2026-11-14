import { describe, expect, it } from 'vitest';
import { Inspector, pickFromScreen, projectWorldPoint, raycastStore } from './Inspector';
import { DirtyFlag, WorldStore } from '../core/WorldStore';
import { mat4 } from '../core/Math';

function makeTestViewProj(): Float32Array {
  const view = new Float32Array(16);
  const proj = new Float32Array(16);
  const vp = new Float32Array(16);
  mat4.perspective(proj, (45 * Math.PI) / 180, 16 / 9, 0.1, 100);
  mat4.lookAt(view, 0, 2, 6, 0, 0, 0);
  mat4.multiply(vp, proj, view);
  return vp;
}

describe('Inspector (Step 5)', () => {
  it('detects visible entities, screenRegion, screenCoverage, and off-screen warnings', () => {
    const store = new WorldStore(8);
    const vp = makeTestViewProj();

    const centerId = store.allocate();
    store.positions.set([0, 0, 0], centerId * 3);

    const lampId = store.allocate();
    store.positions.set([50, 0, 0], lampId * 3);

    const behindId = store.allocate();
    store.positions.set([0, 2, 20], behindId * 3);

    store.updateMatrices();

    const inspector = new Inspector(store);
    const names: Record<number, string> = {
      [centerId]: 'table',
      [lampId]: 'lamp',
      [behindId]: 'wall',
    };
    const desc = inspector.describe({
      viewProjMatrix: vp,
      resolveName: (id) => names[id],
    });

    expect(desc.entityCount).toBe(3);

    const table = desc.entities.find((e) => e.id === 'table')!;
    expect(table.visible).toBe(true);
    expect(table.screenRegion).toBe('center');
    expect(table.screenCoverage).toBeGreaterThan(0);

    const lamp = desc.entities.find((e) => e.id === 'lamp')!;
    expect(lamp.visible).toBe(false);
    expect(lamp.screenRegion).toBe('offscreen-right');

    const wall = desc.entities.find((e) => e.id === 'wall')!;
    expect(wall.visible).toBe(false);
    expect(wall.screenRegion).toBe('behind-camera');

    expect(desc.warnings.some((w) => w.includes("'lamp'") && w.includes('outside camera view'))).toBe(true);
    expect(desc.warnings.some((w) => w.includes("'wall'") && w.includes('outside camera view'))).toBe(true);
  });

  it('detects on, near, and collidingWith relations with intersection warnings', () => {
    const store = new WorldStore(8);
    const vp = makeTestViewProj();

    const tableId = store.allocate();
    store.positions.set([0, 0.5, 0], tableId * 3);

    const mugId = store.allocate();
    store.positions.set([0, 1.5, 0], mugId * 3); // minY = 1.0 == table maxY = 1.0

    const chairId = store.allocate();
    store.positions.set([0.2, 0.5, 0], chairId * 3); // overlaps table

    store.updateMatrices();

    const inspector = new Inspector(store);
    const names: Record<number, string> = {
      [tableId]: 'table',
      [mugId]: 'mug',
      [chairId]: 'chair',
    };
    const desc = inspector.describe({
      viewProjMatrix: vp,
      resolveName: (id) => names[id],
    });

    const mug = desc.entities.find((e) => e.id === 'mug')!;
    expect(mug.relations.on).toContain('table');

    const chair = desc.entities.find((e) => e.id === 'chair')!;
    expect(chair.relations.collidingWith).toContain('table');
    expect(desc.warnings.some((w) => w.includes("'table'") && w.includes("intersects with 'chair'"))).toBe(true);
  });

  it('supports raycastStore and pickFromScreen for 3D selection', () => {
    const store = new WorldStore(8);
    const vp = makeTestViewProj();

    const boxId = store.allocate();
    store.positions.set([0, 0, 0], boxId * 3);
    store.updateMatrices();

    const names: Record<number, string> = { [boxId]: 'target-box' };
    const hit = raycastStore(store, [0, 0, 5], [0, 0, -1], (id) => names[id]);
    expect(hit).not.toBeNull();
    expect(hit!.id).toBe('target-box');
    expect(hit!.distance).toBeCloseTo(4.5);

    const miss = raycastStore(store, [10, 10, 5], [0, 0, -1]);
    expect(miss).toBeNull();

    const screenHit = pickFromScreen(store, vp, 0.5, 0.5, (id) => names[id]);
    expect(screenHit).not.toBeNull();
    expect(screenHit!.id).toBe('target-box');
  });

  it('emits actionable suggestedFixes for colliding entities that resolve the overlap', () => {
    const store = new WorldStore(8);
    const vp = makeTestViewProj();

    const aId = store.allocate();
    store.positions.set([0, 0.5, 0], aId * 3);

    const bId = store.allocate();
    store.positions.set([0.3, 0.5, 0], bId * 3);

    store.updateMatrices();

    const inspector = new Inspector(store);
    const names: Record<number, string> = { [aId]: 'crate-a', [bId]: 'crate-b' };
    const desc = inspector.describe({ viewProjMatrix: vp, resolveName: (id) => names[id] });

    expect(desc.suggestedFixes.length).toBe(1);
    const fix = desc.suggestedFixes[0]!;
    expect(fix.kind).toBe('resolve-collision');
    expect(fix.patch.entities[0]!.id).toBe('crate-a');

    // Apply the suggested fix position and verify collision is resolved
    const [nx, ny, nz] = fix.patch.entities[0]!.position;
    store.positions.set([nx, ny, nz], aId * 3);
    store.dirtyFlags[aId] = DirtyFlag.Transform;
    store.updateMatrices();

    const after = inspector.describe({ viewProjMatrix: vp, resolveName: (id) => names[id] });
    expect(after.suggestedFixes.length).toBe(0);
    expect(after.entities.find((e) => e.id === 'crate-a')!.relations.collidingWith).toEqual([]);
  });

  it('projects 3D world points to viewport coordinates with projectWorldPoint', () => {
    const vp = makeTestViewProj();

    const center = projectWorldPoint(vp, [0, 0, 0], 1280, 720);
    expect(center.visible).toBe(true);
    expect(center.nx).toBeCloseTo(0.5, 3);
    expect(center.ny).toBeCloseTo(0.5, 3);
    expect(center.x).toBeCloseTo(640, 1);
    expect(center.y).toBeCloseTo(360, 1);

    const behind = projectWorldPoint(vp, [0, 2, 20], 1280, 720);
    expect(behind.visible).toBe(false);
  });
});

