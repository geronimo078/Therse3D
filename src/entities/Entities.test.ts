import { describe, expect, it } from 'vitest';
import {
  applyLineEndpoints,
  createBoxGeometry,
  createCylinderGeometry,
  createLineGeometry,
  createPlaneGeometry,
  createSphereGeometry,
} from './Primitives';
import { WorldStore } from '../core/WorldStore';
import { parseColor, resolveMaterial } from './Materials';

describe('Procedural Primitives (Schritt 2)', () => {
  it('generates valid vertex and index buffers for Box, Sphere, Cylinder, Plane', () => {
    const box = createBoxGeometry();
    expect(box.vertices.length).toBe(24 * 8);
    expect(box.indices.length).toBe(36);

    const sphere = createSphereGeometry(16, 8);
    expect(sphere.vertices.length).toBe(17 * 9 * 8);
    expect(sphere.indices.length).toBe(16 * 8 * 6);

    const cyl = createCylinderGeometry(12);
    expect(cyl.vertices.length).toBeGreaterThan(0);
    expect(cyl.indices.length).toBe(12 * 12);

    const plane = createPlaneGeometry();
    expect(plane.vertices.length).toBe(4 * 8);
    expect(plane.indices.length).toBe(6);

    const line = createLineGeometry();
    expect(line.vertices.length).toBe(2 * 8);
    expect(line.indices.length).toBe(2);
  });

  it('computes midpoint, length scale, and orientation for Line endpoints', () => {
    const store = new WorldStore(4);
    const id = store.allocate();
    applyLineEndpoints(store, id, [0, 0, 0], [3, 0, 4]);
    expect(store.positions[id * 3]).toBeCloseTo(1.5);
    expect(store.positions[id * 3 + 1]).toBeCloseTo(0);
    expect(store.positions[id * 3 + 2]).toBeCloseTo(2);
    expect(store.scales[id * 3 + 2]).toBeCloseTo(5);
  });
});

describe('Materials & Color Auto-Healing (Schritt 2)', () => {
  it('resolves semantic material presets and warns on unknown names', () => {
    const oak = resolveMaterial('oak-wood');
    expect(oak.metalness).toBe(0);
    expect(oak.roughness).toBeCloseTo(0.72);

    const warnings: string[] = [];
    const fallback = resolveMaterial('unknown-mat', warnings);
    expect(fallback.materialId).toBe(3); // matte-plastic
    expect(warnings.length).toBe(1);
  });

  it('parses hex, CSS names, and arrays with safe fallback warnings', () => {
    const red = parseColor('#ff0000');
    expect(red[0]).toBeCloseTo(1);
    expect(red[1]).toBeCloseTo(0);

    const shortHex = parseColor('#0f0');
    expect(shortHex[1]).toBeCloseTo(1);

    const warnings: string[] = [];
    const invalid = parseColor('not-a-color', warnings);
    expect(invalid).toEqual([1, 1, 1, 1]);
    expect(warnings.length).toBe(1);
  });
});

