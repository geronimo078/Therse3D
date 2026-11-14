import { describe, expect, it } from 'vitest';
import {
  applyLineEndpoints,
  createBoxGeometry,
  createConeGeometry,
  createCylinderGeometry,
  createLineGeometry,
  createPlaneGeometry,
  createSphereGeometry,
  createTorusGeometry,
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

    const cone = createConeGeometry(12);
    expect(cone.vertices.length).toBeGreaterThan(0);
    expect(cone.indices.length).toBe(12 * 6);

    const torus = createTorusGeometry(16, 8);
    expect(torus.vertices.length).toBe(17 * 9 * 8);
    expect(torus.indices.length).toBe(16 * 8 * 6);
  });

  it('uses outward-facing CCW triangle winding for Box, Sphere, Cylinder, Plane, Cone, and Torus', () => {
    const meshes = [
      createBoxGeometry(),
      createSphereGeometry(12, 8),
      createCylinderGeometry(12),
      createPlaneGeometry(),
      createConeGeometry(12),
      createTorusGeometry(16, 8),
    ];

    for (const mesh of meshes) {
      for (let i = 0; i < mesh.indices.length; i += 3) {
        const i0 = mesh.indices[i]! * 8;
        const i1 = mesh.indices[i + 1]! * 8;
        const i2 = mesh.indices[i + 2]! * 8;

        const ax = mesh.vertices[i1]! - mesh.vertices[i0]!;
        const ay = mesh.vertices[i1 + 1]! - mesh.vertices[i0 + 1]!;
        const az = mesh.vertices[i1 + 2]! - mesh.vertices[i0 + 2]!;
        const bx = mesh.vertices[i2]! - mesh.vertices[i0]!;
        const by = mesh.vertices[i2 + 1]! - mesh.vertices[i0 + 1]!;
        const bz = mesh.vertices[i2 + 2]! - mesh.vertices[i0 + 2]!;

        const cx = ay * bz - az * by;
        const cy = az * bx - ax * bz;
        const cz = ax * by - ay * bx;

        const nx = mesh.vertices[i0 + 3]! + mesh.vertices[i1 + 3]! + mesh.vertices[i2 + 3]!;
        const ny = mesh.vertices[i0 + 4]! + mesh.vertices[i1 + 4]! + mesh.vertices[i2 + 4]!;
        const nz = mesh.vertices[i0 + 5]! + mesh.vertices[i1 + 5]! + mesh.vertices[i2 + 5]!;

        expect(cx * nx + cy * ny + cz * nz).toBeGreaterThanOrEqual(0);
      }
    }
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

