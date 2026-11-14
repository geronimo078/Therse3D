import type { Engine } from '../core/Engine';
import { quat } from '../core/Math';
import type { SpatialPlacement } from '../core/SpatialLayout';
import { DirtyFlag, type WorldStore } from '../core/WorldStore';
import type { EntityHandle } from './EntityHandle';
import type { MaterialDefinition, MaterialPresetName } from './Materials';

export type PrimitiveType = 'box' | 'sphere' | 'cylinder' | 'plane' | 'line' | 'model' | 'group' | 'light';
export type LightKind = 'point' | 'directional' | 'spot';

export interface PrimitiveOptions extends SpatialPlacement {
  id?: string;
  parent?: string;
  position?: [number, number, number];
  rotation?: [number, number, number] | [number, number, number, number];
  scale?: [number, number, number] | number;
  size?: [number, number, number];
  radius?: number;
  color?: string | [number, number, number] | [number, number, number, number];
  metalness?: number;
  roughness?: number;
  emissive?: number;
  visible?: boolean;
  material?: MaterialPresetName | Partial<MaterialDefinition>;
  kind?: LightKind;
  intensity?: number;
  range?: number;
  castShadow?: boolean;
  from?: [number, number, number];
  to?: [number, number, number];
  url?: string;
  mesh?: GeometryMeshData;
}

export interface PrimitiveDescriptor extends PrimitiveOptions {
  readonly __descriptor: 'primitive-descriptor';
  readonly type: PrimitiveType;
}

export interface GeometryMeshData {
  /** Interleaved: [px, py, pz, nx, ny, nz, u, v] = 8 floats/vertex */
  vertices: Float32Array;
  indices: Uint16Array | Uint32Array;
}

function makeFactory(type: PrimitiveType) {
  function factory(options?: PrimitiveOptions): PrimitiveDescriptor;
  function factory(engine: Engine, options?: PrimitiveOptions): EntityHandle;
  function factory(
    engineOrOptions?: Engine | PrimitiveOptions,
    maybeOptions?: PrimitiveOptions,
  ): PrimitiveDescriptor | EntityHandle {
    if (engineOrOptions && typeof (engineOrOptions as Engine).add === 'function') {
      return (engineOrOptions as Engine).add({
        __descriptor: 'primitive-descriptor',
        type,
        ...maybeOptions,
      });
    }
    return {
      __descriptor: 'primitive-descriptor',
      type,
      ...(engineOrOptions as PrimitiveOptions | undefined),
    };
  }
  return factory;
}

export const Box = makeFactory('box');
export const Sphere = makeFactory('sphere');
export const Cylinder = makeFactory('cylinder');
export const Plane = makeFactory('plane');
export const Line = makeFactory('line');
export const Model = makeFactory('model');
export const Group = makeFactory('group');
export const Light = makeFactory('light');

/**
 * Prozeduraler Unit-Box-Generator (1x1x1 um Ursprung, 24 Vertices, 36 Indices).
 */
export function createBoxGeometry(): GeometryMeshData {
  const h = 0.5;
  // prettier-ignore
  const vertices = new Float32Array([
    // Front (+Z)
    -h, -h,  h,  0,  0,  1,  0, 0,
     h, -h,  h,  0,  0,  1,  1, 0,
     h,  h,  h,  0,  0,  1,  1, 1,
    -h,  h,  h,  0,  0,  1,  0, 1,
    // Back (-Z)
     h, -h, -h,  0,  0, -1,  0, 0,
    -h, -h, -h,  0,  0, -1,  1, 0,
    -h,  h, -h,  0,  0, -1,  1, 1,
     h,  h, -h,  0,  0, -1,  0, 1,
    // Top (+Y)
    -h,  h,  h,  0,  1,  0,  0, 0,
     h,  h,  h,  0,  1,  0,  1, 0,
     h,  h, -h,  0,  1,  0,  1, 1,
    -h,  h, -h,  0,  1,  0,  0, 1,
    // Bottom (-Y)
    -h, -h, -h,  0, -1,  0,  0, 0,
     h, -h, -h,  0, -1,  0,  1, 0,
     h, -h,  h,  0, -1,  0,  1, 1,
    -h, -h,  h,  0, -1,  0,  0, 1,
    // Right (+X)
     h, -h,  h,  1,  0,  0,  0, 0,
     h, -h, -h,  1,  0,  0,  1, 0,
     h,  h, -h,  1,  0,  0,  1, 1,
     h,  h,  h,  1,  0,  0,  0, 1,
    // Left (-X)
    -h, -h, -h, -1,  0,  0,  0, 0,
    -h, -h,  h, -1,  0,  0,  1, 0,
    -h,  h,  h, -1,  0,  0,  1, 1,
    -h,  h, -h, -1,  0,  0,  0, 1,
  ]);

  // prettier-ignore
  const indices = new Uint16Array([
     0,  1,  2,  0,  2,  3,
     4,  5,  6,  4,  6,  7,
     8,  9, 10,  8, 10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23,
  ]);

  return { vertices, indices };
}

/**
 * Prozeduraler Unit-Sphere-Generator (Durchmesser 1.0, Radius 0.5).
 */
export function createSphereGeometry(segments = 24, rings = 16): GeometryMeshData {
  const radius = 0.5;
  const vertCount = (rings + 1) * (segments + 1);
  const vertices = new Float32Array(vertCount * 8);
  const indices = new Uint16Array(rings * segments * 6);

  let v = 0;
  for (let r = 0; r <= rings; r++) {
    const vCoord = r / rings;
    const phi = vCoord * Math.PI;
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);

    for (let s = 0; s <= segments; s++) {
      const uCoord = s / segments;
      const theta = uCoord * Math.PI * 2;
      const nx = Math.cos(theta) * sinPhi;
      const ny = cosPhi;
      const nz = Math.sin(theta) * sinPhi;

      vertices[v++] = nx * radius;
      vertices[v++] = ny * radius;
      vertices[v++] = nz * radius;
      vertices[v++] = nx;
      vertices[v++] = ny;
      vertices[v++] = nz;
      vertices[v++] = uCoord;
      vertices[v++] = 1 - vCoord;
    }
  }

  let idx = 0;
  const stride = segments + 1;
  for (let r = 0; r < rings; r++) {
    for (let s = 0; s < segments; s++) {
      const a = r * stride + s;
      const b = a + stride;
      indices[idx++] = a;
      indices[idx++] = b;
      indices[idx++] = a + 1;
      indices[idx++] = b;
      indices[idx++] = b + 1;
      indices[idx++] = a + 1;
    }
  }

  return { vertices, indices };
}

/**
 * Prozeduraler Unit-Cylinder-Generator (Höhe 1.0, Radius 0.5).
 */
export function createCylinderGeometry(segments = 24): GeometryMeshData {
  const radius = 0.5;
  const halfH = 0.5;
  const sideVerts = (segments + 1) * 2;
  const capVerts = (segments + 2) * 2;
  const vertices = new Float32Array((sideVerts + capVerts) * 8);
  const indices = new Uint16Array(segments * 6 + segments * 6);

  let v = 0;
  for (let i = 0; i <= segments; i++) {
    const u = i / segments;
    const theta = u * Math.PI * 2;
    const nx = Math.cos(theta);
    const nz = Math.sin(theta);

    // Top ring
    vertices[v++] = nx * radius; vertices[v++] = halfH; vertices[v++] = nz * radius;
    vertices[v++] = nx; vertices[v++] = 0; vertices[v++] = nz;
    vertices[v++] = u; vertices[v++] = 1;

    // Bottom ring
    vertices[v++] = nx * radius; vertices[v++] = -halfH; vertices[v++] = nz * radius;
    vertices[v++] = nx; vertices[v++] = 0; vertices[v++] = nz;
    vertices[v++] = u; vertices[v++] = 0;
  }

  let idx = 0;
  for (let i = 0; i < segments; i++) {
    const a = i * 2;
    const b = a + 1;
    const c = a + 2;
    const d = a + 3;
    indices[idx++] = a; indices[idx++] = b; indices[idx++] = c;
    indices[idx++] = b; indices[idx++] = d; indices[idx++] = c;
  }

  // Top cap
  const topCenterIdx = v / 8;
  vertices[v++] = 0; vertices[v++] = halfH; vertices[v++] = 0;
  vertices[v++] = 0; vertices[v++] = 1; vertices[v++] = 0;
  vertices[v++] = 0.5; vertices[v++] = 0.5;

  for (let i = 0; i <= segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    const cx = Math.cos(theta);
    const cz = Math.sin(theta);
    vertices[v++] = cx * radius; vertices[v++] = halfH; vertices[v++] = cz * radius;
    vertices[v++] = 0; vertices[v++] = 1; vertices[v++] = 0;
    vertices[v++] = 0.5 + cx * 0.5; vertices[v++] = 0.5 + cz * 0.5;
  }
  for (let i = 0; i < segments; i++) {
    indices[idx++] = topCenterIdx;
    indices[idx++] = topCenterIdx + 1 + i + 1;
    indices[idx++] = topCenterIdx + 1 + i;
  }

  // Bottom cap
  const botCenterIdx = v / 8;
  vertices[v++] = 0; vertices[v++] = -halfH; vertices[v++] = 0;
  vertices[v++] = 0; vertices[v++] = -1; vertices[v++] = 0;
  vertices[v++] = 0.5; vertices[v++] = 0.5;

  for (let i = 0; i <= segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    const cx = Math.cos(theta);
    const cz = Math.sin(theta);
    vertices[v++] = cx * radius; vertices[v++] = -halfH; vertices[v++] = cz * radius;
    vertices[v++] = 0; vertices[v++] = -1; vertices[v++] = 0;
    vertices[v++] = 0.5 + cx * 0.5; vertices[v++] = 0.5 + cz * 0.5;
  }
  for (let i = 0; i < segments; i++) {
    indices[idx++] = botCenterIdx;
    indices[idx++] = botCenterIdx + 1 + i;
    indices[idx++] = botCenterIdx + 1 + i + 1;
  }

  return { vertices, indices };
}

/**
 * Prozeduraler Unit-Plane-Generator (1x1 auf XZ-Ebene, Normale +Y).
 */
export function createPlaneGeometry(): GeometryMeshData {
  const h = 0.5;
  // prettier-ignore
  const vertices = new Float32Array([
    -h, 0,  h,  0, 1, 0,  0, 0,
     h, 0,  h,  0, 1, 0,  1, 0,
     h, 0, -h,  0, 1, 0,  1, 1,
    -h, 0, -h,  0, 1, 0,  0, 1,
  ]);
  // prettier-ignore
  const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
  return { vertices, indices };
}

/**
 * Prozeduraler Unit-Line-Generator (Länge 1.0 entlang Z-Achse von -0.5 bis +0.5).
 */
export function createLineGeometry(): GeometryMeshData {
  // prettier-ignore
  const vertices = new Float32Array([
    0, 0, -0.5,  0, 1, 0,  0, 0,
    0, 0,  0.5,  0, 1, 0,  1, 0,
  ]);
  const indices = new Uint16Array([0, 1]);
  return { vertices, indices };
}

/**
 * Berechnet Position, Rotation und Skalierung eines Line-Primitives aus Start- und Endpunkt.
 */
export function applyLineEndpoints(
  store: WorldStore,
  entityId: number,
  from: readonly [number, number, number],
  to: readonly [number, number, number],
): void {
  const fx = Number.isFinite(from[0]) ? from[0] : 0;
  const fy = Number.isFinite(from[1]) ? from[1] : 0;
  const fz = Number.isFinite(from[2]) ? from[2] : 0;
  const tx = Number.isFinite(to[0]) ? to[0] : 0;
  const ty = Number.isFinite(to[1]) ? to[1] : 0;
  const tz = Number.isFinite(to[2]) ? to[2] : 0;

  const pIdx = entityId * 3;
  store.positions[pIdx] = (fx + tx) * 0.5;
  store.positions[pIdx + 1] = (fy + ty) * 0.5;
  store.positions[pIdx + 2] = (fz + tz) * 0.5;

  const len = Math.hypot(tx - fx, ty - fy, tz - fz);
  store.scales[pIdx] = 1;
  store.scales[pIdx + 1] = 1;
  store.scales[pIdx + 2] = Math.max(len, 1e-4);

  quat.fromLookAt(store.rotations, entityId * 4, fx, fy, fz, tx, ty, tz);
  store.markDirty(entityId, DirtyFlag.Transform | DirtyFlag.Bounds);
}
