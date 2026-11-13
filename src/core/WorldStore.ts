import { aabb, mat4, quat } from './Math';

/**
 * ECS Data-Layer auf Basis flacher TypedArrays + Dirty-Flags.
 */
export const MAX_ENTITIES_DEFAULT = 10_000;

export const enum DirtyFlag {
  None = 0,
  Transform = 1 << 0,
  Material = 1 << 1,
  Geometry = 1 << 2,
  Bounds = 1 << 3,
}

export const enum GeometryTypeId {
  Box = 0,
  Sphere = 1,
  Cylinder = 2,
  Plane = 3,
  Line = 4,
  Model = 5,
}

export class WorldStore {
  capacity: number;
  count = 0;
  highWaterMark = 0;
  generation = 0;

  positions: Float32Array;
  rotations: Float32Array;
  scales: Float32Array;
  modelMatrices: Float32Array;
  colors: Float32Array;
  pbrParams: Float32Array;
  boundsMin: Float32Array;
  boundsMax: Float32Array;
  dirtyFlags: Uint8Array;
  visible: Uint8Array;
  alive: Uint8Array;
  geometryTypes: Uint8Array;

  private readonly freeList: number[] = [];

  constructor(capacity = MAX_ENTITIES_DEFAULT) {
    this.capacity = Math.max(1, capacity);
    this.positions = new Float32Array(this.capacity * 3);
    this.rotations = new Float32Array(this.capacity * 4);
    this.scales = new Float32Array(this.capacity * 3);
    this.modelMatrices = new Float32Array(this.capacity * 16);
    this.colors = new Float32Array(this.capacity * 4);
    this.pbrParams = new Float32Array(this.capacity * 4);
    this.boundsMin = new Float32Array(this.capacity * 3);
    this.boundsMax = new Float32Array(this.capacity * 3);
    this.dirtyFlags = new Uint8Array(this.capacity);
    this.visible = new Uint8Array(this.capacity);
    this.alive = new Uint8Array(this.capacity);
    this.geometryTypes = new Uint8Array(this.capacity);
  }

  allocate(): number {
    let id: number;
    if (this.freeList.length > 0) {
      id = this.freeList.pop()!;
    } else {
      if (this.highWaterMark >= this.capacity) {
        this.grow();
      }
      id = this.highWaterMark++;
    }

    this.count++;
    this.alive[id] = 1;
    this.visible[id] = 1;

    const pIdx = id * 3;
    this.positions[pIdx] = 0;
    this.positions[pIdx + 1] = 0;
    this.positions[pIdx + 2] = 0;

    quat.identity(this.rotations, id * 4);

    this.scales[pIdx] = 1;
    this.scales[pIdx + 1] = 1;
    this.scales[pIdx + 2] = 1;

    const cIdx = id * 4;
    this.colors[cIdx] = 1;
    this.colors[cIdx + 1] = 1;
    this.colors[cIdx + 2] = 1;
    this.colors[cIdx + 3] = 1;

    this.pbrParams[cIdx] = 0;     // metalness
    this.pbrParams[cIdx + 1] = 0.5; // roughness
    this.pbrParams[cIdx + 2] = 0;   // emissive
    this.pbrParams[cIdx + 3] = 0;   // materialId

    mat4.identity(this.modelMatrices, id * 16);
    aabb.fromCenterHalfExtents(this.boundsMin, this.boundsMax, pIdx, 0, 0, 0, 0.5, 0.5, 0.5);

    this.dirtyFlags[id] =
      DirtyFlag.Transform | DirtyFlag.Material | DirtyFlag.Geometry | DirtyFlag.Bounds;
    return id;
  }

  free(id: number): void {
    if (id < 0 || id >= this.highWaterMark || this.alive[id] === 0) return;
    this.alive[id] = 0;
    this.visible[id] = 0;
    this.dirtyFlags[id] = DirtyFlag.None;
    this.count--;
    this.freeList.push(id);
  }

  release(id: number): void {
    this.free(id);
  }

  markDirty(id: number, flags: DirtyFlag): void {
    if (id < 0 || id >= this.highWaterMark || this.alive[id] === 0) return;
    this.dirtyFlags[id] = (this.dirtyFlags[id] ?? 0) | flags;
  }

  /**
   * Zero-Allocation Matrix- & Bounds-Update für alle als dirty markierten Entities.
   */
  updateMatrices(): number {
    let updated = 0;
    const n = this.highWaterMark;
    const dirty = this.dirtyFlags;
    const alive = this.alive;
    const pos = this.positions;
    const rot = this.rotations;
    const scl = this.scales;
    const mats = this.modelMatrices;
    const bMin = this.boundsMin;
    const bMax = this.boundsMax;

    for (let id = 0; id < n; id++) {
      if (alive[id] === 0) continue;
      const flags = dirty[id] ?? 0;
      if ((flags & DirtyFlag.Transform) === 0) continue;

      const pIdx = id * 3;
      const rIdx = id * 4;
      const px = pos[pIdx] ?? 0;
      const py = pos[pIdx + 1] ?? 0;
      const pz = pos[pIdx + 2] ?? 0;
      const sx = scl[pIdx] ?? 1;
      const sy = scl[pIdx + 1] ?? 1;
      const sz = scl[pIdx + 2] ?? 1;

      mat4.fromTranslationRotationScale(
        mats,
        id * 16,
        px,
        py,
        pz,
        rot[rIdx] ?? 0,
        rot[rIdx + 1] ?? 0,
        rot[rIdx + 2] ?? 0,
        rot[rIdx + 3] ?? 1,
        sx,
        sy,
        sz,
      );

      const gType = this.geometryTypes[id] ?? 0;
      const halfY = gType === GeometryTypeId.Plane ? 0 : Math.abs(sy) * 0.5;
      aabb.fromCenterHalfExtents(
        bMin,
        bMax,
        pIdx,
        px,
        py,
        pz,
        Math.abs(sx) * 0.5,
        halfY,
        Math.abs(sz) * 0.5,
      );

      dirty[id] = flags & ~DirtyFlag.Transform;
      updated++;
    }
    return updated;
  }

  private grow(): void {
    const nextCap = this.capacity * 2;
    this.positions = growFloat32(this.positions, nextCap * 3);
    this.rotations = growFloat32(this.rotations, nextCap * 4);
    this.scales = growFloat32(this.scales, nextCap * 3);
    this.modelMatrices = growFloat32(this.modelMatrices, nextCap * 16);
    this.colors = growFloat32(this.colors, nextCap * 4);
    this.pbrParams = growFloat32(this.pbrParams, nextCap * 4);
    this.boundsMin = growFloat32(this.boundsMin, nextCap * 3);
    this.boundsMax = growFloat32(this.boundsMax, nextCap * 3);
    this.dirtyFlags = growUint8(this.dirtyFlags, nextCap);
    this.visible = growUint8(this.visible, nextCap);
    this.alive = growUint8(this.alive, nextCap);
    this.geometryTypes = growUint8(this.geometryTypes, nextCap);
    this.capacity = nextCap;
    this.generation++;
  }
}

function growFloat32(src: Float32Array, nextLength: number): Float32Array {
  const dst = new Float32Array(nextLength);
  dst.set(src);
  return dst;
}

function growUint8(src: Uint8Array, nextLength: number): Uint8Array {
  const dst = new Uint8Array(nextLength);
  dst.set(src);
  return dst;
}
