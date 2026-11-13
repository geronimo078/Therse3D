import { quat } from './Math';
import { SpatialLayout, type SpatialPlacement } from './SpatialLayout';
import { DirtyFlag, GeometryTypeId, type WorldStore } from './WorldStore';
import { parseColor, resolveMaterial } from '../entities/Materials';
import type { PrimitiveType } from '../entities/Primitives';

export type CameraAnglePreset = 'front' | 'front-elevated' | 'top' | 'isometric';

export interface SceneSyncCamera {
  target?: string | [number, number, number];
  angle?: CameraAnglePreset;
  distance?: number;
}

export interface SceneSyncNode extends SpatialPlacement {
  id: string;
  type: PrimitiveType | string;
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
  material?: string;
}

export interface SceneSyncState {
  preset?: string;
  transition?: number;
  camera?: SceneSyncCamera;
  entities: SceneSyncNode[];
}

export interface SceneSyncResult {
  applied: number;
  added: number;
  updated: number;
  removed: number;
  warnings: string[];
}

export interface SceneSyncHooks {
  onPreset?: (preset: string, warnings: string[]) => void;
  onCamera?: (camera: SceneSyncCamera, resolveEntityId: (name: string) => number | undefined, warnings: string[]) => void;
  onEntityCreated?: (name: string, entityId: number) => void;
  onEntityRemoved?: (name: string, entityId: number) => void;
}

const VALID_TYPE_MAP: Record<string, GeometryTypeId> = {
  box: GeometryTypeId.Box,
  sphere: GeometryTypeId.Sphere,
  cylinder: GeometryTypeId.Cylinder,
  plane: GeometryTypeId.Plane,
  line: GeometryTypeId.Line,
  model: GeometryTypeId.Model,
};

function sanitizeNum(v: unknown, fallback: number, label: string, warnings: string[]): number {
  if (typeof v === 'number' && Number.isFinite(v)) {
    return v;
  }
  warnings.push(`Invalid number for ${label}; using default ${fallback}.`);
  return fallback;
}

function sanitizeVec3(
  arr: unknown,
  fallback: [number, number, number],
  label: string,
  warnings: string[],
): [number, number, number] {
  if (!Array.isArray(arr) || arr.length < 3) {
    warnings.push(`Invalid vec3 for ${label}; using default [${fallback.join(', ')}].`);
    return [fallback[0], fallback[1], fallback[2]];
  }
  return [
    sanitizeNum(arr[0], fallback[0], `${label}[0]`, warnings),
    sanitizeNum(arr[1], fallback[1], `${label}[1]`, warnings),
    sanitizeNum(arr[2], fallback[2], `${label}[2]`, warnings),
  ];
}

/**
 * Idempotent JSON State Sync with Diffing, Zero-Allocation Auto-Tweening & Auto-Healing.
 */
export class SceneSync {
  private readonly store: WorldStore | undefined;
  private readonly layout: SpatialLayout;
  private readonly nameToEntityId = new Map<string, number>();

  // Pre-allocated tween buffers (indexed by entityId, grown only if store grows)
  private tweenCapacity = 0;
  private tweenActiveCount = 0;
  private tweenDuration: Float32Array = new Float32Array(0);
  private tweenElapsed: Float32Array = new Float32Array(0);
  private tweenStartPos: Float32Array = new Float32Array(0);
  private tweenTargetPos: Float32Array = new Float32Array(0);
  private tweenStartRot: Float32Array = new Float32Array(0);
  private tweenTargetRot: Float32Array = new Float32Array(0);
  private tweenStartColor: Float32Array = new Float32Array(0);
  private tweenTargetColor: Float32Array = new Float32Array(0);

  constructor(store?: WorldStore, layout = new SpatialLayout()) {
    this.store = store;
    this.layout = layout;
    if (store) {
      this.ensureTweenCapacity(store.capacity);
    }
  }

  getEntityId(name: string): number | undefined {
    return this.nameToEntityId.get(name);
  }

  registerExternalEntity(name: string, entityId: number): void {
    this.nameToEntityId.set(name, entityId);
  }

  unregisterExternalEntity(name: string): void {
    this.nameToEntityId.delete(name);
  }

  private ensureTweenCapacity(capacity: number): void {
    if (this.tweenCapacity >= capacity) return;
    this.tweenCapacity = capacity;
    this.tweenDuration = growFloat32(this.tweenDuration, capacity);
    this.tweenElapsed = growFloat32(this.tweenElapsed, capacity);
    this.tweenStartPos = growFloat32(this.tweenStartPos, capacity * 3);
    this.tweenTargetPos = growFloat32(this.tweenTargetPos, capacity * 3);
    this.tweenStartRot = growFloat32(this.tweenStartRot, capacity * 4);
    this.tweenTargetRot = growFloat32(this.tweenTargetRot, capacity * 4);
    this.tweenStartColor = growFloat32(this.tweenStartColor, capacity * 4);
    this.tweenTargetColor = growFloat32(this.tweenTargetColor, capacity * 4);
  }

  apply(state: SceneSyncState, hooks?: SceneSyncHooks): SceneSyncResult {
    const warnings: string[] = [];

    if (!state || !Array.isArray(state.entities)) {
      warnings.push('SceneSyncState.entities must be an array; ignoring sync.');
      return { applied: 0, added: 0, updated: 0, removed: 0, warnings };
    }

    if (state.preset !== undefined && hooks?.onPreset) {
      hooks.onPreset(state.preset, warnings);
    }

    const transition =
      state.transition !== undefined
        ? Math.max(0, sanitizeNum(state.transition, 0.25, 'transition', warnings))
        : 0.25;

    const validNodes: SceneSyncNode[] = [];
    const nextIds = new Set<string>();

    for (const rawNode of state.entities) {
      if (!rawNode || typeof rawNode.id !== 'string' || rawNode.id.trim() === '') {
        warnings.push('Entity without a valid string "id" skipped during SceneSync.');
        continue;
      }
      if (nextIds.has(rawNode.id)) {
        warnings.push(`Duplicate entity id "${rawNode.id}" skipped.`);
        continue;
      }
      nextIds.add(rawNode.id);
      validNodes.push(rawNode);
    }

    // 1. Remove deleted entities
    let removed = 0;
    for (const [oldName, oldId] of this.nameToEntityId.entries()) {
      if (!nextIds.has(oldName)) {
        this.cancelTween(oldId);
        this.store?.free(oldId);
        this.nameToEntityId.delete(oldName);
        hooks?.onEntityRemoved?.(oldName, oldId);
        removed++;
      }
    }

    // 2. Topological sort for `on` dependencies
    const sortedNodes = this.layout.topologicalSort(validNodes, warnings);

    let added = 0;
    let updated = 0;

    if (!this.store) {
      // Lightweight mode without WorldStore
      for (const node of sortedNodes) {
        if (this.nameToEntityId.has(node.id)) {
          updated++;
        } else {
          this.nameToEntityId.set(node.id, added + updated);
          added++;
        }
      }
      return { applied: added + updated, added, updated, removed, warnings };
    }

    const store = this.store;
    this.ensureTweenCapacity(store.capacity);

    for (const node of sortedNodes) {
      const existingId = this.nameToEntityId.get(node.id);
      const isNew = existingId === undefined;
      const entityId = isNew ? store.allocate() : existingId;

      if (store.capacity > this.tweenCapacity) {
        this.ensureTweenCapacity(store.capacity);
      }

      if (isNew) {
        this.nameToEntityId.set(node.id, entityId);
        hooks?.onEntityCreated?.(node.id, entityId);
        added++;
      } else {
        updated++;
      }

      // Type
      const mappedType = VALID_TYPE_MAP[node.type];
      if (mappedType === undefined) {
        warnings.push(`Unknown entity type "${String(node.type)}" for "${node.id}"; falling back to "box".`);
        store.geometryTypes[entityId] = GeometryTypeId.Box;
      } else {
        store.geometryTypes[entityId] = mappedType;
      }

      // Scale / Size / Radius
      const pIdx = entityId * 3;
      if (node.size !== undefined) {
        const [sx, sy, sz] = sanitizeVec3(node.size, [1, 1, 1], `${node.id}.size`, warnings);
        store.scales[pIdx] = sx;
        store.scales[pIdx + 1] = sy;
        store.scales[pIdx + 2] = sz;
      } else if (node.radius !== undefined) {
        const r = Math.max(0.001, sanitizeNum(node.radius, 0.5, `${node.id}.radius`, warnings));
        const d = r * 2;
        store.scales[pIdx] = d;
        store.scales[pIdx + 1] = d;
        store.scales[pIdx + 2] = d;
      } else if (node.scale !== undefined) {
        if (typeof node.scale === 'number') {
          const s = sanitizeNum(node.scale, 1, `${node.id}.scale`, warnings);
          store.scales[pIdx] = s;
          store.scales[pIdx + 1] = s;
          store.scales[pIdx + 2] = s;
        } else {
          const [sx, sy, sz] = sanitizeVec3(node.scale, [1, 1, 1], `${node.id}.scale`, warnings);
          store.scales[pIdx] = sx;
          store.scales[pIdx + 1] = sy;
          store.scales[pIdx + 2] = sz;
        }
      }

      // Visibility
      if (node.visible !== undefined) {
        store.visible[entityId] = node.visible ? 1 : 0;
      }

      // Capture previous values before applying target transform/color for smooth tweening
      const prevPx = store.positions[pIdx] ?? 0;
      const prevPy = store.positions[pIdx + 1] ?? 0;
      const prevPz = store.positions[pIdx + 2] ?? 0;
      const qIdx = entityId * 4;
      const prevRx = store.rotations[qIdx] ?? 0;
      const prevRy = store.rotations[qIdx + 1] ?? 0;
      const prevRz = store.rotations[qIdx + 2] ?? 0;
      const prevRw = store.rotations[qIdx + 3] ?? 1;
      const prevCr = store.colors[qIdx] ?? 1;
      const prevCg = store.colors[qIdx + 1] ?? 1;
      const prevCb = store.colors[qIdx + 2] ?? 1;
      const prevCa = store.colors[qIdx + 3] ?? 1;

      // Material & Color
      const mat = resolveMaterial(node.material, warnings);
      const targetColor =
        node.color !== undefined ? parseColor(node.color, warnings) : mat.color;
      store.pbrParams[qIdx] =
        node.metalness !== undefined
          ? Math.min(1, Math.max(0, sanitizeNum(node.metalness, mat.metalness, `${node.id}.metalness`, warnings)))
          : mat.metalness;
      store.pbrParams[qIdx + 1] =
        node.roughness !== undefined
          ? Math.min(1, Math.max(0.04, sanitizeNum(node.roughness, mat.roughness, `${node.id}.roughness`, warnings)))
          : mat.roughness;
      store.pbrParams[qIdx + 2] =
        node.emissive !== undefined
          ? Math.max(0, sanitizeNum(node.emissive, mat.emissive, `${node.id}.emissive`, warnings))
          : mat.emissive;
      store.pbrParams[qIdx + 3] = mat.materialId;

      // Target Position & Rotation
      if (node.position !== undefined) {
        const [px, py, pz] = sanitizeVec3(node.position, [0, 0, 0], `${node.id}.position`, warnings);
        store.positions[pIdx] = px;
        store.positions[pIdx + 1] = py;
        store.positions[pIdx + 2] = pz;
      }

      if (node.rotation !== undefined) {
        if (Array.isArray(node.rotation) && node.rotation.length === 4) {
          store.rotations[qIdx] = sanitizeNum(node.rotation[0], 0, `${node.id}.rotation[0]`, warnings);
          store.rotations[qIdx + 1] = sanitizeNum(node.rotation[1], 0, `${node.id}.rotation[1]`, warnings);
          store.rotations[qIdx + 2] = sanitizeNum(node.rotation[2], 0, `${node.id}.rotation[2]`, warnings);
          store.rotations[qIdx + 3] = sanitizeNum(node.rotation[3], 1, `${node.id}.rotation[3]`, warnings);
        } else {
          const [rx, ry, rz] = sanitizeVec3(node.rotation, [0, 0, 0], `${node.id}.rotation`, warnings);
          quat.fromEuler(store.rotations, rx, ry, rz, qIdx);
        }
      }

      store.markDirty(entityId, DirtyFlag.Transform | DirtyFlag.Bounds);
      store.updateMatrices();

      // Relative SpatialLayout placement (resolves against target AABBs)
      if (node.on !== undefined || node.offset !== undefined || node.lookAt !== undefined) {
        this.layout.applyPlacement(
          store,
          entityId,
          node,
          (name) => this.nameToEntityId.get(name),
          warnings,
        );
      }

      const targetPx = store.positions[pIdx] ?? 0;
      const targetPy = store.positions[pIdx + 1] ?? 0;
      const targetPz = store.positions[pIdx + 2] ?? 0;
      const targetRx = store.rotations[qIdx] ?? 0;
      const targetRy = store.rotations[qIdx + 1] ?? 0;
      const targetRz = store.rotations[qIdx + 2] ?? 0;
      const targetRw = store.rotations[qIdx + 3] ?? 1;

      if (!isNew && transition > 0) {
        // Schedule zero-allocation tween from previous state to target state
        this.startTween(
          entityId,
          transition,
          prevPx, prevPy, prevPz,
          targetPx, targetPy, targetPz,
          prevRx, prevRy, prevRz, prevRw,
          targetRx, targetRy, targetRz, targetRw,
          prevCr, prevCg, prevCb, prevCa,
          targetColor[0], targetColor[1], targetColor[2], targetColor[3],
        );
        // Restore start state for frame 0 of tween
        store.positions[pIdx] = prevPx;
        store.positions[pIdx + 1] = prevPy;
        store.positions[pIdx + 2] = prevPz;
        store.rotations[qIdx] = prevRx;
        store.rotations[qIdx + 1] = prevRy;
        store.rotations[qIdx + 2] = prevRz;
        store.rotations[qIdx + 3] = prevRw;
        store.colors[qIdx] = prevCr;
        store.colors[qIdx + 1] = prevCg;
        store.colors[qIdx + 2] = prevCb;
        store.colors[qIdx + 3] = prevCa;
        store.markDirty(entityId, DirtyFlag.Transform);
        store.updateMatrices();
      } else {
        store.colors[qIdx] = targetColor[0];
        store.colors[qIdx + 1] = targetColor[1];
        store.colors[qIdx + 2] = targetColor[2];
        store.colors[qIdx + 3] = targetColor[3];
      }
    }

    if (state.camera && hooks?.onCamera) {
      hooks.onCamera(state.camera, (name) => this.nameToEntityId.get(name), warnings);
    }

    return {
      applied: added + updated,
      added,
      updated,
      removed,
      warnings,
    };
  }

  private startTween(
    id: number,
    duration: number,
    spx: number, spy: number, spz: number,
    tpx: number, tpy: number, tpz: number,
    srx: number, sry: number, srz: number, srw: number,
    trx: number, try_: number, trz: number, trw: number,
    scr: number, scg: number, scb: number, sca: number,
    tcr: number, tcg: number, tcb: number, tca: number,
  ): void {
    if ((this.tweenDuration[id] ?? 0) <= 0) {
      this.tweenActiveCount++;
    }
    this.tweenDuration[id] = duration;
    this.tweenElapsed[id] = 0;

    const pIdx = id * 3;
    this.tweenStartPos[pIdx] = spx;
    this.tweenStartPos[pIdx + 1] = spy;
    this.tweenStartPos[pIdx + 2] = spz;
    this.tweenTargetPos[pIdx] = tpx;
    this.tweenTargetPos[pIdx + 1] = tpy;
    this.tweenTargetPos[pIdx + 2] = tpz;

    const qIdx = id * 4;
    this.tweenStartRot[qIdx] = srx;
    this.tweenStartRot[qIdx + 1] = sry;
    this.tweenStartRot[qIdx + 2] = srz;
    this.tweenStartRot[qIdx + 3] = srw;
    this.tweenTargetRot[qIdx] = trx;
    this.tweenTargetRot[qIdx + 1] = try_;
    this.tweenTargetRot[qIdx + 2] = trz;
    this.tweenTargetRot[qIdx + 3] = trw;

    this.tweenStartColor[qIdx] = scr;
    this.tweenStartColor[qIdx + 1] = scg;
    this.tweenStartColor[qIdx + 2] = scb;
    this.tweenStartColor[qIdx + 3] = sca;
    this.tweenTargetColor[qIdx] = tcr;
    this.tweenTargetColor[qIdx + 1] = tcg;
    this.tweenTargetColor[qIdx + 2] = tcb;
    this.tweenTargetColor[qIdx + 3] = tca;
  }

  private cancelTween(id: number): void {
    if (id < this.tweenCapacity && (this.tweenDuration[id] ?? 0) > 0) {
      this.tweenDuration[id] = 0;
      this.tweenActiveCount = Math.max(0, this.tweenActiveCount - 1);
    }
  }

  /**
   * Zero-Allocation Tween-Update für den Render-Loop.
   */
  updateTweens(dt: number): void {
    if (this.tweenActiveCount === 0 || !this.store) return;

    const store = this.store;
    const n = Math.min(store.highWaterMark, this.tweenCapacity);

    for (let id = 0; id < n; id++) {
      const dur = this.tweenDuration[id] ?? 0;
      if (dur <= 0) continue;

      const nextElapsed = (this.tweenElapsed[id] ?? 0) + dt;
      this.tweenElapsed[id] = nextElapsed;
      const rawT = Math.min(1, nextElapsed / dur);
      const t = rawT * rawT * (3 - 2 * rawT); // smoothstep

      const pIdx = id * 3;
      const spx = this.tweenStartPos[pIdx] ?? 0;
      const spy = this.tweenStartPos[pIdx + 1] ?? 0;
      const spz = this.tweenStartPos[pIdx + 2] ?? 0;
      store.positions[pIdx] = spx + ((this.tweenTargetPos[pIdx] ?? 0) - spx) * t;
      store.positions[pIdx + 1] = spy + ((this.tweenTargetPos[pIdx + 1] ?? 0) - spy) * t;
      store.positions[pIdx + 2] = spz + ((this.tweenTargetPos[pIdx + 2] ?? 0) - spz) * t;

      const qIdx = id * 4;
      const srx = this.tweenStartRot[qIdx] ?? 0;
      const sry = this.tweenStartRot[qIdx + 1] ?? 0;
      const srz = this.tweenStartRot[qIdx + 2] ?? 0;
      const srw = this.tweenStartRot[qIdx + 3] ?? 1;
      let rx = srx + ((this.tweenTargetRot[qIdx] ?? 0) - srx) * t;
      let ry = sry + ((this.tweenTargetRot[qIdx + 1] ?? 0) - sry) * t;
      let rz = srz + ((this.tweenTargetRot[qIdx + 2] ?? 0) - srz) * t;
      let rw = srw + ((this.tweenTargetRot[qIdx + 3] ?? 1) - srw) * t;
      const qLen = Math.hypot(rx, ry, rz, rw) || 1;
      store.rotations[qIdx] = rx / qLen;
      store.rotations[qIdx + 1] = ry / qLen;
      store.rotations[qIdx + 2] = rz / qLen;
      store.rotations[qIdx + 3] = rw / qLen;

      const scr = this.tweenStartColor[qIdx] ?? 1;
      const scg = this.tweenStartColor[qIdx + 1] ?? 1;
      const scb = this.tweenStartColor[qIdx + 2] ?? 1;
      const sca = this.tweenStartColor[qIdx + 3] ?? 1;
      store.colors[qIdx] = scr + ((this.tweenTargetColor[qIdx] ?? 1) - scr) * t;
      store.colors[qIdx + 1] = scg + ((this.tweenTargetColor[qIdx + 1] ?? 1) - scg) * t;
      store.colors[qIdx + 2] = scb + ((this.tweenTargetColor[qIdx + 2] ?? 1) - scb) * t;
      store.colors[qIdx + 3] = sca + ((this.tweenTargetColor[qIdx + 3] ?? 1) - sca) * t;

      store.markDirty(id, DirtyFlag.Transform | DirtyFlag.Bounds);

      if (rawT >= 1) {
        this.tweenDuration[id] = 0;
        this.tweenActiveCount = Math.max(0, this.tweenActiveCount - 1);
      }
    }
  }
}

function growFloat32(src: Float32Array, nextLength: number): Float32Array {
  const dst = new Float32Array(nextLength);
  dst.set(src);
  return dst;
}
