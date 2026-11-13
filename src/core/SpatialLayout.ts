import { quat } from './Math';
import { DirtyFlag, GeometryTypeId, type WorldStore } from './WorldStore';

/**
 * Bounding-Box-Berechnung & relative Platzierung (on, align, margin, offset, lookAt).
 */
export type SpatialAlign =
  | 'center'
  | 'front'
  | 'back'
  | 'left'
  | 'right'
  | 'front-left'
  | 'front-right'
  | 'back-left'
  | 'back-right';

export interface SpatialPlacement {
  on?: string;
  align?: SpatialAlign;
  margin?: number;
  offset?: [number, number, number];
  lookAt?: string | [number, number, number];
}

export interface AABB {
  min: [number, number, number];
  max: [number, number, number];
}

export interface LayoutNodeInput extends SpatialPlacement {
  id: string;
  entityId: number;
}

const FLOOR_AABB: AABB = {
  min: [-5, 0, -5],
  max: [5, 0, 5],
};

export class SpatialLayout {
  /**
   * Topologische Sortierung von Layout-Nodes nach `on`-Abhängigkeiten.
   * Erkennt Zyklen fehlertolerant und meldet Warnungen statt zu crashen.
   */
  topologicalSort<T extends { id: string; on?: string }>(
    nodes: readonly T[],
    warnings?: string[],
  ): T[] {
    const byId = new Map<string, T>();
    for (const node of nodes) {
      if (node.id) {
        byId.set(node.id, node);
      }
    }

    const visited = new Set<string>();
    const visiting = new Set<string>();
    const sorted: T[] = [];

    const visit = (node: T) => {
      if (visited.has(node.id)) return;
      if (visiting.has(node.id)) {
        warnings?.push(`Circular "on" dependency detected involving "${node.id}".`);
        return;
      }
      visiting.add(node.id);
      if (node.on && node.on !== node.id) {
        const parent = byId.get(node.on);
        if (parent) {
          visit(parent);
        }
      }
      visiting.delete(node.id);
      visited.add(node.id);
      sorted.push(node);
    };

    for (const node of nodes) {
      visit(node);
    }
    return sorted;
  }

  getEntityAABB(store: WorldStore, entityId: number): AABB {
    const idx = entityId * 3;
    return {
      min: [
        store.boundsMin[idx] ?? 0,
        store.boundsMin[idx + 1] ?? 0,
        store.boundsMin[idx + 2] ?? 0,
      ],
      max: [
        store.boundsMax[idx] ?? 0,
        store.boundsMax[idx + 1] ?? 0,
        store.boundsMax[idx + 2] ?? 0,
      ],
    };
  }

  getHalfExtents(store: WorldStore, entityId: number): [number, number, number] {
    const idx = entityId * 3;
    const sx = Math.abs(store.scales[idx] ?? 1);
    const sy = Math.abs(store.scales[idx + 1] ?? 1);
    const sz = Math.abs(store.scales[idx + 2] ?? 1);
    const gType = store.geometryTypes[entityId] ?? GeometryTypeId.Box;
    const hy = gType === GeometryTypeId.Plane ? 0 : sy * 0.5;
    return [sx * 0.5, hy, sz * 0.5];
  }

  computeRelativePosition(
    target: AABB,
    subjectSize: [number, number, number],
    placement: SpatialPlacement,
  ): [number, number, number] {
    const halfX = subjectSize[0] * 0.5;
    const halfY = subjectSize[1] * 0.5;
    const halfZ = subjectSize[2] * 0.5;
    const margin = Number.isFinite(placement.margin) ? placement.margin! : 0;

    const targetCenterX = (target.min[0] + target.max[0]) * 0.5;
    const targetCenterZ = (target.min[2] + target.max[2]) * 0.5;

    let x = targetCenterX;
    let z = targetCenterZ;
    const align = placement.align ?? 'center';

    if (align.includes('left')) {
      x = Math.min(targetCenterX, target.min[0] + halfX + margin);
    } else if (align.includes('right')) {
      x = Math.max(targetCenterX, target.max[0] - halfX - margin);
    }

    if (align.includes('front')) {
      z = Math.max(targetCenterZ, target.max[2] - halfZ - margin);
    } else if (align.includes('back')) {
      z = Math.min(targetCenterZ, target.min[2] + halfZ + margin);
    }

    const y = target.max[1] + halfY;
    const ox = placement.offset && Number.isFinite(placement.offset[0]) ? placement.offset[0] : 0;
    const oy = placement.offset && Number.isFinite(placement.offset[1]) ? placement.offset[1] : 0;
    const oz = placement.offset && Number.isFinite(placement.offset[2]) ? placement.offset[2] : 0;

    return [x + ox, y + oy, z + oz];
  }

  applyPlacement(
    store: WorldStore,
    entityId: number,
    placement: SpatialPlacement,
    resolveEntityId: (name: string) => number | undefined,
    warnings?: string[],
  ): void {
    if (placement.on !== undefined) {
      let targetAABB: AABB | undefined;
      const targetId = resolveEntityId(placement.on);
      if (targetId !== undefined) {
        targetAABB = this.getEntityAABB(store, targetId);
      } else if (placement.on === 'floor') {
        targetAABB = FLOOR_AABB;
      } else {
        warnings?.push(`Unknown "on" target "${placement.on}"; placing on floor instead.`);
        targetAABB = FLOOR_AABB;
      }

      const [hx, hy, hz] = this.getHalfExtents(store, entityId);
      const [px, py, pz] = this.computeRelativePosition(
        targetAABB,
        [hx * 2, hy * 2, hz * 2],
        placement,
      );
      const pIdx = entityId * 3;
      store.positions[pIdx] = px;
      store.positions[pIdx + 1] = py;
      store.positions[pIdx + 2] = pz;
      store.markDirty(entityId, DirtyFlag.Transform | DirtyFlag.Bounds);
      store.updateMatrices();
    } else if (placement.offset) {
      const pIdx = entityId * 3;
      const ox = Number.isFinite(placement.offset[0]) ? placement.offset[0] : 0;
      const oy = Number.isFinite(placement.offset[1]) ? placement.offset[1] : 0;
      const oz = Number.isFinite(placement.offset[2]) ? placement.offset[2] : 0;
      store.positions[pIdx] = (store.positions[pIdx] ?? 0) + ox;
      store.positions[pIdx + 1] = (store.positions[pIdx + 1] ?? 0) + oy;
      store.positions[pIdx + 2] = (store.positions[pIdx + 2] ?? 0) + oz;
      store.markDirty(entityId, DirtyFlag.Transform | DirtyFlag.Bounds);
      store.updateMatrices();
    }

    if (placement.lookAt !== undefined) {
      let tx = 0, ty = 0, tz = 0;
      let valid = true;
      if (typeof placement.lookAt === 'string') {
        const targetId = resolveEntityId(placement.lookAt);
        if (targetId !== undefined) {
          const tIdx = targetId * 3;
          tx = store.positions[tIdx] ?? 0;
          ty = store.positions[tIdx + 1] ?? 0;
          tz = store.positions[tIdx + 2] ?? 0;
        } else {
          warnings?.push(`Unknown "lookAt" target "${placement.lookAt}".`);
          valid = false;
        }
      } else {
        tx = Number.isFinite(placement.lookAt[0]) ? placement.lookAt[0] : 0;
        ty = Number.isFinite(placement.lookAt[1]) ? placement.lookAt[1] : 0;
        tz = Number.isFinite(placement.lookAt[2]) ? placement.lookAt[2] : 0;
      }

      if (valid) {
        const pIdx = entityId * 3;
        quat.fromLookAt(
          store.rotations,
          entityId * 4,
          store.positions[pIdx] ?? 0,
          store.positions[pIdx + 1] ?? 0,
          store.positions[pIdx + 2] ?? 0,
          tx,
          ty,
          tz,
        );
        store.markDirty(entityId, DirtyFlag.Transform);
        store.updateMatrices();
      }
    }
  }
}
