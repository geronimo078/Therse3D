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

export type SpatialSide = 'left' | 'right' | 'front' | 'back';
export type SpatialGroupLayoutKind = 'grid' | 'ring' | 'row';

export interface SceneLayoutGroup {
  kind: SpatialGroupLayoutKind;
  entities: string[];
  center?: [number, number, number];
  radius?: number;
  spacing?: number;
  columns?: number;
  lookAtCenter?: boolean;
}

export interface SpatialPlacement {
  on?: string;
  beside?: string;
  side?: SpatialSide;
  inside?: string;
  align?: SpatialAlign;
  margin?: number;
  offset?: [number, number, number];
  lookAt?: string | [number, number, number];
  avoidOverlap?: boolean;
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
   * Topologische Sortierung von Layout-Nodes nach `on`- und `parent`-Abhängigkeiten.
   * Erkennt Zyklen fehlertolerant und meldet Warnungen statt zu crashen.
   */
  topologicalSort<T extends { id: string; on?: string; beside?: string; inside?: string; parent?: string }>(
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
        warnings?.push(`Circular "on"/"beside"/"inside"/"parent" dependency detected involving "${node.id}".`);
        return;
      }
      visiting.add(node.id);
      const deps = [node.parent, node.on, node.beside, node.inside];
      for (const depId of deps) {
        if (depId && depId !== node.id) {
          const depNode = byId.get(depId);
          if (depNode) {
            visit(depNode);
          }
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

  computeBesidePosition(
    target: AABB,
    subjectSize: [number, number, number],
    placement: SpatialPlacement,
  ): [number, number, number] {
    const halfX = subjectSize[0] * 0.5;
    const halfY = subjectSize[1] * 0.5;
    const halfZ = subjectSize[2] * 0.5;
    const margin = Number.isFinite(placement.margin) ? placement.margin! : 0.1;

    const targetCenterX = (target.min[0] + target.max[0]) * 0.5;
    const targetCenterZ = (target.min[2] + target.max[2]) * 0.5;
    const side: SpatialSide =
      placement.side ??
      (placement.align?.includes('left')
        ? 'left'
        : placement.align?.includes('front')
          ? 'front'
          : placement.align?.includes('back')
            ? 'back'
            : 'right');

    let x = targetCenterX;
    let z = targetCenterZ;
    if (side === 'left') {
      x = target.min[0] - halfX - margin;
    } else if (side === 'right') {
      x = target.max[0] + halfX + margin;
    } else if (side === 'front') {
      z = target.max[2] + halfZ + margin;
    } else {
      z = target.min[2] - halfZ - margin;
    }

    const y = target.min[1] + halfY;
    const ox = placement.offset && Number.isFinite(placement.offset[0]) ? placement.offset[0] : 0;
    const oy = placement.offset && Number.isFinite(placement.offset[1]) ? placement.offset[1] : 0;
    const oz = placement.offset && Number.isFinite(placement.offset[2]) ? placement.offset[2] : 0;
    return [x + ox, y + oy, z + oz];
  }

  computeInsidePosition(
    target: AABB,
    subjectSize: [number, number, number],
    placement: SpatialPlacement,
  ): [number, number, number] {
    const [x, , z] = this.computeRelativePosition(target, subjectSize, placement);
    const halfY = subjectSize[1] * 0.5;
    const margin = Number.isFinite(placement.margin) ? placement.margin! : 0;
    const oy = placement.offset && Number.isFinite(placement.offset[1]) ? placement.offset[1] : 0;
    const y = target.min[1] + halfY + margin + oy;
    return [x, y, z];
  }

  resolveOverlapXZ(
    store: WorldStore,
    entityId: number,
    ignoreId?: number,
    warnings?: string[],
  ): boolean {
    let moved = false;
    const pIdx = entityId * 3;
    const bMin = store.boundsMin;
    const bMax = store.boundsMax;

    for (let other = 0; other < store.highWaterMark; other++) {
      if (other === entityId || other === ignoreId) continue;
      if (store.alive[other] === 0 || store.visible[other] === 0) continue;
      if ((store.geometryTypes[other] ?? 0) === GeometryTypeId.Plane) continue;
      if ((store.geometryTypes[other] ?? 0) === GeometryTypeId.Light) continue;

      const oIdx = other * 3;
      const overlapX = Math.min(bMax[pIdx] ?? 0, bMax[oIdx] ?? 0) - Math.max(bMin[pIdx] ?? 0, bMin[oIdx] ?? 0);
      const overlapY = Math.min(bMax[pIdx + 1] ?? 0, bMax[oIdx + 1] ?? 0) - Math.max(bMin[pIdx + 1] ?? 0, bMin[oIdx + 1] ?? 0);
      const overlapZ = Math.min(bMax[pIdx + 2] ?? 0, bMax[oIdx + 2] ?? 0) - Math.max(bMin[pIdx + 2] ?? 0, bMin[oIdx + 2] ?? 0);

      if (overlapX > 0.001 && overlapY > 0.001 && overlapZ > 0.001) {
        const cx = ((bMin[pIdx] ?? 0) + (bMax[pIdx] ?? 0)) * 0.5;
        const cz = ((bMin[pIdx + 2] ?? 0) + (bMax[pIdx + 2] ?? 0)) * 0.5;
        const ocx = ((bMin[oIdx] ?? 0) + (bMax[oIdx] ?? 0)) * 0.5;
        const ocz = ((bMin[oIdx + 2] ?? 0) + (bMax[oIdx + 2] ?? 0)) * 0.5;

        if (overlapX <= overlapZ) {
          const dir = cx >= ocx ? 1 : -1;
          store.positions[pIdx] = (store.positions[pIdx] ?? 0) + dir * (overlapX + 0.02);
        } else {
          const dir = cz >= ocz ? 1 : -1;
          store.positions[pIdx + 2] = (store.positions[pIdx + 2] ?? 0) + dir * (overlapZ + 0.02);
        }
        store.markDirty(entityId, DirtyFlag.Transform | DirtyFlag.Bounds);
        store.updateMatrices();
        moved = true;
      }
    }

    if (moved) {
      warnings?.push(`Adjusted position of entity #${entityId} to avoid overlap.`);
    }
    return moved;
  }

  private resolveTargetAABB(
    store: WorldStore,
    targetName: string,
    kind: string,
    resolveEntityId: (name: string) => number | undefined,
    warnings?: string[],
  ): { aabb: AABB; targetId: number | undefined } {
    const targetId = resolveEntityId(targetName);
    if (targetId !== undefined) {
      return { aabb: this.getEntityAABB(store, targetId), targetId };
    }
    if (targetName === 'floor') {
      return { aabb: FLOOR_AABB, targetId: undefined };
    }
    warnings?.push(`Unknown "${kind}" target "${targetName}"; placing relative to floor instead.`);
    return { aabb: FLOOR_AABB, targetId: undefined };
  }

  applyPlacement(
    store: WorldStore,
    entityId: number,
    placement: SpatialPlacement,
    resolveEntityId: (name: string) => number | undefined,
    warnings?: string[],
  ): void {
    let ignoreOverlapId: number | undefined;

    if (placement.on !== undefined) {
      const resolved = this.resolveTargetAABB(store, placement.on, 'on', resolveEntityId, warnings);
      ignoreOverlapId = resolved.targetId;
      const [hx, hy, hz] = this.getHalfExtents(store, entityId);
      const [px, py, pz] = this.computeRelativePosition(resolved.aabb, [hx * 2, hy * 2, hz * 2], placement);
      const pIdx = entityId * 3;
      store.positions[pIdx] = px;
      store.positions[pIdx + 1] = py;
      store.positions[pIdx + 2] = pz;
      store.markDirty(entityId, DirtyFlag.Transform | DirtyFlag.Bounds);
      store.updateMatrices();
    } else if (placement.beside !== undefined) {
      const resolved = this.resolveTargetAABB(store, placement.beside, 'beside', resolveEntityId, warnings);
      ignoreOverlapId = resolved.targetId;
      const [hx, hy, hz] = this.getHalfExtents(store, entityId);
      const [px, py, pz] = this.computeBesidePosition(resolved.aabb, [hx * 2, hy * 2, hz * 2], placement);
      const pIdx = entityId * 3;
      store.positions[pIdx] = px;
      store.positions[pIdx + 1] = py;
      store.positions[pIdx + 2] = pz;
      store.markDirty(entityId, DirtyFlag.Transform | DirtyFlag.Bounds);
      store.updateMatrices();
    } else if (placement.inside !== undefined) {
      const resolved = this.resolveTargetAABB(store, placement.inside, 'inside', resolveEntityId, warnings);
      ignoreOverlapId = resolved.targetId;
      const [hx, hy, hz] = this.getHalfExtents(store, entityId);
      const [px, py, pz] = this.computeInsidePosition(resolved.aabb, [hx * 2, hy * 2, hz * 2], placement);
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

    if (placement.avoidOverlap) {
      this.resolveOverlapXZ(store, entityId, ignoreOverlapId, warnings);
    }
  }

  arrangeGroup(
    store: WorldStore,
    group: SceneLayoutGroup,
    resolveEntityId: (name: string) => number | undefined,
    warnings?: string[],
  ): void {
    const ids: number[] = [];
    for (const name of group.entities) {
      const id = resolveEntityId(name);
      if (id !== undefined) {
        ids.push(id);
      } else {
        warnings?.push(`Layout group references unknown entity "${name}".`);
      }
    }
    if (ids.length === 0) return;

    const cx = group.center && Number.isFinite(group.center[0]) ? group.center[0] : 0;
    const cy = group.center && Number.isFinite(group.center[1]) ? group.center[1] : 0;
    const cz = group.center && Number.isFinite(group.center[2]) ? group.center[2] : 0;
    const spacing = Number.isFinite(group.spacing) ? Math.max(0.1, group.spacing!) : 1.2;
    const radius = Number.isFinite(group.radius) ? Math.max(0.2, group.radius!) : 2.0;

    if (group.kind === 'ring') {
      const step = (Math.PI * 2) / ids.length;
      for (let i = 0; i < ids.length; i++) {
        const id = ids[i]!;
        const angle = i * step;
        const pIdx = id * 3;
        const px = cx + Math.sin(angle) * radius;
        const pz = cz + Math.cos(angle) * radius;
        store.positions[pIdx] = px;
        store.positions[pIdx + 1] = cy + (store.positions[pIdx + 1] ?? 0);
        store.positions[pIdx + 2] = pz;
        if (group.lookAtCenter) {
          quat.fromLookAt(store.rotations, id * 4, px, store.positions[pIdx + 1] ?? 0, pz, cx, store.positions[pIdx + 1] ?? 0, cz);
        }
        store.markDirty(id, DirtyFlag.Transform | DirtyFlag.Bounds);
      }
    } else if (group.kind === 'grid') {
      const cols = Math.max(1, Math.floor(group.columns ?? Math.ceil(Math.sqrt(ids.length))));
      const rows = Math.ceil(ids.length / cols);
      const startX = cx - ((cols - 1) * spacing) * 0.5;
      const startZ = cz - ((rows - 1) * spacing) * 0.5;
      for (let i = 0; i < ids.length; i++) {
        const id = ids[i]!;
        const col = i % cols;
        const row = Math.floor(i / cols);
        const pIdx = id * 3;
        store.positions[pIdx] = startX + col * spacing;
        store.positions[pIdx + 1] = cy + (store.positions[pIdx + 1] ?? 0);
        store.positions[pIdx + 2] = startZ + row * spacing;
        store.markDirty(id, DirtyFlag.Transform | DirtyFlag.Bounds);
      }
    } else {
      const startX = cx - ((ids.length - 1) * spacing) * 0.5;
      for (let i = 0; i < ids.length; i++) {
        const id = ids[i]!;
        const pIdx = id * 3;
        store.positions[pIdx] = startX + i * spacing;
        store.positions[pIdx + 1] = cy + (store.positions[pIdx + 1] ?? 0);
        store.positions[pIdx + 2] = cz;
        store.markDirty(id, DirtyFlag.Transform | DirtyFlag.Bounds);
      }
    }
    store.updateMatrices();
  }
}
