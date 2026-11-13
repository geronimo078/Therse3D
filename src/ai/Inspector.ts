import type { WorldStore } from '../core/WorldStore';

export type ScreenRegion =
  | 'center'
  | 'top-left'
  | 'top'
  | 'top-right'
  | 'left'
  | 'right'
  | 'bottom-left'
  | 'bottom'
  | 'bottom-right'
  | 'offscreen-left'
  | 'offscreen-right'
  | 'offscreen-top'
  | 'offscreen-bottom'
  | 'behind-camera';

export interface EntityRelations {
  on: string[];
  near: string[];
  collidingWith: string[];
}

export interface EntityDescription {
  id: string;
  entityId: number;
  position: [number, number, number];
  scale: [number, number, number];
  visible: boolean;
  screenRegion: ScreenRegion;
  screenCoverage: number;
  /** Normalized viewport box [xMin, yMin, xMax, yMax] in 0..1 */
  screenBox: [number, number, number, number];
  relations: EntityRelations;
}

export interface SceneDescription {
  entityCount: number;
  entities: EntityDescription[];
  warnings: string[];
}

export interface DescribeContext {
  viewProjMatrix?: Float32Array;
  resolveName?: (entityId: number) => string | undefined;
}

const EPSILON = 1e-3;
const NEAR_THRESHOLD = 0.75;

/**
 * Provides a compact, token-efficient spatial graph & frustum/collision warnings for LLMs.
 */
export class Inspector {
  private readonly store: WorldStore;

  constructor(store: WorldStore) {
    this.store = store;
  }

  describe(context: DescribeContext = {}): SceneDescription {
    const store = this.store;
    const vp = context.viewProjMatrix;
    const entities: EntityDescription[] = [];
    const warnings: string[] = [];

    const activeIds: number[] = [];
    const names: string[] = [];

    for (let i = 0; i < store.highWaterMark; i++) {
      if (store.alive[i] === 0) continue;
      activeIds.push(i);
      names.push(context.resolveName?.(i) ?? `#${i}`);
    }

    for (let a = 0; a < activeIds.length; a++) {
      const id = activeIds[a]!;
      const name = names[a]!;
      const pIdx = id * 3;

      const px = store.positions[pIdx] ?? 0;
      const py = store.positions[pIdx + 1] ?? 0;
      const pz = store.positions[pIdx + 2] ?? 0;

      const sx = store.scales[pIdx] ?? 1;
      const sy = store.scales[pIdx + 1] ?? 1;
      const sz = store.scales[pIdx + 2] ?? 1;

      const minX = store.boundsMin[pIdx] ?? 0;
      const minY = store.boundsMin[pIdx + 1] ?? 0;
      const minZ = store.boundsMin[pIdx + 2] ?? 0;
      const maxX = store.boundsMax[pIdx] ?? 0;
      const maxY = store.boundsMax[pIdx + 1] ?? 0;
      const maxZ = store.boundsMax[pIdx + 2] ?? 0;

      const isEnabled = (store.visible[id] ?? 0) === 1;
      const proj = vp
        ? projectAABB(vp, minX, minY, minZ, maxX, maxY, maxZ)
        : { visible: isEnabled, screenRegion: 'center' as ScreenRegion, screenCoverage: 0, screenBox: [0, 0, 0, 0] as [number, number, number, number] };

      const visible = isEnabled && proj.visible;
      if (isEnabled && !proj.visible) {
        warnings.push(`Entity '${name}' is outside camera view (${proj.screenRegion})`);
      }

      const relations: EntityRelations = { on: [], near: [], collidingWith: [] };

      for (let b = 0; b < activeIds.length; b++) {
        if (a === b) continue;
        const otherId = activeIds[b]!;
        const otherName = names[b]!;
        const oIdx = otherId * 3;

        const oMinX = store.boundsMin[oIdx] ?? 0;
        const oMinY = store.boundsMin[oIdx + 1] ?? 0;
        const oMinZ = store.boundsMin[oIdx + 2] ?? 0;
        const oMaxX = store.boundsMax[oIdx] ?? 0;
        const oMaxY = store.boundsMax[oIdx + 1] ?? 0;
        const oMaxZ = store.boundsMax[oIdx + 2] ?? 0;

        // Check if `id` is resting on `otherId`
        const xOverlap = Math.min(maxX, oMaxX) - Math.max(minX, oMinX);
        const zOverlap = Math.min(maxZ, oMaxZ) - Math.max(minZ, oMinZ);
        const yOverlap = Math.min(maxY, oMaxY) - Math.max(minY, oMinY);

        if (xOverlap > EPSILON && zOverlap > EPSILON && Math.abs(minY - oMaxY) <= 0.02) {
          relations.on.push(otherName);
        } else if (xOverlap > EPSILON && yOverlap > EPSILON && zOverlap > EPSILON) {
          relations.collidingWith.push(otherName);
          if (id < otherId) {
            warnings.push(`Entity '${name}' intersects with '${otherName}'`);
          }
        } else {
          // Distance between AABBs
          const dx = Math.max(0, minX - oMaxX, oMinX - maxX);
          const dy = Math.max(0, minY - oMaxY, oMinY - maxY);
          const dz = Math.max(0, minZ - oMaxZ, oMinZ - maxZ);
          if (Math.hypot(dx, dy, dz) <= NEAR_THRESHOLD) {
            relations.near.push(otherName);
          }
        }
      }

      entities.push({
        id: name,
        entityId: id,
        position: [px, py, pz],
        scale: [sx, sy, sz],
        visible,
        screenRegion: proj.screenRegion,
        screenCoverage: proj.screenCoverage,
        screenBox: proj.screenBox,
        relations,
      });
    }

    return {
      entityCount: entities.length,
      entities,
      warnings,
    };
  }
}

function projectAABB(
  vp: Float32Array,
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
): {
  visible: boolean;
  screenRegion: ScreenRegion;
  screenCoverage: number;
  screenBox: [number, number, number, number];
} {
  let ndcMinX = Infinity;
  let ndcMinY = Infinity;
  let ndcMaxX = -Infinity;
  let ndcMaxY = -Infinity;
  let inFrontCount = 0;

  for (let i = 0; i < 8; i++) {
    const x = (i & 1) ? maxX : minX;
    const y = (i & 2) ? maxY : minY;
    const z = (i & 4) ? maxZ : minZ;

    const cx = (vp[0] ?? 0) * x + (vp[4] ?? 0) * y + (vp[8] ?? 0) * z + (vp[12] ?? 0);
    const cy = (vp[1] ?? 0) * x + (vp[5] ?? 0) * y + (vp[9] ?? 0) * z + (vp[13] ?? 0);
    const cw = (vp[3] ?? 0) * x + (vp[7] ?? 0) * y + (vp[11] ?? 0) * z + (vp[15] ?? 1);

    if (cw > 1e-4) {
      inFrontCount++;
      const nx = cx / cw;
      const ny = cy / cw;
      if (nx < ndcMinX) ndcMinX = nx;
      if (ny < ndcMinY) ndcMinY = ny;
      if (nx > ndcMaxX) ndcMaxX = nx;
      if (ny > ndcMaxY) ndcMaxY = ny;
    }
  }

  if (inFrontCount === 0) {
    return {
      visible: false,
      screenRegion: 'behind-camera',
      screenCoverage: 0,
      screenBox: [0, 0, 0, 0],
    };
  }

  const offLeft = ndcMaxX < -1;
  const offRight = ndcMinX > 1;
  const offBottom = ndcMaxY < -1;
  const offTop = ndcMinY > 1;

  if (offLeft) return { visible: false, screenRegion: 'offscreen-left', screenCoverage: 0, screenBox: [0, 0, 0, 0] };
  if (offRight) return { visible: false, screenRegion: 'offscreen-right', screenCoverage: 0, screenBox: [0, 0, 0, 0] };
  if (offTop) return { visible: false, screenRegion: 'offscreen-top', screenCoverage: 0, screenBox: [0, 0, 0, 0] };
  if (offBottom) return { visible: false, screenRegion: 'offscreen-bottom', screenCoverage: 0, screenBox: [0, 0, 0, 0] };

  // Convert NDC (-1..1, +Y up) to normalized screen space (0..1, +Y down)
  const sxMin = Math.max(0, (ndcMinX + 1) * 0.5);
  const sxMax = Math.min(1, (ndcMaxX + 1) * 0.5);
  const syMin = Math.max(0, (1 - ndcMaxY) * 0.5);
  const syMax = Math.min(1, (1 - ndcMinY) * 0.5);

  const coverage = Math.round(Math.max(0, sxMax - sxMin) * Math.max(0, syMax - syMin) * 1000) / 10;
  const cx = (sxMin + sxMax) * 0.5;
  const cy = (syMin + syMax) * 0.5;

  const h = cx < 0.33 ? 'left' : cx > 0.66 ? 'right' : '';
  const v = cy < 0.33 ? 'top' : cy > 0.66 ? 'bottom' : '';
  const region: ScreenRegion = v && h ? `${v}-${h}` : (v || h || 'center') as ScreenRegion;

  return {
    visible: true,
    screenRegion: region,
    screenCoverage: coverage,
    screenBox: [sxMin, syMin, sxMax, syMax],
  };
}
