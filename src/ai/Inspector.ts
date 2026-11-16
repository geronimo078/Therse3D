import { mat4, type Mat4 } from '../core/Math';
import { GeometryTypeId, MAX_MORPH_TARGETS, type WorldStore } from '../core/WorldStore';

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
  morphTargets?: string[];
  morphWeights?: Record<string, number>;
}

export interface SuggestedFix {
  kind: 'resolve-collision';
  entityId: string;
  otherEntityId: string;
  reason: string;
  patch: {
    entities: Array<{
      id: string;
      position: [number, number, number];
    }>;
  };
}

export interface ProjectedPoint {
  x: number;
  y: number;
  nx: number;
  ny: number;
  ndc: [number, number, number];
  visible: boolean;
}

export interface SceneDescription {
  entityCount: number;
  entities: EntityDescription[];
  warnings: string[];
  suggestedFixes: SuggestedFix[];
}

export interface DescribeContext {
  viewProjMatrix?: Float32Array;
  resolveName?: (entityId: number) => string | undefined;
  resolveMorphTargetNames?: (entityId: number) => string[] | undefined;
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
    const suggestedFixes: SuggestedFix[] = [];

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
            suggestedFixes.push(
              buildCollisionFix(
                name,
                otherName,
                px, py, pz,
                minX, minY, minZ, maxX, maxY, maxZ,
                oMinX, oMinY, oMinZ, oMaxX, oMaxY, oMaxZ,
                xOverlap, yOverlap, zOverlap,
              ),
            );
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

      const morphTargetNames = context.resolveMorphTargetNames?.(id);
      let activeMorphWeights: Record<string, number> | undefined;
      if (morphTargetNames && morphTargetNames.length > 0) {
        const mBase = id * MAX_MORPH_TARGETS;
        const mCount = Math.min(morphTargetNames.length, MAX_MORPH_TARGETS);
        const weightsObj: Record<string, number> = {};
        let hasActive = false;
        for (let m = 0; m < mCount; m++) {
          const w = store.morphWeights[mBase + m] ?? 0;
          if (Math.abs(w) > 1e-4) {
            weightsObj[morphTargetNames[m]!] = Number(w.toFixed(3));
            hasActive = true;
          }
        }
        if (hasActive) {
          activeMorphWeights = weightsObj;
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
        ...(morphTargetNames && morphTargetNames.length > 0 ? { morphTargets: morphTargetNames } : {}),
        ...(activeMorphWeights ? { morphWeights: activeMorphWeights } : {}),
      });
    }

    return {
      entityCount: entities.length,
      entities,
      warnings,
      suggestedFixes,
    };
  }
}

const FIX_CLEARANCE = 0.05;

function buildCollisionFix(
  entityId: string,
  otherEntityId: string,
  px: number, py: number, pz: number,
  minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number,
  oMinX: number, oMinY: number, oMinZ: number, oMaxX: number, oMaxY: number, oMaxZ: number,
  xOverlap: number, yOverlap: number, zOverlap: number,
): SuggestedFix {
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  const ocx = (oMinX + oMaxX) * 0.5;
  const ocy = (oMinY + oMaxY) * 0.5;
  const ocz = (oMinZ + oMaxZ) * 0.5;

  let nx = px;
  let ny = py;
  let nz = pz;

  if (yOverlap <= xOverlap && yOverlap <= zOverlap) {
    // Prefer lifting above the other entity if centers are close or entity is above
    const dir = cy >= ocy ? 1 : -1;
    ny = py + dir * (yOverlap + FIX_CLEARANCE);
  } else if (xOverlap <= zOverlap) {
    const dir = cx >= ocx ? 1 : -1;
    nx = px + dir * (xOverlap + FIX_CLEARANCE);
  } else {
    const dir = cz >= ocz ? 1 : -1;
    nz = pz + dir * (zOverlap + FIX_CLEARANCE);
  }

  return {
    kind: 'resolve-collision',
    entityId,
    otherEntityId,
    reason: `Separate '${entityId}' from '${otherEntityId}' along minimum penetration axis.`,
    patch: {
      entities: [
        {
          id: entityId,
          position: [
            Math.round(nx * 1000) / 1000,
            Math.round(ny * 1000) / 1000,
            Math.round(nz * 1000) / 1000,
          ],
        },
      ],
    },
  };
}

export function projectWorldPoint(
  vp: Float32Array,
  point: readonly [number, number, number],
  viewportWidth: number,
  viewportHeight: number,
): ProjectedPoint {
  const x = point[0] ?? 0;
  const y = point[1] ?? 0;
  const z = point[2] ?? 0;

  const cx = (vp[0] ?? 0) * x + (vp[4] ?? 0) * y + (vp[8] ?? 0) * z + (vp[12] ?? 0);
  const cy = (vp[1] ?? 0) * x + (vp[5] ?? 0) * y + (vp[9] ?? 0) * z + (vp[13] ?? 0);
  const cz = (vp[2] ?? 0) * x + (vp[6] ?? 0) * y + (vp[10] ?? 0) * z + (vp[14] ?? 0);
  const cw = (vp[3] ?? 0) * x + (vp[7] ?? 0) * y + (vp[11] ?? 0) * z + (vp[15] ?? 1);

  if (cw <= 1e-4) {
    return {
      x: 0,
      y: 0,
      nx: 0,
      ny: 0,
      ndc: [0, 0, -1],
      visible: false,
    };
  }

  const ndcX = cx / cw;
  const ndcY = cy / cw;
  const ndcZ = cz / cw;
  const nx = (ndcX + 1) * 0.5;
  const ny = (1 - ndcY) * 0.5;
  const visible = ndcX >= -1 && ndcX <= 1 && ndcY >= -1 && ndcY <= 1 && ndcZ >= 0 && ndcZ <= 1;

  return {
    x: nx * Math.max(1, viewportWidth),
    y: ny * Math.max(1, viewportHeight),
    nx,
    ny,
    ndc: [ndcX, ndcY, ndcZ],
    visible,
  };
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

export interface RaycastHit {
  entityId: number;
  id: string;
  distance: number;
  point: [number, number, number];
}

const INV_VP_SCRATCH = new Float32Array(16);

export function raycastStore(
  store: WorldStore,
  origin: readonly [number, number, number],
  direction: readonly [number, number, number],
  resolveName?: (entityId: number) => string | undefined,
): RaycastHit | null {
  const ox = origin[0], oy = origin[1], oz = origin[2];
  const dLen = Math.hypot(direction[0], direction[1], direction[2]);
  if (dLen < 1e-8) return null;
  const dx = direction[0] / dLen;
  const dy = direction[1] / dLen;
  const dz = direction[2] / dLen;

  const invDx = Math.abs(dx) > 1e-8 ? 1 / dx : 1e8;
  const invDy = Math.abs(dy) > 1e-8 ? 1 / dy : 1e8;
  const invDz = Math.abs(dz) > 1e-8 ? 1 / dz : 1e8;

  let bestT = Infinity;
  let bestId = -1;

  for (let id = 0; id < store.highWaterMark; id++) {
    if (store.alive[id] === 0 || store.visible[id] === 0) continue;
    const gType = store.geometryTypes[id] ?? 0;
    if (gType === GeometryTypeId.Light || gType === GeometryTypeId.Group) continue;

    const pIdx = id * 3;
    const minX = store.boundsMin[pIdx] ?? 0;
    const minY = store.boundsMin[pIdx + 1] ?? 0;
    const minZ = store.boundsMin[pIdx + 2] ?? 0;
    const maxX = store.boundsMax[pIdx] ?? 0;
    const maxY = store.boundsMax[pIdx + 1] ?? 0;
    const maxZ = store.boundsMax[pIdx + 2] ?? 0;

    const t1 = (minX - ox) * invDx;
    const t2 = (maxX - ox) * invDx;
    const t3 = (minY - oy) * invDy;
    const t4 = (maxY - oy) * invDy;
    const t5 = (minZ - oz) * invDz;
    const t6 = (maxZ - oz) * invDz;

    const tMin = Math.max(Math.min(t1, t2), Math.min(t3, t4), Math.min(t5, t6));
    const tMax = Math.min(Math.max(t1, t2), Math.max(t3, t4), Math.max(t5, t6));

    if (tMax < 0 || tMin > tMax) continue;
    const hitT = tMin >= 0 ? tMin : tMax;
    if (hitT < bestT) {
      bestT = hitT;
      bestId = id;
    }
  }

  if (bestId < 0) return null;
  return {
    entityId: bestId,
    id: resolveName?.(bestId) ?? `entity_${bestId}`,
    distance: bestT,
    point: [ox + dx * bestT, oy + dy * bestT, oz + dz * bestT],
  };
}

export function pickFromScreen(
  store: WorldStore,
  viewProjMatrix: Mat4,
  nx: number,
  ny: number,
  resolveName?: (entityId: number) => string | undefined,
): RaycastHit | null {
  if (!mat4.invert(INV_VP_SCRATCH, viewProjMatrix)) return null;
  const ndcX = nx * 2 - 1;
  const ndcY = 1 - ny * 2;

  const m = INV_VP_SCRATCH;
  const unproject = (z: number): [number, number, number] => {
    const w = (m[3] ?? 0) * ndcX + (m[7] ?? 0) * ndcY + (m[11] ?? 0) * z + (m[15] ?? 1);
    const invW = Math.abs(w) > 1e-8 ? 1 / w : 1;
    return [
      ((m[0] ?? 0) * ndcX + (m[4] ?? 0) * ndcY + (m[8] ?? 0) * z + (m[12] ?? 0)) * invW,
      ((m[1] ?? 0) * ndcX + (m[5] ?? 0) * ndcY + (m[9] ?? 0) * z + (m[13] ?? 0)) * invW,
      ((m[2] ?? 0) * ndcX + (m[6] ?? 0) * ndcY + (m[10] ?? 0) * z + (m[14] ?? 0)) * invW,
    ];
  };

  const nearPt = unproject(0);
  const farPt = unproject(1);
  return raycastStore(
    store,
    nearPt,
    [farPt[0] - nearPt[0], farPt[1] - nearPt[1], farPt[2] - nearPt[2]],
    resolveName,
  );
}
