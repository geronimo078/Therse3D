import { DirtyFlag, type WorldStore } from '../core/WorldStore';
import type { Engine, EnginePlugin } from '../core/Engine';
import type { EntityHandle } from '../entities/EntityHandle';
import { parseColor } from '../entities/Materials';
import { Sprite } from '../entities/Primitives';

export type ParticlePresetName =
  | 'fire'
  | 'smoke'
  | 'magic'
  | 'rain'
  | 'snow'
  | 'sparks'
  | 'explosion'
  | 'custom';

export type ParticleShape = 'soft-circle' | 'spark' | 'ring' | 'streak' | 'snowflake';
export type ParticleSpawnArea = 'point' | 'box' | 'sphere' | 'ring';
export type ParticleColorInput =
  | string
  | readonly [number, number, number]
  | readonly [number, number, number, number];

export interface ParticleEmitterOptions {
  id?: string;
  preset?: ParticlePresetName;
  position?: readonly [number, number, number];
  enabled?: boolean;
  /** Particles spawned per second (0 for burst-only emitters). */
  rate?: number;
  /** Immediate burst spawned upon creation. */
  initialBurst?: number;
  /** Spawn region shape around emitter position. */
  spawnArea?: ParticleSpawnArea;
  /** Half-extents for 'box' [x,y,z], radius for 'sphere'/'ring' (x = radius, y = vertical jitter). */
  spawnExtents?: readonly [number, number, number];
  /** Base initial velocity [vx, vy, vz]. */
  velocity?: readonly [number, number, number];
  /** Random per-axis velocity spread [sx, sy, sz]. */
  velocitySpread?: readonly [number, number, number];
  /** Outward radial speed from emitter center (great for explosions & magic rings). */
  radialSpeed?: number;
  /** Constant acceleration [ax, ay, az], e.g. [0, -9.8, 0] for rain or [0, 1.8, 0] for fire buoyancy. */
  gravity?: readonly [number, number, number];
  /** Exponential velocity damping per second (0 = none). */
  drag?: number;
  /** Sinusoidal flutter/turbulence strength (great for fire, smoke, snow, magic). */
  turbulence?: number;
  /** Angular swirl velocity (rad/s) around the emitter's vertical Y axis (great for magic spells). */
  swirlSpeed?: number;
  /** Particle lifetime in seconds [min, max] or single number. */
  lifetime?: number | readonly [number, number];
  /** Starting sprite size [min, max] or single number. */
  startSize?: number | readonly [number, number];
  /** Ending sprite size [min, max] or single number. */
  endSize?: number | readonly [number, number];
  /** Width-to-height ratio (e.g. 0.14 for elongated rain streaks, 1.0 for square sprites). */
  aspectRatio?: number;
  /** Starting RGBA color (supports hex string or [r,g,b(,a)], RGB > 1 boosts HDR/bloom glow). */
  startColor?: ParticleColorInput;
  /** Ending RGBA color. */
  endColor?: ParticleColorInput;
  /** Normalized age fraction (0..0.5) spent fading alpha in. */
  fadeIn?: number;
  /** Normalized age fraction (0..0.9) spent fading alpha out. */
  fadeOut?: number;
  /** Procedural particle texture shape. */
  shape?: ParticleShape;
  /** Optional custom registered texture ID (overrides `shape`). */
  textureId?: number;
}

export type ParticleEmitterPatch = Partial<Omit<ParticleEmitterOptions, 'id' | 'preset'>>;

export interface ParticleEmitterSummary {
  id: string;
  preset: ParticlePresetName;
  enabled: boolean;
  position: [number, number, number];
  rate: number;
  shape: ParticleShape;
  activeParticles: number;
}

export interface ParticleEmitterHandle {
  readonly id: string;
  readonly preset: ParticlePresetName;
  setPosition(x: number, y: number, z: number): this;
  getPosition(): [number, number, number];
  setEnabled(enabled: boolean): this;
  isEnabled(): boolean;
  setRate(rate: number): this;
  update(patch: ParticleEmitterPatch): this;
  burst(count?: number): number;
  remove(): boolean;
}

export interface SpriteParticlePluginOptions {
  /** Maximum simultaneous active particles across all emitters (default: 320). */
  maxParticles?: number;
  /** Resolution (width = height) of generated procedural particle textures (default: 64). */
  textureSize?: number;
  /** Automatically advance particle simulation on every engine tick (default: true). */
  autoUpdate?: boolean;
  /** Optional deterministic seed for reproducible particle motion. */
  seed?: number;
}

export interface SpriteParticleStats {
  activeParticles: number;
  pooledParticles: number;
  maxParticles: number;
  emitterCount: number;
  paused: boolean;
}

interface ResolvedEmitterConfig {
  id: string;
  preset: ParticlePresetName;
  enabled: boolean;
  posX: number;
  posY: number;
  posZ: number;
  rate: number;
  accumulator: number;
  spawnArea: ParticleSpawnArea;
  extX: number;
  extY: number;
  extZ: number;
  velX: number;
  velY: number;
  velZ: number;
  spreadX: number;
  spreadY: number;
  spreadZ: number;
  radialSpeed: number;
  gravX: number;
  gravY: number;
  gravZ: number;
  drag: number;
  turbulence: number;
  swirlSpeed: number;
  lifeMin: number;
  lifeMax: number;
  startSizeMin: number;
  startSizeMax: number;
  endSizeMin: number;
  endSizeMax: number;
  aspectRatio: number;
  startR: number;
  startG: number;
  startB: number;
  startA: number;
  endR: number;
  endG: number;
  endB: number;
  endA: number;
  fadeIn: number;
  fadeOut: number;
  shape: ParticleShape;
  customTextureId: number | null;
  activeCount: number;
  slotIndex: number;
}

export const PARTICLE_PRESETS: Record<Exclude<ParticlePresetName, 'custom'>, Omit<ParticleEmitterOptions, 'id' | 'preset'>> = {
  fire: {
    rate: 42,
    spawnArea: 'ring',
    spawnExtents: [0.22, 0.06, 0.22],
    velocity: [0, 1.35, 0],
    velocitySpread: [0.35, 0.45, 0.35],
    radialSpeed: -0.15,
    gravity: [0, 0.9, 0],
    drag: 0.65,
    turbulence: 0.65,
    swirlSpeed: 0.6,
    lifetime: [0.45, 0.95],
    startSize: [0.32, 0.5],
    endSize: [0.06, 0.14],
    aspectRatio: 0.85,
    startColor: [1.5, 0.92, 0.28, 0.95],
    endColor: [0.95, 0.12, 0.02, 0.0],
    fadeIn: 0.12,
    fadeOut: 0.65,
    shape: 'soft-circle',
  },
  smoke: {
    rate: 16,
    spawnArea: 'sphere',
    spawnExtents: [0.24, 0.12, 0.24],
    velocity: [0.08, 0.85, 0.04],
    velocitySpread: [0.25, 0.2, 0.25],
    radialSpeed: 0.1,
    gravity: [0.08, 0.22, 0],
    drag: 0.45,
    turbulence: 0.45,
    swirlSpeed: 0.25,
    lifetime: [1.6, 2.8],
    startSize: [0.35, 0.55],
    endSize: [1.15, 1.65],
    aspectRatio: 1.0,
    startColor: [0.35, 0.36, 0.4, 0.45],
    endColor: [0.18, 0.19, 0.22, 0.0],
    fadeIn: 0.2,
    fadeOut: 0.7,
    shape: 'soft-circle',
  },
  magic: {
    rate: 38,
    spawnArea: 'ring',
    spawnExtents: [0.65, 0.15, 0.65],
    velocity: [0, 0.85, 0],
    velocitySpread: [0.22, 0.35, 0.22],
    radialSpeed: -0.12,
    gravity: [0, 0.35, 0],
    drag: 0.35,
    turbulence: 0.55,
    swirlSpeed: 3.1,
    lifetime: [0.9, 1.8],
    startSize: [0.16, 0.32],
    endSize: [0.03, 0.09],
    aspectRatio: 1.0,
    startColor: [0.45, 0.95, 1.6, 0.95],
    endColor: [0.92, 0.28, 1.35, 0.0],
    fadeIn: 0.15,
    fadeOut: 0.6,
    shape: 'spark',
  },
  rain: {
    rate: 110,
    spawnArea: 'box',
    spawnExtents: [4.5, 0.4, 4.5],
    velocity: [-0.45, -9.5, 0.2],
    velocitySpread: [0.2, 1.4, 0.2],
    radialSpeed: 0,
    gravity: [0, -3.0, 0],
    drag: 0.0,
    turbulence: 0.0,
    swirlSpeed: 0.0,
    lifetime: [0.65, 1.05],
    startSize: [0.45, 0.65],
    endSize: [0.45, 0.65],
    aspectRatio: 0.08,
    startColor: [0.68, 0.84, 1.0, 0.62],
    endColor: [0.58, 0.76, 0.96, 0.18],
    fadeIn: 0.08,
    fadeOut: 0.25,
    shape: 'streak',
  },
  snow: {
    rate: 45,
    spawnArea: 'box',
    spawnExtents: [4.5, 0.4, 4.5],
    velocity: [0.15, -0.95, 0.08],
    velocitySpread: [0.28, 0.25, 0.28],
    radialSpeed: 0,
    gravity: [0, -0.05, 0],
    drag: 0.2,
    turbulence: 0.85,
    swirlSpeed: 0.0,
    lifetime: [3.2, 5.2],
    startSize: [0.12, 0.24],
    endSize: [0.1, 0.2],
    aspectRatio: 1.0,
    startColor: [0.96, 0.98, 1.0, 0.9],
    endColor: [0.88, 0.94, 1.0, 0.0],
    fadeIn: 0.12,
    fadeOut: 0.3,
    shape: 'snowflake',
  },
  sparks: {
    rate: 36,
    spawnArea: 'point',
    spawnExtents: [0.05, 0.05, 0.05],
    velocity: [0, 2.6, 0],
    velocitySpread: [1.5, 1.1, 1.5],
    radialSpeed: 0.8,
    gravity: [0, -7.2, 0],
    drag: 0.35,
    turbulence: 0.15,
    swirlSpeed: 0.0,
    lifetime: [0.45, 0.95],
    startSize: [0.16, 0.26],
    endSize: [0.03, 0.06],
    aspectRatio: 0.55,
    startColor: [1.7, 1.35, 0.45, 1.0],
    endColor: [1.1, 0.22, 0.04, 0.0],
    fadeIn: 0.05,
    fadeOut: 0.55,
    shape: 'spark',
  },
  explosion: {
    rate: 0,
    initialBurst: 48,
    spawnArea: 'sphere',
    spawnExtents: [0.15, 0.15, 0.15],
    velocity: [0, 0.6, 0],
    velocitySpread: [0.7, 0.7, 0.7],
    radialSpeed: 4.2,
    gravity: [0, -3.2, 0],
    drag: 1.85,
    turbulence: 0.5,
    swirlSpeed: 0.0,
    lifetime: [0.45, 1.1],
    startSize: [0.35, 0.65],
    endSize: [0.08, 0.18],
    aspectRatio: 1.0,
    startColor: [1.8, 1.1, 0.35, 1.0],
    endColor: [0.45, 0.1, 0.05, 0.0],
    fadeIn: 0.04,
    fadeOut: 0.7,
    shape: 'soft-circle',
  },
};

function sanitizeNumber(value: unknown, fallback: number, min = -1e6, max = 1e6): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(max, value));
}

function sanitizeRange(
  value: number | readonly [number, number] | undefined,
  fallbackMin: number,
  fallbackMax: number,
  minBound = 0.001,
  maxBound = 1000,
): [number, number] {
  if (typeof value === 'number') {
    const v = sanitizeNumber(value, fallbackMin, minBound, maxBound);
    return [v, v];
  }
  if (Array.isArray(value) && value.length >= 2) {
    const a = sanitizeNumber(value[0], fallbackMin, minBound, maxBound);
    const b = sanitizeNumber(value[1], fallbackMax, minBound, maxBound);
    return a <= b ? [a, b] : [b, a];
  }
  return [fallbackMin, fallbackMax];
}

function sanitizeVec3(
  value: readonly [number, number, number] | undefined,
  fx: number,
  fy: number,
  fz: number,
): [number, number, number] {
  if (!value || value.length < 3) return [fx, fy, fz];
  return [
    sanitizeNumber(value[0], fx),
    sanitizeNumber(value[1], fy),
    sanitizeNumber(value[2], fz),
  ];
}

function resolveRgba(
  input: ParticleColorInput | undefined,
  fallback: readonly [number, number, number, number],
): [number, number, number, number] {
  if (typeof input === 'string') {
    const parsed = parseColor(input);
    return [parsed[0], parsed[1], parsed[2], parsed[3]];
  }
  if (Array.isArray(input) && input.length >= 3) {
    return [
      sanitizeNumber(input[0], fallback[0], 0, 8),
      sanitizeNumber(input[1], fallback[1], 0, 8),
      sanitizeNumber(input[2], fallback[2], 0, 8),
      sanitizeNumber(input[3] ?? fallback[3], fallback[3], 0, 1),
    ];
  }
  return [fallback[0], fallback[1], fallback[2], fallback[3]];
}

/**
 * Generates a procedural RGBA pixel buffer for built-in particle sprite shapes.
 * Works identically in browser and headless Node/test environments (zero DOM dependencies).
 */
export function generateParticleTexturePixels(shape: ParticleShape, size = 64): Uint8Array {
  const dim = Math.max(8, Math.min(256, Math.floor(Number.isFinite(size) ? size : 64)));
  const pixels = new Uint8Array(dim * dim * 4);
  const invHalf = 2 / dim;

  for (let y = 0; y < dim; y++) {
    const ny = (y + 0.5) * invHalf - 1;
    for (let x = 0; x < dim; x++) {
      const nx = (x + 0.5) * invHalf - 1;
      const r = Math.hypot(nx, ny);
      let intensity = 0;
      let alpha = 0;

      if (shape === 'soft-circle') {
        const t = Math.max(0, 1 - r);
        alpha = t * t * (3 - 2 * t);
        intensity = 0.72 + 0.28 * Math.exp(-r * r * 4.0);
      } else if (shape === 'spark') {
        const core = Math.exp(-r * r * 9.5);
        const crossH = Math.exp(-Math.abs(ny) * 14.0) * Math.max(0, 1 - Math.abs(nx));
        const crossV = Math.exp(-Math.abs(nx) * 14.0) * Math.max(0, 1 - Math.abs(ny));
        const diag1 = Math.exp(-Math.abs(nx + ny) * 16.0) * Math.max(0, 1 - r * 1.25) * 0.45;
        const diag2 = Math.exp(-Math.abs(nx - ny) * 16.0) * Math.max(0, 1 - r * 1.25) * 0.45;
        alpha = Math.min(1, core + (crossH + crossV) * 0.72 + diag1 + diag2);
        intensity = Math.min(1, 0.7 + core * 0.3);
      } else if (shape === 'ring') {
        const ringDist = Math.abs(r - 0.64);
        const band = Math.exp(-ringDist * ringDist * 45.0);
        const innerGlow = Math.exp(-r * r * 5.0) * 0.22;
        alpha = Math.min(1, (band + innerGlow) * (r < 0.98 ? 1 : 0));
        intensity = 0.85 + 0.15 * band;
      } else if (shape === 'streak') {
        const xFalloff = Math.exp(-nx * nx * 10.0);
        const yTaper = Math.max(0, 1 - Math.abs(ny));
        const ySmooth = Math.pow(yTaper, 0.65);
        alpha = xFalloff * ySmooth;
        intensity = 0.75 + 0.25 * xFalloff;
      } else if (shape === 'snowflake') {
        const angle = Math.atan2(ny, nx);
        const spoke = Math.pow(0.5 + 0.5 * Math.cos(angle * 6), 6);
        const radial = Math.max(0, 1 - r);
        const core = Math.exp(-r * r * 10.0);
        alpha = Math.min(1, core * 0.9 + radial * radial * (0.28 + 0.72 * spoke));
        intensity = 0.9 + 0.1 * core;
      }

      const idx = (y * dim + x) * 4;
      const c = Math.max(0, Math.min(255, Math.round(intensity * 255)));
      const a = Math.max(0, Math.min(255, Math.round(alpha * 255)));
      pixels[idx] = c;
      pixels[idx + 1] = c;
      pixels[idx + 2] = c;
      pixels[idx + 3] = a;
    }
  }

  return pixels;
}

/**
 * Optional sprite-based particle system manager for therse3d.
 * Uses zero-allocation SoA TypedArray pools and instanced billboard sprites.
 */
export class SpriteParticleManager {
  readonly maxParticles: number;
  readonly textureSize: number;

  private readonly engine: Engine;
  private readonly store: WorldStore | null;
  private readonly emitters = new Map<string, ResolvedEmitterConfig>();
  private readonly emitterSlots: Array<ResolvedEmitterConfig | null> = [];
  private readonly shapeTextureIds = new Map<ParticleShape, number>();

  // Pooled Sprite entities (lazily grown up to maxParticles, never allocated per frame once warm)
  private readonly entities: Array<EntityHandle | null>;
  private readonly entityIds: Int32Array;
  private readonly entityTextureIds: Int32Array;

  // Particle SoA simulation arrays
  private readonly active: Uint8Array;
  private readonly ownerSlot: Int32Array;
  private readonly age: Float32Array;
  private readonly lifetime: Float32Array;
  private readonly posX: Float32Array;
  private readonly posY: Float32Array;
  private readonly posZ: Float32Array;
  private readonly velX: Float32Array;
  private readonly velY: Float32Array;
  private readonly velZ: Float32Array;
  private readonly gravX: Float32Array;
  private readonly gravY: Float32Array;
  private readonly gravZ: Float32Array;
  private readonly drag: Float32Array;
  private readonly turbulence: Float32Array;
  private readonly swirlSpeed: Float32Array;
  private readonly phase: Float32Array;
  private readonly startSizeX: Float32Array;
  private readonly startSizeY: Float32Array;
  private readonly endSizeX: Float32Array;
  private readonly endSizeY: Float32Array;
  private readonly startR: Float32Array;
  private readonly startG: Float32Array;
  private readonly startB: Float32Array;
  private readonly startA: Float32Array;
  private readonly endR: Float32Array;
  private readonly endG: Float32Array;
  private readonly endB: Float32Array;
  private readonly endA: Float32Array;
  private readonly fadeIn: Float32Array;
  private readonly fadeOut: Float32Array;

  // Free-list stack for O(1) particle slot acquire/release
  private readonly freeStack: Int32Array;
  private freeTop: number;
  private activeCount = 0;
  private pooledCount = 0;
  private nextEmitterSeq = 1;
  private rngState: number;
  private paused = false;
  private disposed = false;
  private unsubTick: (() => void) | null = null;

  constructor(engine: Engine, options: SpriteParticlePluginOptions = {}) {
    this.engine = engine;
    this.store = (engine as unknown as { store?: WorldStore }).store ?? null;
    this.maxParticles = Math.max(1, Math.min(8192, Math.floor(sanitizeNumber(options.maxParticles, 320, 1, 8192))));
    this.textureSize = Math.max(16, Math.min(256, Math.floor(sanitizeNumber(options.textureSize, 64, 16, 256))));
    this.rngState = Math.floor(sanitizeNumber(options.seed, 0x9e3779b9, 1, 0x7fffffff)) >>> 0;

    const n = this.maxParticles;
    this.entities = new Array<EntityHandle | null>(n).fill(null);
    this.entityIds = new Int32Array(n).fill(-1);
    this.entityTextureIds = new Int32Array(n).fill(-1);

    this.active = new Uint8Array(n);
    this.ownerSlot = new Int32Array(n).fill(-1);
    this.age = new Float32Array(n);
    this.lifetime = new Float32Array(n);
    this.posX = new Float32Array(n);
    this.posY = new Float32Array(n);
    this.posZ = new Float32Array(n);
    this.velX = new Float32Array(n);
    this.velY = new Float32Array(n);
    this.velZ = new Float32Array(n);
    this.gravX = new Float32Array(n);
    this.gravY = new Float32Array(n);
    this.gravZ = new Float32Array(n);
    this.drag = new Float32Array(n);
    this.turbulence = new Float32Array(n);
    this.swirlSpeed = new Float32Array(n);
    this.phase = new Float32Array(n);
    this.startSizeX = new Float32Array(n);
    this.startSizeY = new Float32Array(n);
    this.endSizeX = new Float32Array(n);
    this.endSizeY = new Float32Array(n);
    this.startR = new Float32Array(n);
    this.startG = new Float32Array(n);
    this.startB = new Float32Array(n);
    this.startA = new Float32Array(n);
    this.endR = new Float32Array(n);
    this.endG = new Float32Array(n);
    this.endB = new Float32Array(n);
    this.endA = new Float32Array(n);
    this.fadeIn = new Float32Array(n);
    this.fadeOut = new Float32Array(n);

    this.freeStack = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      this.freeStack[i] = n - 1 - i;
    }
    this.freeTop = n;

    if (options.autoUpdate !== false && typeof engine.onTick === 'function') {
      this.unsubTick = engine.onTick((dt) => {
        this.update(dt);
      });
    }
  }

  private nextRandom(): number {
    // Mulberry32 / Xorshift32 fast deterministic PRNG (zero allocations)
    let x = this.rngState;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.rngState = x >>> 0;
    return (this.rngState & 0x0fffffff) / 0x10000000;
  }

  private randRange(min: number, max: number): number {
    return min + (max - min) * this.nextRandom();
  }

  private ensureShapeTexture(shape: ParticleShape): number {
    const cached = this.shapeTextureIds.get(shape);
    if (cached !== undefined) return cached;
    if (typeof this.engine.registerTexture !== 'function') {
      this.shapeTextureIds.set(shape, 0);
      return 0;
    }
    const pixels = generateParticleTexturePixels(shape, this.textureSize);
    const texId = this.engine.registerTexture(pixels, this.textureSize, this.textureSize);
    this.shapeTextureIds.set(shape, texId);
    return texId;
  }

  private resolveEmitterConfig(id: string, slotIndex: number, options: ParticleEmitterOptions): ResolvedEmitterConfig {
    const rawPreset = options.preset ?? 'fire';
    const preset: ParticlePresetName =
      rawPreset === 'custom' || rawPreset in PARTICLE_PRESETS ? rawPreset : 'fire';
    const base = preset === 'custom' ? {} : PARTICLE_PRESETS[preset];
    const merged: ParticleEmitterOptions = { ...base, ...options };

    const [posX, posY, posZ] = sanitizeVec3(merged.position, 0, 0, 0);
    const [extX, extY, extZ] = sanitizeVec3(merged.spawnExtents, 0.1, 0.1, 0.1);
    const [velX, velY, velZ] = sanitizeVec3(merged.velocity, 0, 1, 0);
    const [spreadX, spreadY, spreadZ] = sanitizeVec3(merged.velocitySpread, 0.25, 0.25, 0.25);
    const [gravX, gravY, gravZ] = sanitizeVec3(merged.gravity, 0, 0, 0);
    const [lifeMin, lifeMax] = sanitizeRange(merged.lifetime, 0.6, 1.2, 0.05, 60);
    const [startSizeMin, startSizeMax] = sanitizeRange(merged.startSize, 0.2, 0.35, 0.005, 50);
    const [endSizeMin, endSizeMax] = sanitizeRange(merged.endSize, 0.05, 0.12, 0.001, 50);
    const [startR, startG, startB, startA] = resolveRgba(merged.startColor, [1, 1, 1, 1]);
    const [endR, endG, endB, endA] = resolveRgba(merged.endColor, [1, 1, 1, 0]);

    const shape: ParticleShape =
      merged.shape === 'spark' ||
      merged.shape === 'ring' ||
      merged.shape === 'streak' ||
      merged.shape === 'snowflake'
        ? merged.shape
        : 'soft-circle';

    const spawnArea: ParticleSpawnArea =
      merged.spawnArea === 'box' || merged.spawnArea === 'sphere' || merged.spawnArea === 'ring'
        ? merged.spawnArea
        : 'point';

    return {
      id,
      preset,
      enabled: merged.enabled !== false,
      posX,
      posY,
      posZ,
      rate: sanitizeNumber(merged.rate, 24, 0, 2000),
      accumulator: 0,
      spawnArea,
      extX: Math.abs(extX),
      extY: Math.abs(extY),
      extZ: Math.abs(extZ),
      velX,
      velY,
      velZ,
      spreadX: Math.abs(spreadX),
      spreadY: Math.abs(spreadY),
      spreadZ: Math.abs(spreadZ),
      radialSpeed: sanitizeNumber(merged.radialSpeed, 0, -100, 100),
      gravX,
      gravY,
      gravZ,
      drag: sanitizeNumber(merged.drag, 0, 0, 20),
      turbulence: sanitizeNumber(merged.turbulence, 0, 0, 20),
      swirlSpeed: sanitizeNumber(merged.swirlSpeed, 0, -30, 30),
      lifeMin,
      lifeMax,
      startSizeMin,
      startSizeMax,
      endSizeMin,
      endSizeMax,
      aspectRatio: sanitizeNumber(merged.aspectRatio, 1.0, 0.02, 20),
      startR,
      startG,
      startB,
      startA,
      endR,
      endG,
      endB,
      endA,
      fadeIn: sanitizeNumber(merged.fadeIn, 0.1, 0, 0.5),
      fadeOut: sanitizeNumber(merged.fadeOut, 0.5, 0, 0.95),
      shape,
      customTextureId:
        typeof merged.textureId === 'number' && Number.isFinite(merged.textureId) && merged.textureId >= 0
          ? Math.floor(merged.textureId)
          : null,
      activeCount: 0,
      slotIndex,
    };
  }

  createEmitter(options: ParticleEmitterOptions = {}): ParticleEmitterHandle {
    if (this.disposed) {
      throw new Error('SpriteParticleManager has been disposed');
    }
    const id =
      typeof options.id === 'string' && options.id.trim().length > 0
        ? options.id.trim()
        : `emitter-${this.nextEmitterSeq++}`;

    if (this.emitters.has(id)) {
      this.removeEmitter(id);
    }

    let slotIndex = this.emitterSlots.indexOf(null);
    if (slotIndex === -1) {
      slotIndex = this.emitterSlots.length;
      this.emitterSlots.push(null);
    }

    const cfg = this.resolveEmitterConfig(id, slotIndex, options);
    this.emitters.set(id, cfg);
    this.emitterSlots[slotIndex] = cfg;

    const initialBurst =
      options.initialBurst !== undefined
        ? Math.floor(sanitizeNumber(options.initialBurst, 0, 0, this.maxParticles))
        : cfg.preset !== 'custom' && PARTICLE_PRESETS[cfg.preset].initialBurst
          ? Math.floor(PARTICLE_PRESETS[cfg.preset].initialBurst!)
          : 0;

    if (initialBurst > 0) {
      this.spawnFromEmitter(cfg, initialBurst);
    }

    return this.createHandle(id, cfg.preset);
  }

  private createHandle(id: string, preset: ParticlePresetName): ParticleEmitterHandle {
    const handle: ParticleEmitterHandle = {
      id,
      preset,
      setPosition: (x, y, z) => {
        this.setEmitterPosition(id, [x, y, z]);
        return handle;
      },
      getPosition: () => {
        const e = this.emitters.get(id);
        return e ? [e.posX, e.posY, e.posZ] : [0, 0, 0];
      },
      setEnabled: (enabled) => {
        this.setEmitterEnabled(id, enabled);
        return handle;
      },
      isEnabled: () => this.emitters.get(id)?.enabled ?? false,
      setRate: (rate) => {
        this.updateEmitter(id, { rate });
        return handle;
      },
      update: (patch) => {
        this.updateEmitter(id, patch);
        return handle;
      },
      burst: (count) => this.burst(id, count),
      remove: () => this.removeEmitter(id),
    };
    return handle;
  }

  getEmitter(id: string): ParticleEmitterHandle | undefined {
    const cfg = this.emitters.get(id);
    return cfg ? this.createHandle(cfg.id, cfg.preset) : undefined;
  }

  listEmitters(): ParticleEmitterSummary[] {
    const out: ParticleEmitterSummary[] = [];
    for (const cfg of this.emitters.values()) {
      out.push({
        id: cfg.id,
        preset: cfg.preset,
        enabled: cfg.enabled,
        position: [cfg.posX, cfg.posY, cfg.posZ],
        rate: cfg.rate,
        shape: cfg.shape,
        activeParticles: cfg.activeCount,
      });
    }
    return out;
  }

  setEmitterPosition(id: string, position: readonly [number, number, number]): boolean {
    const cfg = this.emitters.get(id);
    if (!cfg) return false;
    cfg.posX = sanitizeNumber(position[0], cfg.posX);
    cfg.posY = sanitizeNumber(position[1], cfg.posY);
    cfg.posZ = sanitizeNumber(position[2], cfg.posZ);
    return true;
  }

  setEmitterEnabled(id: string, enabled: boolean): boolean {
    const cfg = this.emitters.get(id);
    if (!cfg) return false;
    cfg.enabled = Boolean(enabled);
    return true;
  }

  updateEmitter(id: string, patch: ParticleEmitterPatch): boolean {
    const cfg = this.emitters.get(id);
    if (!cfg) return false;

    if (patch.position) {
      [cfg.posX, cfg.posY, cfg.posZ] = sanitizeVec3(patch.position, cfg.posX, cfg.posY, cfg.posZ);
    }
    if (patch.enabled !== undefined) cfg.enabled = Boolean(patch.enabled);
    if (patch.rate !== undefined) cfg.rate = sanitizeNumber(patch.rate, cfg.rate, 0, 2000);
    if (patch.spawnArea !== undefined) cfg.spawnArea = patch.spawnArea;
    if (patch.spawnExtents) {
      const [ex, ey, ez] = sanitizeVec3(patch.spawnExtents, cfg.extX, cfg.extY, cfg.extZ);
      cfg.extX = Math.abs(ex);
      cfg.extY = Math.abs(ey);
      cfg.extZ = Math.abs(ez);
    }
    if (patch.velocity) {
      [cfg.velX, cfg.velY, cfg.velZ] = sanitizeVec3(patch.velocity, cfg.velX, cfg.velY, cfg.velZ);
    }
    if (patch.velocitySpread) {
      const [sx, sy, sz] = sanitizeVec3(patch.velocitySpread, cfg.spreadX, cfg.spreadY, cfg.spreadZ);
      cfg.spreadX = Math.abs(sx);
      cfg.spreadY = Math.abs(sy);
      cfg.spreadZ = Math.abs(sz);
    }
    if (patch.radialSpeed !== undefined) cfg.radialSpeed = sanitizeNumber(patch.radialSpeed, cfg.radialSpeed, -100, 100);
    if (patch.gravity) {
      [cfg.gravX, cfg.gravY, cfg.gravZ] = sanitizeVec3(patch.gravity, cfg.gravX, cfg.gravY, cfg.gravZ);
    }
    if (patch.drag !== undefined) cfg.drag = sanitizeNumber(patch.drag, cfg.drag, 0, 20);
    if (patch.turbulence !== undefined) cfg.turbulence = sanitizeNumber(patch.turbulence, cfg.turbulence, 0, 20);
    if (patch.swirlSpeed !== undefined) cfg.swirlSpeed = sanitizeNumber(patch.swirlSpeed, cfg.swirlSpeed, -30, 30);
    if (patch.lifetime !== undefined) {
      [cfg.lifeMin, cfg.lifeMax] = sanitizeRange(patch.lifetime, cfg.lifeMin, cfg.lifeMax, 0.05, 60);
    }
    if (patch.startSize !== undefined) {
      [cfg.startSizeMin, cfg.startSizeMax] = sanitizeRange(patch.startSize, cfg.startSizeMin, cfg.startSizeMax, 0.005, 50);
    }
    if (patch.endSize !== undefined) {
      [cfg.endSizeMin, cfg.endSizeMax] = sanitizeRange(patch.endSize, cfg.endSizeMin, cfg.endSizeMax, 0.001, 50);
    }
    if (patch.aspectRatio !== undefined) cfg.aspectRatio = sanitizeNumber(patch.aspectRatio, cfg.aspectRatio, 0.02, 20);
    if (patch.startColor !== undefined) {
      [cfg.startR, cfg.startG, cfg.startB, cfg.startA] = resolveRgba(patch.startColor, [
        cfg.startR,
        cfg.startG,
        cfg.startB,
        cfg.startA,
      ]);
    }
    if (patch.endColor !== undefined) {
      [cfg.endR, cfg.endG, cfg.endB, cfg.endA] = resolveRgba(patch.endColor, [
        cfg.endR,
        cfg.endG,
        cfg.endB,
        cfg.endA,
      ]);
    }
    if (patch.fadeIn !== undefined) cfg.fadeIn = sanitizeNumber(patch.fadeIn, cfg.fadeIn, 0, 0.5);
    if (patch.fadeOut !== undefined) cfg.fadeOut = sanitizeNumber(patch.fadeOut, cfg.fadeOut, 0, 0.95);
    if (patch.shape !== undefined) cfg.shape = patch.shape;
    if (patch.textureId !== undefined) {
      cfg.customTextureId =
        Number.isFinite(patch.textureId) && patch.textureId >= 0 ? Math.floor(patch.textureId) : null;
    }
    return true;
  }

  removeEmitter(id: string): boolean {
    const cfg = this.emitters.get(id);
    if (!cfg) return false;
    this.emitters.delete(id);
    this.emitterSlots[cfg.slotIndex] = null;

    // Recycle all particles currently belonging to this emitter
    for (let i = 0; i < this.maxParticles; i++) {
      if (this.active[i] === 1 && this.ownerSlot[i] === cfg.slotIndex) {
        this.releaseParticle(i);
      }
    }
    return true;
  }

  /**
   * Triggers an immediate burst of particles from an existing emitter ID or an ad-hoc preset/options object.
   */
  burst(emitterIdOrOptions: string | ParticleEmitterOptions, count = 24): number {
    if (this.disposed) return 0;
    const n = Math.floor(sanitizeNumber(count, 24, 0, this.maxParticles));
    if (n <= 0) return 0;

    if (typeof emitterIdOrOptions === 'string') {
      const cfg = this.emitters.get(emitterIdOrOptions);
      if (!cfg) return 0;
      return this.spawnFromEmitter(cfg, n);
    }

    const tempCfg = this.resolveEmitterConfig('__burst__', -1, emitterIdOrOptions);
    return this.spawnFromEmitter(tempCfg, n);
  }

  pause(): void {
    this.paused = true;
  }

  resume(): void {
    this.paused = false;
  }

  clear(): void {
    for (let i = 0; i < this.maxParticles; i++) {
      if (this.active[i] === 1) {
        this.releaseParticle(i);
      }
    }
    for (const cfg of this.emitters.values()) {
      cfg.accumulator = 0;
      cfg.activeCount = 0;
    }
  }

  getStats(): SpriteParticleStats {
    return {
      activeParticles: this.activeCount,
      pooledParticles: this.pooledCount,
      maxParticles: this.maxParticles,
      emitterCount: this.emitters.size,
      paused: this.paused,
    };
  }

  private ensureParticleEntity(slot: number, textureId: number): number {
    let entity = this.entities[slot];
    if (!entity) {
      const spriteOpts = {
        id: `therse3d-particle-${slot}`,
        position: [0, 0, 0] as [number, number, number],
        scale: [0.1, 0.1, 1] as [number, number, number],
        color: [1, 1, 1, 0] as [number, number, number, number],
        visible: false,
        ...(textureId > 0 ? { texture: textureId } : {}),
      };
      entity = this.engine.add(Sprite(spriteOpts));
      this.entities[slot] = entity;
      this.entityIds[slot] = entity.id;
      this.entityTextureIds[slot] = textureId;
      this.pooledCount++;
      if (this.store && textureId > 0) {
        this.store.setTextureId(entity.id, textureId);
      }
      return entity.id;
    }

    const eid = this.entityIds[slot]!;
    if (this.entityTextureIds[slot] !== textureId) {
      this.entityTextureIds[slot] = textureId;
      if (this.store) {
        this.store.setTextureId(eid, textureId);
      }
    }
    return eid;
  }

  private writeParticleToStore(
    slot: number,
    px: number,
    py: number,
    pz: number,
    sx: number,
    sy: number,
    r: number,
    g: number,
    b: number,
    a: number,
    visible: boolean,
  ): void {
    const eid = this.entityIds[slot]!;
    const store = this.store;
    if (store) {
      const pIdx = eid * 3;
      const cIdx = eid * 4;
      store.positions[pIdx] = px;
      store.positions[pIdx + 1] = py;
      store.positions[pIdx + 2] = pz;
      store.scales[pIdx] = sx;
      store.scales[pIdx + 1] = sy;
      store.scales[pIdx + 2] = 1;
      store.colors[cIdx] = r;
      store.colors[cIdx + 1] = g;
      store.colors[cIdx + 2] = b;
      store.colors[cIdx + 3] = a;
      store.visible[eid] = visible ? 1 : 0;
      store.markDirty(eid, DirtyFlag.Transform | DirtyFlag.Bounds | DirtyFlag.Material);
      return;
    }

    const handle = this.entities[slot];
    if (!handle) return;
    handle.visible = visible;
    if (typeof handle.setPosition === 'function') {
      handle.setPosition(px, py, pz);
    }
    if (typeof handle.setScale === 'function') {
      handle.setScale(sx, sy, 1);
    }
    handle.color = [r, g, b, a];
  }

  private releaseParticle(slot: number): void {
    if (this.active[slot] === 0) return;
    this.active[slot] = 0;
    const owner = this.ownerSlot[slot]!;
    if (owner >= 0 && owner < this.emitterSlots.length) {
      const cfg = this.emitterSlots[owner];
      if (cfg && cfg.activeCount > 0) cfg.activeCount--;
    }
    this.ownerSlot[slot] = -1;
    this.activeCount--;
    this.freeStack[this.freeTop++] = slot;

    const eid = this.entityIds[slot]!;
    if (eid >= 0) {
      if (this.store) {
        this.store.visible[eid] = 0;
      } else {
        const handle = this.entities[slot];
        if (handle) handle.visible = false;
      }
    }
  }

  private spawnFromEmitter(cfg: ResolvedEmitterConfig, count: number): number {
    const texId = cfg.customTextureId !== null ? cfg.customTextureId : this.ensureShapeTexture(cfg.shape);
    let spawned = 0;

    for (let k = 0; k < count; k++) {
      if (this.freeTop <= 0) break;
      const slot = this.freeStack[--this.freeTop]!;
      this.ensureParticleEntity(slot, texId);

      // Compute spawn offset within spawnArea
      let ox = 0;
      let oy = 0;
      let oz = 0;
      if (cfg.spawnArea === 'box') {
        ox = (this.nextRandom() * 2 - 1) * cfg.extX;
        oy = (this.nextRandom() * 2 - 1) * cfg.extY;
        oz = (this.nextRandom() * 2 - 1) * cfg.extZ;
      } else if (cfg.spawnArea === 'ring') {
        const ang = this.nextRandom() * Math.PI * 2;
        const rad = cfg.extX * (0.75 + 0.25 * this.nextRandom());
        ox = Math.cos(ang) * rad;
        oy = (this.nextRandom() * 2 - 1) * cfg.extY;
        oz = Math.sin(ang) * rad;
      } else if (cfg.spawnArea === 'sphere') {
        const u = this.nextRandom() * 2 - 1;
        const theta = this.nextRandom() * Math.PI * 2;
        const rr = Math.cbrt(this.nextRandom());
        const sinPhi = Math.sqrt(Math.max(0, 1 - u * u));
        ox = rr * sinPhi * Math.cos(theta) * cfg.extX;
        oy = rr * u * cfg.extY;
        oz = rr * sinPhi * Math.sin(theta) * cfg.extZ;
      }

      const px = cfg.posX + ox;
      const py = cfg.posY + oy;
      const pz = cfg.posZ + oz;

      // Radial velocity direction
      let rx = ox;
      let ry = oy;
      let rz = oz;
      let rLen = Math.hypot(rx, ry, rz);
      if (rLen < 1e-4 && cfg.radialSpeed !== 0) {
        const u = this.nextRandom() * 2 - 1;
        const theta = this.nextRandom() * Math.PI * 2;
        const sinPhi = Math.sqrt(Math.max(0, 1 - u * u));
        rx = sinPhi * Math.cos(theta);
        ry = u;
        rz = sinPhi * Math.sin(theta);
        rLen = 1;
      }
      const rScale = rLen > 1e-4 ? cfg.radialSpeed / rLen : 0;

      const vx = cfg.velX + (this.nextRandom() * 2 - 1) * cfg.spreadX + rx * rScale;
      const vy = cfg.velY + (this.nextRandom() * 2 - 1) * cfg.spreadY + ry * rScale;
      const vz = cfg.velZ + (this.nextRandom() * 2 - 1) * cfg.spreadZ + rz * rScale;

      const sSize = this.randRange(cfg.startSizeMin, cfg.startSizeMax);
      const eSize = this.randRange(cfg.endSizeMin, cfg.endSizeMax);
      const aspect = cfg.aspectRatio;

      this.active[slot] = 1;
      this.ownerSlot[slot] = cfg.slotIndex;
      this.age[slot] = 0;
      this.lifetime[slot] = this.randRange(cfg.lifeMin, cfg.lifeMax);
      this.posX[slot] = px;
      this.posY[slot] = py;
      this.posZ[slot] = pz;
      this.velX[slot] = vx;
      this.velY[slot] = vy;
      this.velZ[slot] = vz;
      this.gravX[slot] = cfg.gravX;
      this.gravY[slot] = cfg.gravY;
      this.gravZ[slot] = cfg.gravZ;
      this.drag[slot] = cfg.drag;
      this.turbulence[slot] = cfg.turbulence;
      this.swirlSpeed[slot] = cfg.swirlSpeed;
      this.phase[slot] = this.nextRandom() * Math.PI * 2;
      this.startSizeX[slot] = sSize * aspect;
      this.startSizeY[slot] = sSize;
      this.endSizeX[slot] = eSize * aspect;
      this.endSizeY[slot] = eSize;
      this.startR[slot] = cfg.startR;
      this.startG[slot] = cfg.startG;
      this.startB[slot] = cfg.startB;
      this.startA[slot] = cfg.startA;
      this.endR[slot] = cfg.endR;
      this.endG[slot] = cfg.endG;
      this.endB[slot] = cfg.endB;
      this.endA[slot] = cfg.endA;
      this.fadeIn[slot] = cfg.fadeIn;
      this.fadeOut[slot] = cfg.fadeOut;

      this.activeCount++;
      if (cfg.slotIndex >= 0) cfg.activeCount++;

      const initialAlpha = cfg.fadeIn > 0.001 ? 0 : cfg.startA;
      this.writeParticleToStore(
        slot,
        px,
        py,
        pz,
        sSize * aspect,
        sSize,
        cfg.startR,
        cfg.startG,
        cfg.startB,
        initialAlpha,
        true,
      );
      spawned++;
    }

    return spawned;
  }

  /**
   * Advances all emitters and active particles by `dt` seconds.
   * Zero heap allocations per call.
   */
  update(dt: number): void {
    if (this.disposed || this.paused) return;
    if (!Number.isFinite(dt) || dt <= 0) return;
    const clampedDt = Math.min(dt, 0.1);

    // 1. Spawn new particles from continuous emitters
    for (const cfg of this.emitters.values()) {
      if (!cfg.enabled || cfg.rate <= 0) continue;
      cfg.accumulator += cfg.rate * clampedDt;
      const toSpawn = Math.floor(cfg.accumulator);
      if (toSpawn > 0) {
        cfg.accumulator -= toSpawn;
        this.spawnFromEmitter(cfg, toSpawn);
      }
    }

    // 2. Integrate active particles (zero-allocation flat loop)
    const maxP = this.maxParticles;
    for (let i = 0; i < maxP; i++) {
      if (this.active[i] === 0) continue;

      const nextAge = this.age[i]! + clampedDt;
      const life = this.lifetime[i]!;
      if (nextAge >= life) {
        this.releaseParticle(i);
        continue;
      }
      this.age[i] = nextAge;
      const t = nextAge / life;

      // Velocity integration + gravity + exponential drag
      let vx = this.velX[i]! + this.gravX[i]! * clampedDt;
      let vy = this.velY[i]! + this.gravY[i]! * clampedDt;
      let vz = this.velZ[i]! + this.gravZ[i]! * clampedDt;

      const d = this.drag[i]!;
      if (d > 0) {
        const damp = Math.exp(-d * clampedDt);
        vx *= damp;
        vy *= damp;
        vz *= damp;
      }

      // Sinusoidal flutter/turbulence
      const turb = this.turbulence[i]!;
      let tx = 0;
      let tz = 0;
      if (turb > 0) {
        const ph = this.phase[i]! + nextAge * 5.2;
        tx = Math.sin(ph) * turb;
        tz = Math.cos(ph * 0.83) * turb;
      }

      let px = this.posX[i]! + (vx + tx) * clampedDt;
      let py = this.posY[i]! + vy * clampedDt;
      let pz = this.posZ[i]! + (vz + tz) * clampedDt;

      // Swirl around owning emitter's Y axis
      const swirl = this.swirlSpeed[i]!;
      if (swirl !== 0) {
        const owner = this.ownerSlot[i]!;
        const cfg = owner >= 0 ? this.emitterSlots[owner] : null;
        const cx = cfg ? cfg.posX : 0;
        const cz = cfg ? cfg.posZ : 0;
        const dx = px - cx;
        const dz = pz - cz;
        const ang = swirl * clampedDt;
        const ca = Math.cos(ang);
        const sa = Math.sin(ang);
        px = cx + dx * ca - dz * sa;
        pz = cz + dx * sa + dz * ca;
      }

      this.velX[i] = vx;
      this.velY[i] = vy;
      this.velZ[i] = vz;
      this.posX[i] = px;
      this.posY[i] = py;
      this.posZ[i] = pz;

      // Size & color interpolation
      const sx = this.startSizeX[i]! + (this.endSizeX[i]! - this.startSizeX[i]!) * t;
      const sy = this.startSizeY[i]! + (this.endSizeY[i]! - this.startSizeY[i]!) * t;
      const r = this.startR[i]! + (this.endR[i]! - this.startR[i]!) * t;
      const g = this.startG[i]! + (this.endG[i]! - this.startG[i]!) * t;
      const b = this.startB[i]! + (this.endB[i]! - this.startB[i]!) * t;
      let a = this.startA[i]! + (this.endA[i]! - this.startA[i]!) * t;

      const fi = this.fadeIn[i]!;
      if (fi > 0.001 && t < fi) {
        a *= t / fi;
      }
      const fo = this.fadeOut[i]!;
      if (fo > 0.001 && t > 1 - fo) {
        const outT = (1 - t) / fo;
        a *= outT > 0 ? outT : 0;
      }

      this.writeParticleToStore(i, px, py, pz, sx, sy, r, g, b, a, a > 0.01);
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.unsubTick) {
      this.unsubTick();
      this.unsubTick = null;
    }
    this.clear();
    for (let i = 0; i < this.maxParticles; i++) {
      const entity = this.entities[i];
      if (entity) {
        this.engine.remove(entity);
        this.entities[i] = null;
        this.entityIds[i] = -1;
      }
    }
    this.pooledCount = 0;
    this.emitters.clear();
    this.emitterSlots.length = 0;
    this.shapeTextureIds.clear();
  }
}

export function SpriteParticlePlugin(options?: SpriteParticlePluginOptions): EnginePlugin;
export function SpriteParticlePlugin(engine: Engine, options?: SpriteParticlePluginOptions): SpriteParticleManager;
export function SpriteParticlePlugin(
  engineOrOptions?: Engine | SpriteParticlePluginOptions,
  maybeOptions?: SpriteParticlePluginOptions,
): EnginePlugin | SpriteParticleManager {
  if (engineOrOptions && typeof (engineOrOptions as Engine).onTick === 'function') {
    const eng = engineOrOptions as Engine;
    const manager = new SpriteParticleManager(eng, maybeOptions);
    eng.use(() => () => manager.dispose());
    return manager;
  }
  const opts = (engineOrOptions as SpriteParticlePluginOptions | undefined) ?? {};
  return (engine: Engine) => {
    const manager = new SpriteParticleManager(engine, opts);
    return () => manager.dispose();
  };
}
