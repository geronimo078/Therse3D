import { DirtyFlag, GeometryTypeId, LightKindId, WorldStore } from './WorldStore';
import { SpatialLayout } from './SpatialLayout';
import {
  applyMorphWeightsToStore,
  SceneSync,
  type ScenePatchState,
  type SceneSyncCamera,
  type SceneSyncHooks,
  type SceneSyncResult,
  type SceneSyncState,
} from './SceneSync';
import { GPUDeviceContext, type GPUDeviceInitOptions, type QualityMode } from '../gpu/Device';
import { Renderer, type BloomConfig, type RendererOptions, type VolumetricConfig } from '../gpu/Renderer';
import {
  Inspector,
  pickFromScreen,
  projectWorldPoint,
  raycastStore,
  type ProjectedPoint,
  type RaycastHit,
  type SceneDescription,
} from '../ai/Inspector';
import { Snapshot, type SnapshotOptions, type SnapshotResult } from '../ai/Snapshot';
import { Schema } from '../ai/Schema';
import { EntityHandle } from '../entities/EntityHandle';
import { GltfLoader } from '../entities/GltfLoader';
import {
  applyLineEndpoints,
  type GeometryMeshData,
  type PrimitiveDescriptor,
  type PrimitiveType,
} from '../entities/Primitives';
import { parseColor, resolveMaterial } from '../entities/Materials';
import { Camera } from '../controls/OrbitControls';

export type EnginePreset = 'studio' | 'outdoor' | 'dark-ui' | 'cad-blueprint' | 'minimal';

export interface EngineOptions {
  canvas?: HTMLCanvasElement | string;
  preset?: EnginePreset;
  quality?: QualityMode;
  maxDpr?: number;
  autoResize?: boolean;
  antialias?: boolean;
  maxEntities?: number;
  onDeviceLost?: (info: GPUDeviceLostInfo) => void;
}

export type TickCallback = (dt: number, elapsed: number) => void;
export type EnginePlugin = (engine: Engine) => void | (() => void);

const PRIMITIVE_TYPE_MAP: Record<PrimitiveType, GeometryTypeId> = {
  box: GeometryTypeId.Box,
  sphere: GeometryTypeId.Sphere,
  cylinder: GeometryTypeId.Cylinder,
  plane: GeometryTypeId.Plane,
  line: GeometryTypeId.Line,
  model: GeometryTypeId.Model,
  group: GeometryTypeId.Group,
  light: GeometryTypeId.Light,
  cone: GeometryTypeId.Cone,
  torus: GeometryTypeId.Torus,
};

/**
 * Einstiegspunkt, Lifecycle, Presets, Ticker und Plugin-System für therse3d.
 */
export class Engine {
  readonly store: WorldStore;
  readonly gpu: GPUDeviceContext;
  readonly renderer: Renderer;
  readonly camera: Camera;
  readonly layout: SpatialLayout;
  readonly sceneSync: SceneSync;
  readonly inspector: Inspector;
  readonly snapshotter: Snapshot;
  readonly preset: EnginePreset;

  private rafId: number | null = null;
  private lastTime = 0;
  private elapsed = 0;
  private readonly tickCallbacks = new Set<TickCallback>();
  private readonly entitiesById = new Map<number, EntityHandle>();
  private readonly entitiesByName = new Map<string, EntityHandle>();
  private readonly pluginCleanups: Array<() => void> = [];
  private readonly textureCache = new Map<string, number>();
  private readonly gltfLoader = new GltfLoader();
  private readonly modelCache = new Map<string, number>();

  private constructor(
    gpu: GPUDeviceContext,
    renderer: Renderer,
    store: WorldStore,
    preset: EnginePreset,
  ) {
    this.gpu = gpu;
    this.renderer = renderer;
    this.store = store;
    this.preset = preset;
    this.camera = new Camera(this);
    this.layout = new SpatialLayout();
    this.sceneSync = new SceneSync(this.store, this.layout);
    this.inspector = new Inspector(store);
    this.snapshotter = new Snapshot(renderer);
  }

  static async create(options?: EngineOptions): Promise<Engine>;
  static async create(
    selectorOrCanvas: HTMLCanvasElement | string,
    options?: Omit<EngineOptions, 'canvas'>,
  ): Promise<Engine>;
  static async create(
    selectorOrOptions?: HTMLCanvasElement | string | EngineOptions,
    maybeOptions?: Omit<EngineOptions, 'canvas'>,
  ): Promise<Engine> {
    let options: EngineOptions;
    if (typeof selectorOrOptions === 'string' || selectorOrOptions instanceof HTMLCanvasElement) {
      options = { ...maybeOptions, canvas: selectorOrOptions };
    } else {
      options = selectorOrOptions ?? {};
    }

    const canvas = Engine.resolveCanvas(options.canvas);
    const gpuInitOptions: GPUDeviceInitOptions = {
      autoResize: options.autoResize ?? true,
      ...(options.quality !== undefined ? { quality: options.quality } : {}),
      ...(options.maxDpr !== undefined ? { maxDpr: options.maxDpr } : {}),
      ...(options.onDeviceLost !== undefined ? { onDeviceLost: options.onDeviceLost } : {}),
    };
    const gpu = await GPUDeviceContext.init(canvas, gpuInitOptions);
    const store = new WorldStore(options.maxEntities);
    const preset = options.preset ?? 'studio';
    const rendererOptions: RendererOptions = {
      ...(options.quality !== undefined ? { quality: options.quality } : {}),
      ...(options.antialias !== undefined ? { antialias: options.antialias } : {}),
    };
    const renderer = new Renderer(gpu, store, preset, rendererOptions);
    const engine = new Engine(gpu, renderer, store, preset);
    engine.start();
    return engine;
  }

  get isLowSpec(): boolean {
    return this.renderer.isLowSpec;
  }

  static getPromptContext(app?: Engine): string {
    return Schema.getPromptContext(app);
  }

  private static resolveCanvas(target?: HTMLCanvasElement | string): HTMLCanvasElement {
    if (typeof target === 'string') {
      const el = document.querySelector<HTMLCanvasElement>(target);
      if (!el) throw new Error(`Canvas selector "${target}" not found.`);
      return el;
    }
    if (target instanceof HTMLCanvasElement) {
      return target;
    }
    const canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';
    document.body.appendChild(canvas);
    return canvas;
  }

  add(descriptor: PrimitiveDescriptor): EntityHandle {
    const id = this.store.allocate();
    this.store.geometryTypes[id] = PRIMITIVE_TYPE_MAP[descriptor.type] ?? GeometryTypeId.Box;
    const handle = new EntityHandle(this.store, id, descriptor.id);

    const mat = resolveMaterial(descriptor.material);
    const color = descriptor.color !== undefined ? parseColor(descriptor.color) : mat.color;
    const cIdx = id * 4;
    this.store.colors[cIdx] = color[0];
    this.store.colors[cIdx + 1] = color[1];
    this.store.colors[cIdx + 2] = color[2];
    this.store.colors[cIdx + 3] = color[3];

    this.store.pbrParams[cIdx] =
      descriptor.metalness !== undefined && Number.isFinite(descriptor.metalness)
        ? Math.min(1, Math.max(0, descriptor.metalness))
        : mat.metalness;
    this.store.pbrParams[cIdx + 1] =
      descriptor.roughness !== undefined && Number.isFinite(descriptor.roughness)
        ? Math.min(1, Math.max(0.04, descriptor.roughness))
        : mat.roughness;
    this.store.pbrParams[cIdx + 2] =
      descriptor.emissive !== undefined && Number.isFinite(descriptor.emissive)
        ? Math.max(0, descriptor.emissive)
        : mat.emissive;
    this.store.pbrParams[cIdx + 3] = (mat.materialId & 0xff) + (this.store.textureIds[id] ?? 0) * 256;
    if (descriptor.texture !== undefined) {
      this.applyTextureRef(id, descriptor.texture);
    }

    if (descriptor.type === 'light') {
      this.store.lightParams[cIdx] =
        descriptor.kind === 'directional'
          ? LightKindId.Directional
          : descriptor.kind === 'spot'
            ? LightKindId.Spot
            : LightKindId.Point;
      this.store.lightParams[cIdx + 1] =
        descriptor.intensity !== undefined && Number.isFinite(descriptor.intensity)
          ? Math.max(0, descriptor.intensity)
          : 1.5;
      this.store.lightParams[cIdx + 2] =
        descriptor.range !== undefined && Number.isFinite(descriptor.range)
          ? Math.max(0.1, descriptor.range)
          : 10;
      this.store.lightParams[cIdx + 3] = descriptor.castShadow === false ? 0 : 1;
    }

    if (descriptor.parent) {
      const parentHandle = this.entitiesByName.get(descriptor.parent);
      if (parentHandle) {
        this.store.setParent(id, parentHandle.id);
      }
    }

    if (descriptor.position) {
      handle.setPosition(
        descriptor.position[0],
        descriptor.position[1],
        descriptor.position[2],
      );
    }

    if (descriptor.size) {
      handle.setScale(descriptor.size[0], descriptor.size[1], descriptor.size[2]);
    } else if (descriptor.radius !== undefined && Number.isFinite(descriptor.radius)) {
      const d = descriptor.radius * 2;
      handle.setScale(d, d, d);
    } else if (descriptor.scale !== undefined) {
      handle.scale = descriptor.scale;
    }

    if (descriptor.visible !== undefined) {
      handle.visible = descriptor.visible;
    }

    if (descriptor.type === 'line' && descriptor.from && descriptor.to) {
      applyLineEndpoints(this.store, id, descriptor.from, descriptor.to);
    }

    if (descriptor.type === 'model') {
      this.applyModelRef(id, descriptor.url, descriptor.mesh, undefined, descriptor.morphWeights);
      if (descriptor.morphWeights !== undefined) {
        const targetNames = this.renderer.getMeshMorphTargetNames(this.store.meshIds[id] ?? -1);
        applyMorphWeightsToStore(
          this.store,
          id,
          descriptor.morphWeights,
          targetNames,
          [],
          descriptor.id ?? `entity-${id}`,
        );
      }
    }

    this.entitiesById.set(id, handle);
    if (descriptor.id) {
      this.entitiesByName.set(descriptor.id, handle);
      this.sceneSync.registerExternalEntity(descriptor.id, id);
    }
    this.store.updateMatrices();

    if (
      descriptor.on !== undefined ||
      descriptor.beside !== undefined ||
      descriptor.inside !== undefined ||
      descriptor.offset !== undefined ||
      descriptor.lookAt !== undefined ||
      descriptor.avoidOverlap === true
    ) {
      this.layout.applyPlacement(
        this.store,
        id,
        descriptor,
        (name) => this.entitiesByName.get(name)?.id,
      );
    }

    return handle;
  }

  get(id: string | number): EntityHandle | undefined {
    if (typeof id === 'number') {
      return this.entitiesById.get(id);
    }
    return this.entitiesByName.get(id);
  }

  remove(target: string | number | EntityHandle): boolean {
    const handle =
      target instanceof EntityHandle
        ? target
        : typeof target === 'number'
          ? this.entitiesById.get(target)
          : this.entitiesByName.get(target);
    if (!handle) return false;

    this.entitiesById.delete(handle.id);
    if (handle.name) {
      this.entitiesByName.delete(handle.name);
      this.sceneSync.unregisterExternalEntity(handle.name);
    }
    handle.destroy();
    return true;
  }

  use(plugin: EnginePlugin): this {
    const cleanup = plugin(this);
    if (typeof cleanup === 'function') {
      this.pluginCleanups.push(cleanup);
    }
    return this;
  }

  onTick(cb: TickCallback): () => void {
    this.tickCallbacks.add(cb);
    return () => this.tickCallbacks.delete(cb);
  }

  private getSyncHooks(): SceneSyncHooks {
    return {
      onPreset: (preset, warnings) => {
        const valid: EnginePreset[] = ['studio', 'outdoor', 'dark-ui', 'cad-blueprint', 'minimal'];
        if (valid.includes(preset as EnginePreset)) {
          this.renderer.setPreset(preset as EnginePreset);
        } else {
          warnings.push(`Unknown preset "${preset}"; keeping "${this.preset}".`);
        }
      },
      onCamera: (camCfg, resolveEntityId, warnings) => {
        this.applySyncCamera(camCfg, resolveEntityId, warnings);
      },
      onEntityCreated: (name, entityId) => {
        const handle = new EntityHandle(this.store, entityId, name);
        this.entitiesById.set(entityId, handle);
        this.entitiesByName.set(name, handle);
      },
      onEntityRemoved: (name, entityId) => {
        this.entitiesById.delete(entityId);
        this.entitiesByName.delete(name);
      },
      onModelRef: (entityId, url, mesh, warnings, morphWeights) => {
        this.applyModelRef(entityId, url, mesh, warnings, morphWeights);
      },
      onTextureRef: (entityId, texture, warnings) => {
        this.applyTextureRef(entityId, texture, warnings);
      },
      onVolumetric: (config) => {
        this.setVolumetric(config);
      },
      resolveMorphTargetNames: (entityId) => {
        return this.renderer.getMeshMorphTargetNames(this.store.meshIds[entityId] ?? -1);
      },
    };
  }

  setVolumetric(config: VolumetricConfig): this {
    this.renderer.setVolumetric(config);
    return this;
  }

  setBloom(config: BloomConfig): this {
    this.renderer.setBloom(config);
    return this;
  }

  registerTexture(
    source: ImageBitmap | HTMLCanvasElement | OffscreenCanvas | Uint8Array | Uint8ClampedArray,
    width?: number,
    height?: number,
  ): number {
    return this.renderer.registerTexture(source, width, height);
  }

  async loadTexture(url: string): Promise<number> {
    const cached = this.textureCache.get(url);
    if (cached !== undefined) return cached;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch texture "${url}" (${response.status}).`);
    }
    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob, { imageOrientation: 'none' });
    const texId = this.renderer.registerTexture(bitmap);
    bitmap.close();
    this.textureCache.set(url, texId);
    return texId;
  }

  private applyTextureRef(entityId: number, texture: string | number, warnings?: string[]): void {
    if (typeof texture === 'number') {
      this.store.setTextureId(entityId, texture);
      return;
    }
    if (!texture) return;

    const cached = this.textureCache.get(texture);
    if (cached !== undefined) {
      this.store.setTextureId(entityId, cached);
      return;
    }

    void this.loadTexture(texture)
      .then((texId) => {
        if (this.store.alive[entityId] === 1) {
          this.store.setTextureId(entityId, texId);
        }
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        warnings?.push(`Failed to load texture "${texture}": ${msg}`);
        console.warn(`therse3d: Failed to load texture "${texture}": ${msg}`);
      });
  }

  async loadModel(url: string): Promise<number> {
    const cached = this.modelCache.get(url);
    if (cached !== undefined) return cached;

    const result = await this.gltfLoader.load(url);
    const firstMesh = result.meshes[0];
    if (!firstMesh) {
      throw new Error(`No renderable mesh found in glTF model "${url}": ${result.warnings.join('; ')}`);
    }
    const meshId = this.renderer.registerMesh(
      {
        vertices: firstMesh.vertices,
        indices: firstMesh.indices,
        ...(firstMesh.morphTargets ? { morphTargets: firstMesh.morphTargets } : {}),
      },
      firstMesh.name || url,
    );
    this.modelCache.set(url, meshId);
    return meshId;
  }

  private applyModelRef(
    entityId: number,
    url?: string,
    mesh?: GeometryMeshData,
    warnings?: string[],
    pendingMorphWeights?: Record<string, number> | number[],
  ): void {
    if (mesh) {
      const meshId = this.renderer.registerMesh(mesh, `entity-${entityId}-mesh`);
      this.store.meshIds[entityId] = meshId;
      const names = this.renderer.getMeshMorphTargetNames(meshId);
      if (names.length > 0) {
        this.store.morphTargetCounts[entityId] = Math.min(64, names.length);
        this.store.markDirty(entityId, DirtyFlag.Morph);
      }
      return;
    }
    if (!url) return;

    const cached = this.modelCache.get(url);
    if (cached !== undefined) {
      this.store.meshIds[entityId] = cached;
      const names = this.renderer.getMeshMorphTargetNames(cached);
      if (names.length > 0) {
        this.store.morphTargetCounts[entityId] = Math.min(64, names.length);
        this.store.markDirty(entityId, DirtyFlag.Morph);
      }
      return;
    }

    void this.loadModel(url)
      .then((meshId) => {
        if (this.store.alive[entityId] === 1) {
          this.store.meshIds[entityId] = meshId;
          const names = this.renderer.getMeshMorphTargetNames(meshId);
          if (names.length > 0) {
            this.store.morphTargetCounts[entityId] = Math.min(64, names.length);
            if (pendingMorphWeights !== undefined) {
              applyMorphWeightsToStore(
                this.store,
                entityId,
                pendingMorphWeights,
                names,
                warnings ?? [],
                `entity-${entityId}`,
              );
            }
            this.store.markDirty(entityId, DirtyFlag.Morph);
          }
        }
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        warnings?.push(`Failed to load model "${url}": ${msg}`);
        console.warn(`therse3d: Failed to load model "${url}": ${msg}`);
      });
  }

  sync(state: SceneSyncState): SceneSyncResult {
    return this.sceneSync.apply(state, this.getSyncHooks());
  }

  patch(patchState: ScenePatchState): SceneSyncResult {
    return this.sceneSync.patch(patchState, this.getSyncHooks());
  }

  undo(): SceneSyncResult {
    return this.sceneSync.undo(this.getSyncHooks());
  }

  focusCamera(cfg: SceneSyncCamera, warnings: string[] = []): void {
    this.applySyncCamera(cfg, (name) => this.entitiesByName.get(name)?.id, warnings);
  }

  private applySyncCamera(
    cfg: SceneSyncCamera,
    resolveEntityId: (name: string) => number | undefined,
    warnings: string[],
  ): void {
    let tx = 0, ty = 0, tz = 0;
    if (typeof cfg.target === 'string') {
      const id = resolveEntityId(cfg.target);
      if (id !== undefined) {
        this.store.updateMatrices();
        const mOff = id * 16;
        tx = this.store.modelMatrices[mOff + 12] ?? 0;
        ty = this.store.modelMatrices[mOff + 13] ?? 0;
        tz = this.store.modelMatrices[mOff + 14] ?? 0;
      } else {
        warnings.push(`Unknown camera target "${cfg.target}"; targeting origin.`);
      }
    } else if (Array.isArray(cfg.target)) {
      tx = Number.isFinite(cfg.target[0]) ? cfg.target[0] : 0;
      ty = Number.isFinite(cfg.target[1]) ? cfg.target[1] : 0;
      tz = Number.isFinite(cfg.target[2]) ? cfg.target[2] : 0;
    }

    this.camera.target[0] = tx;
    this.camera.target[1] = ty;
    this.camera.target[2] = tz;

    const dist = cfg.distance !== undefined && Number.isFinite(cfg.distance) ? Math.max(0.5, cfg.distance) : 4;
    switch (cfg.angle) {
      case 'top':
        this.camera.position[0] = tx;
        this.camera.position[1] = ty + dist;
        this.camera.position[2] = tz + 0.01;
        break;
      case 'front':
        this.camera.position[0] = tx;
        this.camera.position[1] = ty;
        this.camera.position[2] = tz + dist;
        break;
      case 'isometric':
        this.camera.position[0] = tx + dist * 0.58;
        this.camera.position[1] = ty + dist * 0.58;
        this.camera.position[2] = tz + dist * 0.58;
        break;
      case 'front-elevated':
      default:
        this.camera.position[0] = tx;
        this.camera.position[1] = ty + dist * 0.45;
        this.camera.position[2] = tz + dist * 0.9;
        break;
    }
    this.camera.notifyChanged();
  }

  describe(): SceneDescription {
    const canvas = this.gpu.canvas;
    const aspect = Math.max(1, canvas.width) / Math.max(1, canvas.height);
    const vp = this.camera.updateMatrices(aspect);
    return this.inspector.describe({
      viewProjMatrix: vp,
      resolveName: (id) => this.entitiesById.get(id)?.name,
      resolveMorphTargetNames: (id) => this.renderer.getMeshMorphTargetNames(this.store.meshIds[id] ?? -1),
    });
  }

  raycast(
    origin: readonly [number, number, number],
    direction: readonly [number, number, number],
  ): RaycastHit | null {
    return raycastStore(this.store, origin, direction, (id) => this.entitiesById.get(id)?.name);
  }

  pick(x: number, y: number, options?: { normalized?: boolean }): RaycastHit | null {
    const canvas = this.gpu.canvas;
    const w = Math.max(1, canvas.clientWidth || canvas.width);
    const h = Math.max(1, canvas.clientHeight || canvas.height);
    const nx = options?.normalized ? x : x / w;
    const ny = options?.normalized ? y : y / h;
    const vp = this.camera.updateMatrices(Math.max(1, canvas.width) / Math.max(1, canvas.height));
    return pickFromScreen(this.store, vp, nx, ny, (id) => this.entitiesById.get(id)?.name);
  }

  project(
    target: string | number | EntityHandle | readonly [number, number, number],
  ): ProjectedPoint | null {
    let point: readonly [number, number, number];
    if (Array.isArray(target)) {
      point = target as readonly [number, number, number];
    } else {
      const handle =
        target instanceof EntityHandle ? target : this.get(target as string | number);
      if (!handle) return null;
      this.store.updateMatrices();
      const mOff = handle.id * 16;
      point = [
        this.store.modelMatrices[mOff + 12] ?? 0,
        this.store.modelMatrices[mOff + 13] ?? 0,
        this.store.modelMatrices[mOff + 14] ?? 0,
      ];
    }

    const canvas = this.gpu.canvas;
    const w = Math.max(1, canvas.clientWidth || canvas.width);
    const h = Math.max(1, canvas.clientHeight || canvas.height);
    const vp = this.camera.updateMatrices(Math.max(1, canvas.width) / Math.max(1, canvas.height));
    return projectWorldPoint(vp, point, w, h);
  }

  async snapshot(options?: SnapshotOptions): Promise<SnapshotResult> {
    return this.snapshotter.capture(
      options,
      this.camera,
      this.inspector,
      (id) => this.entitiesById.get(id)?.name,
    );
  }

  start(): void {
    if (this.rafId !== null) return;
    this.lastTime = performance.now();
    const loop = (now: number) => {
      const dt = Math.min((now - this.lastTime) / 1000, 0.1);
      this.lastTime = now;
      this.elapsed += dt;
      for (const cb of this.tickCallbacks) {
        cb(dt, this.elapsed);
      }
      this.sceneSync.updateTweens(dt);
      this.store.updateMatrices();
      this.renderer.renderFrame(this.camera, this.elapsed);
      this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  }

  stop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  destroy(): void {
    this.stop();
    this.tickCallbacks.clear();
    for (const cleanup of this.pluginCleanups) {
      cleanup();
    }
    this.pluginCleanups.length = 0;
    this.entitiesById.clear();
    this.entitiesByName.clear();
    this.renderer.dispose();
    this.gpu.dispose();
  }

  dispose(): void {
    this.destroy();
  }
}
