import type { Engine, EnginePlugin } from '../core/Engine';
import { mat4 } from '../core/Math';

export interface OrbitControlsOptions {
  damping?: number;
  distance?: number;
  target?: [number, number, number];
  theta?: number;
  phi?: number;
}

export interface FitAllOptions {
  padding?: number;
}

/**
 * Kamera-Zustand & Projektionsmatrizen (zero-allocation).
 */
export class Camera {
  readonly viewMatrix = new Float32Array(16);
  readonly projMatrix = new Float32Array(16);
  readonly viewProjMatrix = new Float32Array(16);
  readonly position = new Float32Array([0, 2.2, 5.0]);
  readonly target = new Float32Array([0, 0.5, 0]);

  fovY = (45 * Math.PI) / 180;
  near = 0.05;
  far = 500;

  private readonly engine: Engine;

  constructor(engine: Engine) {
    this.engine = engine;
  }

  fitAll(options: FitAllOptions = {}): void {
    const padding = options.padding ?? 1.25;
    const store = this.engine.store;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let found = false;

    for (let i = 0; i < store.highWaterMark; i++) {
      if (store.alive[i] === 0 || store.visible[i] === 0) continue;
      found = true;
      const idx = i * 3;
      const bMinX = store.boundsMin[idx] ?? 0;
      const bMinY = store.boundsMin[idx + 1] ?? 0;
      const bMinZ = store.boundsMin[idx + 2] ?? 0;
      const bMaxX = store.boundsMax[idx] ?? 0;
      const bMaxY = store.boundsMax[idx + 1] ?? 0;
      const bMaxZ = store.boundsMax[idx + 2] ?? 0;
      if (bMinX < minX) minX = bMinX;
      if (bMinY < minY) minY = bMinY;
      if (bMinZ < minZ) minZ = bMinZ;
      if (bMaxX > maxX) maxX = bMaxX;
      if (bMaxY > maxY) maxY = bMaxY;
      if (bMaxZ > maxZ) maxZ = bMaxZ;
    }

    if (!found) {
      this.target[0] = 0; this.target[1] = 0; this.target[2] = 0;
      this.position[0] = 0; this.position[1] = 2.2; this.position[2] = 5.0;
      return;
    }

    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    const cz = (minZ + maxZ) * 0.5;
    const radius = Math.max(0.5, Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) * 0.5);
    const dist = (radius * padding) / Math.sin(this.fovY * 0.5);

    this.target[0] = cx;
    this.target[1] = cy;
    this.target[2] = cz;

    this.position[0] = cx + dist * 0.55;
    this.position[1] = cy + dist * 0.45;
    this.position[2] = cz + dist * 0.70;
  }

  updateMatrices(aspect: number): Float32Array {
    mat4.perspective(this.projMatrix, this.fovY, aspect, this.near, this.far);
    mat4.lookAt(
      this.viewMatrix,
      this.position[0] ?? 0,
      this.position[1] ?? 2,
      this.position[2] ?? 5,
      this.target[0] ?? 0,
      this.target[1] ?? 0,
      this.target[2] ?? 0,
    );
    mat4.multiply(this.viewProjMatrix, this.projMatrix, this.viewMatrix);
    return this.viewProjMatrix;
  }
}

/**
 * Kamera-Steuerung mit Smooth-Damping, Orbit, Pan, Zoom & camera.fitAll().
 * Kann als `app.use(OrbitControls())` oder `new OrbitControls(app)` genutzt werden.
 */
export function OrbitControls(options?: OrbitControlsOptions): EnginePlugin;
export function OrbitControls(engine: Engine, options?: OrbitControlsOptions): OrbitControlsController;
export function OrbitControls(
  engineOrOptions?: Engine | OrbitControlsOptions,
  maybeOptions?: OrbitControlsOptions,
): EnginePlugin | OrbitControlsController {
  if (engineOrOptions && typeof (engineOrOptions as Engine).onTick === 'function') {
    return new OrbitControlsController(engineOrOptions as Engine, maybeOptions);
  }
  const opts = (engineOrOptions as OrbitControlsOptions | undefined) ?? {};
  return (engine: Engine) => {
    const controller = new OrbitControlsController(engine, opts);
    return () => controller.dispose();
  };
}

export class OrbitControlsController {
  readonly damping: number;
  distance: number;
  theta: number;
  phi: number;

  private targetDistance: number;
  private targetTheta: number;
  private targetPhi: number;

  private isDragging = false;
  private isPanning = false;
  private lastX = 0;
  private lastY = 0;

  private readonly engine: Engine;
  private readonly unsubTick: () => void;

  constructor(engine: Engine, options: OrbitControlsOptions = {}) {
    this.engine = engine;
    this.damping = options.damping ?? 0.12;
    this.distance = options.distance ?? 5;
    this.theta = options.theta ?? 0.65;
    this.phi = options.phi ?? 1.05;

    this.targetDistance = this.distance;
    this.targetTheta = this.theta;
    this.targetPhi = this.phi;

    if (options.target) {
      engine.camera.target[0] = options.target[0];
      engine.camera.target[1] = options.target[1];
      engine.camera.target[2] = options.target[2];
    }

    const canvas = engine.gpu.canvas;
    canvas.addEventListener('pointerdown', this.onPointerDown);
    window.addEventListener('pointermove', this.onPointerMove);
    window.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('wheel', this.onWheel, { passive: false });
    canvas.addEventListener('contextmenu', this.onContextMenu);

    this.unsubTick = engine.onTick(() => this.update());
    this.syncCameraPosition();
  }

  private onContextMenu = (e: MouseEvent): void => {
    e.preventDefault();
  };

  private onPointerDown = (e: PointerEvent): void => {
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    if (e.button === 2 || e.shiftKey) {
      this.isPanning = true;
    } else {
      this.isDragging = true;
    }
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.isDragging && !this.isPanning) return;
    const dx = e.clientX - this.lastX;
    const dy = e.clientY - this.lastY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;

    if (this.isDragging) {
      this.targetTheta -= dx * 0.008;
      this.targetPhi = Math.min(Math.PI - 0.05, Math.max(0.05, this.targetPhi - dy * 0.008));
    } else if (this.isPanning) {
      const panSpeed = this.distance * 0.0015;
      const cosT = Math.cos(this.theta);
      const sinT = Math.sin(this.theta);
      const t = this.engine.camera.target;
      t[0] = (t[0] ?? 0) - (dx * cosT) * panSpeed;
      t[1] = (t[1] ?? 0) + dy * panSpeed;
      t[2] = (t[2] ?? 0) + (dx * sinT) * panSpeed;
    }
  };

  private onPointerUp = (): void => {
    this.isDragging = false;
    this.isPanning = false;
  };

  private onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    this.targetDistance = Math.min(200, Math.max(0.5, this.targetDistance * (1 + e.deltaY * 0.001)));
  };

  update(): void {
    const k = this.damping;
    this.theta += (this.targetTheta - this.theta) * k;
    this.phi += (this.targetPhi - this.phi) * k;
    this.distance += (this.targetDistance - this.distance) * k;
    this.syncCameraPosition();
  }

  private syncCameraPosition(): void {
    const sinPhi = Math.sin(this.phi);
    const cam = this.engine.camera;
    const tx = cam.target[0] ?? 0;
    const ty = cam.target[1] ?? 0;
    const tz = cam.target[2] ?? 0;

    cam.position[0] = tx + this.distance * sinPhi * Math.sin(this.theta);
    cam.position[1] = ty + this.distance * Math.cos(this.phi);
    cam.position[2] = tz + this.distance * sinPhi * Math.cos(this.theta);
  }

  fitAll(options?: FitAllOptions): void {
    this.engine.camera.fitAll(options);
  }

  dispose(): void {
    this.unsubTick();
    const canvas = this.engine.gpu.canvas;
    canvas.removeEventListener('pointerdown', this.onPointerDown);
    window.removeEventListener('pointermove', this.onPointerMove);
    window.removeEventListener('pointerup', this.onPointerUp);
    canvas.removeEventListener('wheel', this.onWheel);
    canvas.removeEventListener('contextmenu', this.onContextMenu);
  }
}
