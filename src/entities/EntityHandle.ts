import { DirtyFlag, type WorldStore } from '../core/WorldStore';

export type Vec3Like = readonly [number, number, number] | Float32Array;
export type Vec4Like = readonly [number, number, number, number] | Float32Array;

/**
 * Leichtgewichtiger Proxy/Wrapper über einen ECS-Index (OOP-Ergonomie).
 * Alle Getter/Setter lesen und schreiben direkt in die flachen TypedArrays von WorldStore.
 */
export class EntityHandle {
  readonly id: number;
  readonly name: string | undefined;
  private readonly store: WorldStore;

  private boundGen = -1;
  private posView!: Float32Array;
  private rotView!: Float32Array;
  private scaleView!: Float32Array;
  private colorView!: Float32Array;

  constructor(store: WorldStore, id: number, name?: string) {
    this.store = store;
    this.id = id;
    this.name = name;
    this.syncViews();
  }

  private syncViews(): void {
    if (this.boundGen === this.store.generation) return;
    const pIdx = this.id * 3;
    const qIdx = this.id * 4;
    this.posView = this.store.positions.subarray(pIdx, pIdx + 3);
    this.rotView = this.store.rotations.subarray(qIdx, qIdx + 4);
    this.scaleView = this.store.scales.subarray(pIdx, pIdx + 3);
    this.colorView = this.store.colors.subarray(qIdx, qIdx + 4);
    this.boundGen = this.store.generation;
  }

  get position(): Float32Array {
    this.syncViews();
    this.store.markDirty(this.id, DirtyFlag.Transform | DirtyFlag.Bounds);
    return this.posView;
  }

  set position(value: Vec3Like) {
    this.setPosition(value[0] ?? 0, value[1] ?? 0, value[2] ?? 0);
  }

  get rotation(): Float32Array {
    this.syncViews();
    this.store.markDirty(this.id, DirtyFlag.Transform | DirtyFlag.Bounds);
    return this.rotView;
  }

  set rotation(value: Vec4Like) {
    const idx = this.id * 4;
    this.store.rotations[idx] = value[0] ?? 0;
    this.store.rotations[idx + 1] = value[1] ?? 0;
    this.store.rotations[idx + 2] = value[2] ?? 0;
    this.store.rotations[idx + 3] = value[3] ?? 1;
    this.store.markDirty(this.id, DirtyFlag.Transform | DirtyFlag.Bounds);
  }

  get scale(): Float32Array {
    this.syncViews();
    this.store.markDirty(this.id, DirtyFlag.Transform | DirtyFlag.Bounds);
    return this.scaleView;
  }

  set scale(value: Vec3Like | number) {
    if (typeof value === 'number') {
      this.setScale(value, value, value);
    } else {
      this.setScale(value[0] ?? 1, value[1] ?? 1, value[2] ?? 1);
    }
  }

  get color(): Float32Array {
    this.syncViews();
    this.store.markDirty(this.id, DirtyFlag.Material);
    return this.colorView;
  }

  set color(value: Vec4Like | Vec3Like) {
    const idx = this.id * 4;
    this.store.colors[idx] = value[0] ?? 1;
    this.store.colors[idx + 1] = value[1] ?? 1;
    this.store.colors[idx + 2] = value[2] ?? 1;
    this.store.colors[idx + 3] = value.length >= 4 ? (value[3] ?? 1) : 1;
    this.store.markDirty(this.id, DirtyFlag.Material);
  }

  get visible(): boolean {
    return (this.store.visible[this.id] ?? 0) === 1;
  }

  set visible(value: boolean) {
    this.store.visible[this.id] = value ? 1 : 0;
  }

  setPosition(x: number, y: number, z: number): this {
    const idx = this.id * 3;
    this.store.positions[idx] = Number.isFinite(x) ? x : 0;
    this.store.positions[idx + 1] = Number.isFinite(y) ? y : 0;
    this.store.positions[idx + 2] = Number.isFinite(z) ? z : 0;
    this.store.markDirty(this.id, DirtyFlag.Transform | DirtyFlag.Bounds);
    return this;
  }

  getPosition(): [number, number, number] {
    const idx = this.id * 3;
    return [
      this.store.positions[idx] ?? 0,
      this.store.positions[idx + 1] ?? 0,
      this.store.positions[idx + 2] ?? 0,
    ];
  }

  setScale(x: number, y = x, z = x): this {
    const idx = this.id * 3;
    this.store.scales[idx] = Number.isFinite(x) ? x : 1;
    this.store.scales[idx + 1] = Number.isFinite(y) ? y : 1;
    this.store.scales[idx + 2] = Number.isFinite(z) ? z : 1;
    this.store.markDirty(this.id, DirtyFlag.Transform | DirtyFlag.Bounds);
    return this;
  }

  get texture(): number {
    return this.store.textureIds[this.id] ?? 0;
  }

  set texture(textureId: number) {
    this.store.setTextureId(this.id, textureId);
  }

  destroy(): void {
    this.store.free(this.id);
  }
}
