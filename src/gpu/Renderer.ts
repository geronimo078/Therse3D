import type { GPUDeviceContext } from './Device';
import { mat4 } from '../core/Math';
import { GeometryTypeId, type WorldStore } from '../core/WorldStore';
import { BufferManager } from './BufferManager';
import {
  createBoxGeometry,
  createConeGeometry,
  createCylinderGeometry,
  createLineGeometry,
  createPlaneGeometry,
  createSphereGeometry,
  createTorusGeometry,
  type GeometryMeshData,
} from '../entities/Primitives';
import type { EnginePreset } from '../core/Engine';
import type { Camera } from '../controls/OrbitControls';
import pbrShaderCode from './shaders/pbr.wgsl?raw';
import idDepthShaderCode from './shaders/id_depth.wgsl?raw';
import volumetricShaderCode from './shaders/volumetric.wgsl?raw';
import bloomShaderCode from './shaders/bloom.wgsl?raw';
import { parseColor } from '../entities/Materials';

export interface BloomConfig {
  enabled?: boolean;
  threshold?: number;
  intensity?: number;
  radius?: number;
}

export interface VolumetricConfig {
  enabled?: boolean;
  fogDensity?: number;
  fogColor?: string | [number, number, number];
  fogHeight?: number;
  cloudDensity?: number;
  cloudCoverage?: number;
  cloudMode?: 'planar' | 'spherical';
  cloudScale?: number;
  cloudSpeed?: number;
  cloudAltitude?: [number, number];
  planetRadius?: number;
  planetCenter?: [number, number, number];
  scatter?: number;
  godRays?: number;
}

interface GeometryGpuMesh {
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  indexCount: number;
  indexFormat: GPUIndexFormat;
}

interface PresetLighting {
  clearColor: GPUColorDict;
  skyColor: [number, number, number, number];
  groundColor: [number, number, number, number];
  keyLightDir: [number, number, number, number];
  keyLightColor: [number, number, number, number];
}

const PRESETS: Record<EnginePreset, PresetLighting> = {
  studio: {
    clearColor: { r: 0.08, g: 0.09, b: 0.12, a: 1.0 },
    skyColor: [0.85, 0.90, 1.00, 1.0],
    groundColor: [0.18, 0.16, 0.15, 1.0],
    keyLightDir: [0.55, 0.85, 0.45, 0.0],
    keyLightColor: [1.0, 0.97, 0.92, 1.0],
  },
  outdoor: {
    clearColor: { r: 0.42, g: 0.64, b: 0.88, a: 1.0 },
    skyColor: [0.65, 0.82, 1.00, 1.0],
    groundColor: [0.28, 0.25, 0.20, 1.0],
    keyLightDir: [0.60, 0.90, 0.35, 0.0],
    keyLightColor: [1.0, 0.95, 0.85, 1.0],
  },
  'dark-ui': {
    clearColor: { r: 0.03, g: 0.04, b: 0.06, a: 1.0 },
    skyColor: [0.40, 0.48, 0.65, 1.0],
    groundColor: [0.06, 0.06, 0.08, 1.0],
    keyLightDir: [0.45, 0.80, 0.55, 0.0],
    keyLightColor: [0.95, 0.97, 1.0, 1.0],
  },
  'cad-blueprint': {
    clearColor: { r: 0.06, g: 0.16, b: 0.32, a: 1.0 },
    skyColor: [0.75, 0.88, 1.0, 1.0],
    groundColor: [0.12, 0.22, 0.38, 1.0],
    keyLightDir: [0.50, 0.85, 0.50, 0.0],
    keyLightColor: [1.0, 1.0, 1.0, 1.0],
  },
  minimal: {
    clearColor: { r: 0.92, g: 0.93, b: 0.95, a: 1.0 },
    skyColor: [1.0, 1.0, 1.0, 1.0],
    groundColor: [0.35, 0.35, 0.38, 1.0],
    keyLightDir: [0.40, 0.85, 0.50, 0.0],
    keyLightColor: [1.0, 1.0, 1.0, 1.0],
  },
};

const MESH_TYPES = [
  GeometryTypeId.Box,
  GeometryTypeId.Sphere,
  GeometryTypeId.Cylinder,
  GeometryTypeId.Plane,
  GeometryTypeId.Cone,
  GeometryTypeId.Torus,
] as const;

export type SnapshotLayer = 'rgb' | 'annotated' | 'depth' | 'ids';

export interface CapturePassResult {
  width: number;
  height: number;
  rgbPixels: Uint8ClampedArray | null;
  objectIds: Uint32Array | null;
  depth: Float32Array | null;
}

const MSAA_SAMPLE_COUNT = 4;
const SHADOW_MAP_SIZE = 2048;
const TEXTURE_SIZE = 1024;
const MAX_TEXTURES = 16;

/**
 * WebGPU Instanced PBR-Renderer mit MSAA (4x), Depth-Buffer & Zero-Allocation Render-Loop.
 */
export class Renderer {
  readonly gpu: GPUDeviceContext;
  readonly store: WorldStore;
  readonly buffers: BufferManager;
  readonly pbrShaderSource: string;
  readonly idDepthShaderSource: string;

  private preset: EnginePreset;
  private readonly pipeline: GPURenderPipeline;
  private readonly transparentPipeline: GPURenderPipeline;
  private readonly linePipeline: GPURenderPipeline;
  private readonly shadowPipeline: GPURenderPipeline;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly shadowBindGroupLayout: GPUBindGroupLayout;
  private offscreenPbrPipeline: GPURenderPipeline | null = null;
  private idDepthPipeline: GPURenderPipeline | null = null;
  private idDepthBindGroupLayout: GPUBindGroupLayout | null = null;
  private captureUniformBuffer: GPUBuffer | null = null;
  private readonly captureUniformScratch = new Float32Array(24); // mat4(16) + camPos(4) + nearFar(4)
  private readonly meshes = new Map<GeometryTypeId, GeometryGpuMesh>();
  private readonly customMeshes = new Map<number, GeometryGpuMesh>();
  private nextCustomMeshId = 1;

  private readonly uniformBuffer: GPUBuffer;
  // mat4(16) + 5*vec4(20) + counts(4) + 8*DynamicLight(64) + 16*vec4 shadowCasters(64) + lightViewProj(16) = 184 floats
  private readonly uniformScratch = new Float32Array(184);
  private readonly lightViewScratch = new Float32Array(16);
  private readonly lightProjScratch = new Float32Array(16);
  private readonly lightViewProjScratch = new Float32Array(16);

  private readonly shadowTexture: GPUTexture;
  private readonly shadowView: GPUTextureView;
  private readonly shadowSampler: GPUSampler;
  private readonly albedoTextureArray: GPUTexture;
  private readonly albedoTextureView: GPUTextureView;
  private readonly albedoSampler: GPUSampler;
  private nextTextureLayer = 1;

  private allocatedCapacity = 0;
  private modelBuffer!: GPUBuffer;
  private colorBuffer!: GPUBuffer;
  private pbrBuffer!: GPUBuffer;
  private instanceIdBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;
  private shadowBindGroup!: GPUBindGroup;

  // Pre-allocated CPU scratch buffers for zero-allocation instancing bucketing
  private readonly bucketCounts = new Uint32Array(12);
  private readonly bucketOffsets = new Uint32Array(12);
  private readonly bucketCursors = new Uint32Array(12);
  private readonly transCounts = new Uint32Array(12);
  private readonly transOffsets = new Uint32Array(12);
  private readonly transCursors = new Uint32Array(12);
  private packedInstanceIds!: Uint32Array;

  // Render target textures (recreated on canvas resize)
  private msaaColorTexture: GPUTexture | null = null;
  private msaaColorView: GPUTextureView | null = null;
  private sceneColorTexture: GPUTexture | null = null;
  private sceneColorView: GPUTextureView | null = null;
  private postColorTexture: GPUTexture | null = null;
  private postColorView: GPUTextureView | null = null;
  private depthTexture: GPUTexture | null = null;
  private depthView: GPUTextureView | null = null;
  private rtWidth = 0;
  private rtHeight = 0;

  // Volumetric Fog & Raymarched Clouds Post-Pass (lazily initialized on enable)
  private volumetricEnabled = false;
  private volumetricPipeline: GPURenderPipeline | null = null;
  private volumetricBindGroupLayout: GPUBindGroupLayout | null = null;
  private volumetricBindGroup: GPUBindGroup | null = null;
  private volumetricUniformBuffer: GPUBuffer | null = null;
  private readonly volumetricScratch = new Float32Array(44); // 176 bytes
  private readonly invViewProjScratch = new Float32Array(16);
  private volumetricConfig: Required<Omit<VolumetricConfig, 'fogColor'>> & { fogColorRgb: [number, number, number] } = {
    enabled: false,
    fogDensity: 0.014,
    fogColorRgb: [0.54, 0.71, 0.85],
    fogHeight: 8.0,
    cloudDensity: 0.85,
    cloudCoverage: 0.48,
    cloudMode: 'planar',
    cloudScale: 1.1,
    cloudSpeed: 0.14,
    cloudAltitude: [4.0, 7.0],
    planetRadius: 2.0,
    planetCenter: [0, 0, 0],
    scatter: 0.62,
    godRays: 0.85,
  };

  // Bloom Post-Pass (lazily initialized on enable)
  private bloomEnabled = false;
  private bloomPipeline: GPURenderPipeline | null = null;
  private bloomBindGroupLayout: GPUBindGroupLayout | null = null;
  private bloomBindGroupScene: GPUBindGroup | null = null;
  private bloomBindGroupPost: GPUBindGroup | null = null;
  private bloomUniformBuffer: GPUBuffer | null = null;
  private bloomSampler: GPUSampler | null = null;
  private readonly bloomScratch = new Float32Array(8); // 32 bytes
  private bloomConfig: Required<BloomConfig> = {
    enabled: false,
    threshold: 0.78,
    intensity: 0.65,
    radius: 1.5,
  };

  constructor(gpu: GPUDeviceContext, store: WorldStore, preset: EnginePreset = 'studio') {
    this.gpu = gpu;
    this.store = store;
    this.preset = preset;
    this.buffers = new BufferManager(gpu.device);
    this.pbrShaderSource = pbrShaderCode;
    this.idDepthShaderSource = idDepthShaderCode;

    this.uniformBuffer = this.buffers.createUniformBuffer(this.uniformScratch.byteLength, 'scene-uniforms');

    this.shadowTexture = this.buffers.createTexture({
      label: 'directional-shadow-map',
      size: [SHADOW_MAP_SIZE, SHADOW_MAP_SIZE],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowView = this.shadowTexture.createView();
    this.shadowSampler = gpu.device.createSampler({
      label: 'shadow-sampler',
      compare: 'less',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    this.albedoTextureArray = this.buffers.createTexture({
      label: 'albedo-texture-array',
      size: [TEXTURE_SIZE, TEXTURE_SIZE, MAX_TEXTURES],
      format: 'rgba8unorm-srgb',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    // Layer 0 = solid white fallback so untextured entities sample (1, 1, 1, 1)
    const whiteLayer = new Uint8Array(TEXTURE_SIZE * TEXTURE_SIZE * 4);
    whiteLayer.fill(255);
    gpu.device.queue.writeTexture(
      { texture: this.albedoTextureArray, origin: [0, 0, 0] },
      whiteLayer,
      { bytesPerRow: TEXTURE_SIZE * 4, rowsPerImage: TEXTURE_SIZE },
      [TEXTURE_SIZE, TEXTURE_SIZE, 1],
    );
    this.albedoTextureView = this.albedoTextureArray.createView({
      dimension: '2d-array',
      arrayLayerCount: MAX_TEXTURES,
    });
    this.albedoSampler = gpu.device.createSampler({
      label: 'albedo-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });

    this.bindGroupLayout = gpu.device.createBindGroupLayout({
      label: 'pbr-bind-group-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    this.shadowBindGroupLayout = gpu.device.createBindGroupLayout({
      label: 'shadow-bind-group-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    const shaderModule = gpu.device.createShaderModule({
      label: 'pbr-shader',
      code: pbrShaderCode,
    });

    this.pipeline = gpu.device.createRenderPipeline({
      label: 'pbr-instanced-pipeline',
      layout: gpu.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [
          {
            arrayStride: 8 * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 3 * 4, format: 'float32x3' },
              { shaderLocation: 2, offset: 6 * 4, format: 'float32x2' },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: gpu.format }],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      multisample: {
        count: MSAA_SAMPLE_COUNT,
      },
    });

    this.transparentPipeline = gpu.device.createRenderPipeline({
      label: 'pbr-transparent-pipeline',
      layout: gpu.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [
          {
            arrayStride: 8 * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 3 * 4, format: 'float32x3' },
              { shaderLocation: 2, offset: 6 * 4, format: 'float32x2' },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [
          {
            format: gpu.format,
            blend: {
              color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            },
          },
        ],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
      multisample: {
        count: MSAA_SAMPLE_COUNT,
      },
    });

    this.linePipeline = gpu.device.createRenderPipeline({
      label: 'line-instanced-pipeline',
      layout: gpu.device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [
          {
            arrayStride: 8 * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 3 * 4, format: 'float32x3' },
              { shaderLocation: 2, offset: 6 * 4, format: 'float32x2' },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_unlit',
        targets: [{ format: gpu.format }],
      },
      primitive: {
        topology: 'line-list',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      multisample: {
        count: MSAA_SAMPLE_COUNT,
      },
    });

    this.shadowPipeline = gpu.device.createRenderPipeline({
      label: 'shadow-map-pipeline',
      layout: gpu.device.createPipelineLayout({ bindGroupLayouts: [this.shadowBindGroupLayout] }),
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_shadow',
        buffers: [
          {
            arrayStride: 8 * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: 'float32x3' },
              { shaderLocation: 1, offset: 3 * 4, format: 'float32x3' },
              { shaderLocation: 2, offset: 6 * 4, format: 'float32x2' },
            ],
          },
        ],
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this.uploadPrimitiveMesh(GeometryTypeId.Box, createBoxGeometry(), 'box');
    this.uploadPrimitiveMesh(GeometryTypeId.Sphere, createSphereGeometry(), 'sphere');
    this.uploadPrimitiveMesh(GeometryTypeId.Cylinder, createCylinderGeometry(), 'cylinder');
    this.uploadPrimitiveMesh(GeometryTypeId.Plane, createPlaneGeometry(), 'plane');
    this.uploadPrimitiveMesh(GeometryTypeId.Line, createLineGeometry(), 'line');
    this.uploadPrimitiveMesh(GeometryTypeId.Cone, createConeGeometry(), 'cone');
    this.uploadPrimitiveMesh(GeometryTypeId.Torus, createTorusGeometry(), 'torus');

    this.ensureStorageCapacity(store.capacity);
  }

  setPreset(preset: EnginePreset): void {
    this.preset = preset;
  }

  private uploadPrimitiveMesh(typeId: GeometryTypeId, data: GeometryMeshData, label: string): void {
    const vertexBuffer = this.buffers.createVertexBuffer(data.vertices, `${label}-vb`);
    const indexBuffer = this.buffers.createIndexBuffer(data.indices, `${label}-ib`);
    this.meshes.set(typeId, {
      vertexBuffer,
      indexBuffer,
      indexCount: data.indices.length,
      indexFormat: data.indices instanceof Uint32Array ? 'uint32' : 'uint16',
    });
  }

  registerMesh(data: GeometryMeshData, label = 'custom-mesh'): number {
    const id = this.nextCustomMeshId++;
    const vertexBuffer = this.buffers.createVertexBuffer(data.vertices, `${label}-${id}-vb`);
    const indexBuffer = this.buffers.createIndexBuffer(data.indices, `${label}-${id}-ib`);
    this.customMeshes.set(id, {
      vertexBuffer,
      indexBuffer,
      indexCount: data.indices.length,
      indexFormat: data.indices instanceof Uint32Array ? 'uint32' : 'uint16',
    });
    return id;
  }

  unloadMesh(meshId: number): void {
    const mesh = this.customMeshes.get(meshId);
    if (!mesh) return;
    this.buffers.destroyBuffer(mesh.vertexBuffer);
    this.buffers.destroyBuffer(mesh.indexBuffer);
    this.customMeshes.delete(meshId);
  }

  registerTexture(
    source: ImageBitmap | HTMLCanvasElement | OffscreenCanvas | Uint8Array | Uint8ClampedArray,
    width?: number,
    height?: number,
  ): number {
    if (this.nextTextureLayer >= MAX_TEXTURES) {
      console.warn(`therse3d: Texture array capacity (${MAX_TEXTURES - 1}) reached; falling back to default texture.`);
      return 0;
    }
    const layer = this.nextTextureLayer++;

    if (source instanceof Uint8Array || source instanceof Uint8ClampedArray) {
      const w = Math.max(1, width ?? TEXTURE_SIZE);
      const h = Math.max(1, height ?? TEXTURE_SIZE);
      if (w === TEXTURE_SIZE && h === TEXTURE_SIZE) {
        this.gpu.device.queue.writeTexture(
          { texture: this.albedoTextureArray, origin: [0, 0, layer] },
          source.buffer as ArrayBuffer,
          { offset: source.byteOffset, bytesPerRow: TEXTURE_SIZE * 4, rowsPerImage: TEXTURE_SIZE },
          [TEXTURE_SIZE, TEXTURE_SIZE, 1],
        );
      } else {
        // Nearest/bilinear resample into 512x512 RGBA8 buffer
        const resampled = new Uint8Array(TEXTURE_SIZE * TEXTURE_SIZE * 4);
        for (let y = 0; y < TEXTURE_SIZE; y++) {
          const sy = Math.min(h - 1, Math.floor((y / TEXTURE_SIZE) * h));
          for (let x = 0; x < TEXTURE_SIZE; x++) {
            const sx = Math.min(w - 1, Math.floor((x / TEXTURE_SIZE) * w));
            const sOff = (sy * w + sx) * 4;
            const dOff = (y * TEXTURE_SIZE + x) * 4;
            resampled[dOff] = source[sOff] ?? 255;
            resampled[dOff + 1] = source[sOff + 1] ?? 255;
            resampled[dOff + 2] = source[sOff + 2] ?? 255;
            resampled[dOff + 3] = source[sOff + 3] ?? 255;
          }
        }
        this.gpu.device.queue.writeTexture(
          { texture: this.albedoTextureArray, origin: [0, 0, layer] },
          resampled,
          { bytesPerRow: TEXTURE_SIZE * 4, rowsPerImage: TEXTURE_SIZE },
          [TEXTURE_SIZE, TEXTURE_SIZE, 1],
        );
      }
      return layer;
    }

    let uploadSource: ImageBitmap | HTMLCanvasElement | OffscreenCanvas = source;
    if (source.width !== TEXTURE_SIZE || source.height !== TEXTURE_SIZE) {
      const canvas =
        typeof OffscreenCanvas !== 'undefined'
          ? new OffscreenCanvas(TEXTURE_SIZE, TEXTURE_SIZE)
          : document.createElement('canvas');
      canvas.width = TEXTURE_SIZE;
      canvas.height = TEXTURE_SIZE;
      const ctx = canvas.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
      if (ctx) {
        ctx.drawImage(source, 0, 0, TEXTURE_SIZE, TEXTURE_SIZE);
        uploadSource = canvas;
      }
    }

    this.gpu.device.queue.copyExternalImageToTexture(
      { source: uploadSource, flipY: false },
      { texture: this.albedoTextureArray, origin: [0, 0, layer] },
      [TEXTURE_SIZE, TEXTURE_SIZE],
    );
    return layer;
  }

  private ensureStorageCapacity(capacity: number): void {
    if (this.allocatedCapacity >= capacity) return;

    if (this.allocatedCapacity > 0) {
      this.buffers.destroyBuffer(this.modelBuffer);
      this.buffers.destroyBuffer(this.colorBuffer);
      this.buffers.destroyBuffer(this.pbrBuffer);
      this.buffers.destroyBuffer(this.instanceIdBuffer);
    }

    this.allocatedCapacity = capacity;
    this.packedInstanceIds = new Uint32Array(capacity);

    this.modelBuffer = this.buffers.createStorageBuffer(capacity * 16 * 4, 'entity-models');
    this.colorBuffer = this.buffers.createStorageBuffer(capacity * 4 * 4, 'entity-colors');
    this.pbrBuffer = this.buffers.createStorageBuffer(capacity * 4 * 4, 'entity-pbr');
    this.instanceIdBuffer = this.buffers.createStorageBuffer(capacity * 4, 'instance-ids');

    this.bindGroup = this.gpu.device.createBindGroup({
      label: 'pbr-bind-group',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.modelBuffer } },
        { binding: 2, resource: { buffer: this.colorBuffer } },
        { binding: 3, resource: { buffer: this.pbrBuffer } },
        { binding: 4, resource: { buffer: this.instanceIdBuffer } },
        { binding: 5, resource: this.shadowView },
        { binding: 6, resource: this.shadowSampler },
        { binding: 7, resource: this.albedoTextureView },
        { binding: 8, resource: this.albedoSampler },
      ],
    });

    this.shadowBindGroup = this.gpu.device.createBindGroup({
      label: 'shadow-bind-group',
      layout: this.shadowBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.modelBuffer } },
        { binding: 3, resource: { buffer: this.pbrBuffer } },
        { binding: 4, resource: { buffer: this.instanceIdBuffer } },
      ],
    });
  }

  setVolumetric(config: VolumetricConfig): void {
    if (config.enabled !== undefined) this.volumetricConfig.enabled = Boolean(config.enabled);
    if (config.fogDensity !== undefined && Number.isFinite(config.fogDensity)) {
      this.volumetricConfig.fogDensity = Math.max(0, config.fogDensity);
    }
    if (config.fogHeight !== undefined && Number.isFinite(config.fogHeight)) {
      this.volumetricConfig.fogHeight = Math.max(0.5, config.fogHeight);
    }
    if (config.fogColor !== undefined) {
      const c = parseColor(config.fogColor);
      this.volumetricConfig.fogColorRgb = [c[0], c[1], c[2]];
    }
    if (config.cloudDensity !== undefined && Number.isFinite(config.cloudDensity)) {
      this.volumetricConfig.cloudDensity = Math.max(0, config.cloudDensity);
    }
    if (config.cloudCoverage !== undefined && Number.isFinite(config.cloudCoverage)) {
      this.volumetricConfig.cloudCoverage = Math.min(1, Math.max(0, config.cloudCoverage));
    }
    if (config.cloudMode === 'planar' || config.cloudMode === 'spherical') {
      this.volumetricConfig.cloudMode = config.cloudMode;
    }
    if (config.cloudScale !== undefined && Number.isFinite(config.cloudScale)) {
      this.volumetricConfig.cloudScale = Math.max(0.05, config.cloudScale);
    }
    if (config.cloudSpeed !== undefined && Number.isFinite(config.cloudSpeed)) {
      this.volumetricConfig.cloudSpeed = config.cloudSpeed;
    }
    if (Array.isArray(config.cloudAltitude) && config.cloudAltitude.length === 2) {
      const a0 = Number.isFinite(config.cloudAltitude[0]) ? config.cloudAltitude[0] : 4.0;
      const a1 = Number.isFinite(config.cloudAltitude[1]) ? config.cloudAltitude[1] : a0 + 1.0;
      this.volumetricConfig.cloudAltitude = [a0, Math.max(a0 + 0.05, a1)];
    }
    if (config.planetRadius !== undefined && Number.isFinite(config.planetRadius)) {
      this.volumetricConfig.planetRadius = Math.max(0.1, config.planetRadius);
    }
    if (Array.isArray(config.planetCenter) && config.planetCenter.length === 3) {
      this.volumetricConfig.planetCenter = [
        Number.isFinite(config.planetCenter[0]) ? config.planetCenter[0] : 0,
        Number.isFinite(config.planetCenter[1]) ? config.planetCenter[1] : 0,
        Number.isFinite(config.planetCenter[2]) ? config.planetCenter[2] : 0,
      ];
    }
    if (config.scatter !== undefined && Number.isFinite(config.scatter)) {
      this.volumetricConfig.scatter = Math.min(0.92, Math.max(0.05, config.scatter));
    }
    if (config.godRays !== undefined && Number.isFinite(config.godRays)) {
      this.volumetricConfig.godRays = Math.min(3.0, Math.max(0, config.godRays));
    }
    this.volumetricEnabled = this.volumetricConfig.enabled;
  }

  private ensureVolumetricPipeline(): void {
    if (this.volumetricPipeline) return;
    const device = this.gpu.device;
    this.volumetricUniformBuffer = this.buffers.createUniformBuffer(
      this.volumetricScratch.byteLength,
      'volumetric-uniforms',
    );
    this.volumetricBindGroupLayout = device.createBindGroupLayout({
      label: 'volumetric-bind-group-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', multisampled: true } },
      ],
    });
    const mod = device.createShaderModule({ label: 'volumetric-shader', code: volumetricShaderCode });
    this.volumetricPipeline = device.createRenderPipeline({
      label: 'volumetric-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.volumetricBindGroupLayout] }),
      vertex: { module: mod, entryPoint: 'vs_volumetric' },
      fragment: {
        module: mod,
        entryPoint: 'fs_volumetric',
        targets: [{ format: this.gpu.format }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  private rebuildVolumetricBindGroup(): void {
    if (!this.volumetricBindGroupLayout || !this.volumetricUniformBuffer || !this.sceneColorView || !this.depthView) {
      return;
    }
    this.volumetricBindGroup = this.gpu.device.createBindGroup({
      label: 'volumetric-bind-group',
      layout: this.volumetricBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.volumetricUniformBuffer } },
        { binding: 1, resource: this.sceneColorView },
        { binding: 2, resource: this.depthView },
      ],
    });
  }

  setBloom(config: BloomConfig): void {
    if (config.enabled !== undefined) this.bloomConfig.enabled = Boolean(config.enabled);
    if (config.threshold !== undefined && Number.isFinite(config.threshold)) {
      this.bloomConfig.threshold = Math.min(2.0, Math.max(0.0, config.threshold));
    }
    if (config.intensity !== undefined && Number.isFinite(config.intensity)) {
      this.bloomConfig.intensity = Math.min(3.0, Math.max(0.0, config.intensity));
    }
    if (config.radius !== undefined && Number.isFinite(config.radius)) {
      this.bloomConfig.radius = Math.min(5.0, Math.max(0.25, config.radius));
    }
    this.bloomEnabled = this.bloomConfig.enabled;
  }

  private ensureBloomPipeline(): void {
    if (this.bloomPipeline) return;
    const device = this.gpu.device;
    this.bloomUniformBuffer = this.buffers.createUniformBuffer(
      this.bloomScratch.byteLength,
      'bloom-uniforms',
    );
    this.bloomSampler = device.createSampler({
      label: 'bloom-linear-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    this.bloomBindGroupLayout = device.createBindGroupLayout({
      label: 'bloom-bind-group-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    const mod = device.createShaderModule({ label: 'bloom-shader', code: bloomShaderCode });
    this.bloomPipeline = device.createRenderPipeline({
      label: 'bloom-pipeline',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bloomBindGroupLayout] }),
      vertex: { module: mod, entryPoint: 'vs_bloom' },
      fragment: {
        module: mod,
        entryPoint: 'fs_bloom',
        targets: [{ format: this.gpu.format }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  private rebuildBloomBindGroups(): void {
    if (
      !this.bloomBindGroupLayout ||
      !this.bloomUniformBuffer ||
      !this.bloomSampler ||
      !this.sceneColorView ||
      !this.postColorView
    ) {
      return;
    }
    this.bloomBindGroupScene = this.gpu.device.createBindGroup({
      label: 'bloom-bind-group-scene',
      layout: this.bloomBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.bloomUniformBuffer } },
        { binding: 1, resource: this.sceneColorView },
        { binding: 2, resource: this.bloomSampler },
      ],
    });
    this.bloomBindGroupPost = this.gpu.device.createBindGroup({
      label: 'bloom-bind-group-post',
      layout: this.bloomBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.bloomUniformBuffer } },
        { binding: 1, resource: this.postColorView },
        { binding: 2, resource: this.bloomSampler },
      ],
    });
  }

  private ensureRenderTargets(width: number, height: number): void {
    if (this.rtWidth === width && this.rtHeight === height && this.depthView && this.msaaColorView) {
      return;
    }

    if (this.msaaColorTexture) this.buffers.destroyTexture(this.msaaColorTexture);
    if (this.sceneColorTexture) this.buffers.destroyTexture(this.sceneColorTexture);
    if (this.postColorTexture) this.buffers.destroyTexture(this.postColorTexture);
    if (this.depthTexture) this.buffers.destroyTexture(this.depthTexture);

    this.rtWidth = width;
    this.rtHeight = height;

    this.msaaColorTexture = this.buffers.createTexture({
      label: 'msaa-color',
      size: [width, height],
      sampleCount: MSAA_SAMPLE_COUNT,
      format: this.gpu.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.msaaColorView = this.msaaColorTexture.createView();

    this.sceneColorTexture = this.buffers.createTexture({
      label: 'scene-color-resolve',
      size: [width, height],
      format: this.gpu.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.sceneColorView = this.sceneColorTexture.createView();

    this.postColorTexture = this.buffers.createTexture({
      label: 'post-color-intermediate',
      size: [width, height],
      format: this.gpu.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.postColorView = this.postColorTexture.createView();

    this.depthTexture = this.buffers.createTexture({
      label: 'depth-24plus',
      size: [width, height],
      sampleCount: MSAA_SAMPLE_COUNT,
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depthView = this.depthTexture.createView();
    this.rebuildVolumetricBindGroup();
    this.rebuildBloomBindGroups();
  }

  renderFrame(camera?: Camera, elapsed = 0): void {
    if (this.gpu.isLost) return;

    const canvas = this.gpu.canvas;
    if (canvas.width === 0 || canvas.height === 0) return;
    const width = canvas.width;
    const height = canvas.height;

    this.ensureRenderTargets(width, height);
    this.ensureStorageCapacity(this.store.capacity);

    const presetCfg = PRESETS[this.preset];
    const u = this.uniformScratch;

    if (camera) {
      const vp = camera.updateMatrices(width / height);
      u.set(vp, 0);
      u[16] = camera.position[0] ?? 0;
      u[17] = camera.position[1] ?? 2;
      u[18] = camera.position[2] ?? 5;
      u[19] = 1;
    } else {
      u[0] = 1; u[5] = 1; u[10] = 1; u[15] = 1;
      u[16] = 0; u[17] = 2; u[18] = 5; u[19] = 1;
    }

    u.set(presetCfg.skyColor, 20);
    u.set(presetCfg.groundColor, 24);
    u.set(presetCfg.keyLightDir, 28);
    u.set(presetCfg.keyLightColor, 32);

    const n = this.store.highWaterMark;
    const counts = this.bucketCounts;
    const offsets = this.bucketOffsets;
    const cursors = this.bucketCursors;
    const tCounts = this.transCounts;
    const tOffsets = this.transOffsets;
    const tCursors = this.transCursors;
    counts.fill(0);
    tCounts.fill(0);

    let dynamicLightCount = 0;
    let shadowCasterCount = 0;

    if (n > 0) {
      this.buffers.writeBuffer(this.modelBuffer, this.store.modelMatrices, 0, 0, n * 16);
      this.buffers.writeBuffer(this.colorBuffer, this.store.colors, 0, 0, n * 4);
      this.buffers.writeBuffer(this.pbrBuffer, this.store.pbrParams, 0, 0, n * 4);

      const alive = this.store.alive;
      const vis = this.store.visible;
      const gTypes = this.store.geometryTypes;
      const mats = this.store.modelMatrices;
      const cols = this.store.colors;
      const lParams = this.store.lightParams;
      const bMin = this.store.boundsMin;
      const bMax = this.store.boundsMax;

      for (let id = 0; id < n; id++) {
        if (alive[id] === 0 || vis[id] === 0) continue;
        const gType = gTypes[id] ?? 0;
        const alpha = cols[id * 4 + 3] ?? 1;
        const isTransparent = alpha < 0.995 && gType !== GeometryTypeId.Line && gType !== GeometryTypeId.Light;
        if (gType < 12) {
          if (isTransparent) {
            tCounts[gType] = (tCounts[gType] ?? 0) + 1;
          } else {
            counts[gType] = (counts[gType] ?? 0) + 1;
          }
        }

        if (gType === GeometryTypeId.Light && dynamicLightCount < 8) {
          const base = 40 + dynamicLightCount * 8;
          const mOff = id * 16;
          const cOff = id * 4;
          const lx = mats[mOff + 12] ?? 0;
          const ly = mats[mOff + 13] ?? 0;
          const lz = mats[mOff + 14] ?? 0;
          const kind = lParams[cOff] ?? 0;
          u[base + 0] = lx;
          u[base + 1] = ly;
          u[base + 2] = lz;
          u[base + 3] = kind; // kind: 0 = point, 1 = directional
          u[base + 4] = cols[cOff] ?? 1;
          u[base + 5] = cols[cOff + 1] ?? 1;
          u[base + 6] = cols[cOff + 2] ?? 1;
          u[base + 7] = lParams[cOff + 1] ?? 1; // intensity
          dynamicLightCount++;

          if (kind > 0.5 && Math.hypot(lx, ly, lz) > 1e-4) {
            u[28] = lx;
            u[29] = ly;
            u[30] = lz;
            u[31] = 0;
          }
        } else if (
          (gType === GeometryTypeId.Box || gType === GeometryTypeId.Sphere || gType === GeometryTypeId.Cylinder) &&
          shadowCasterCount < 16
        ) {
          const base = 104 + shadowCasterCount * 4;
          const pIdx = id * 3;
          const minX = bMin[pIdx] ?? 0;
          const minY = bMin[pIdx + 1] ?? 0;
          const minZ = bMin[pIdx + 2] ?? 0;
          const maxX = bMax[pIdx] ?? 0;
          const maxZ = bMax[pIdx + 2] ?? 0;
          u[base + 0] = (minX + maxX) * 0.5;
          u[base + 1] = minY;
          u[base + 2] = (minZ + maxZ) * 0.5;
          u[base + 3] = Math.max(maxX - minX, maxZ - minZ) * 0.5;
          shadowCasterCount++;
        }
      }

      let totalVisible = 0;
      for (let t = 0; t < 12; t++) {
        offsets[t] = totalVisible;
        cursors[t] = totalVisible;
        totalVisible += counts[t] ?? 0;
      }
      for (let t = 0; t < 12; t++) {
        tOffsets[t] = totalVisible;
        tCursors[t] = totalVisible;
        totalVisible += tCounts[t] ?? 0;
      }

      if (totalVisible > 0) {
        const packed = this.packedInstanceIds;
        for (let id = 0; id < n; id++) {
          if (alive[id] === 0 || vis[id] === 0) continue;
          const gType = gTypes[id] ?? 0;
          if (gType < 12) {
            const alpha = cols[id * 4 + 3] ?? 1;
            const isTransparent = alpha < 0.995 && gType !== GeometryTypeId.Line && gType !== GeometryTypeId.Light;
            if (isTransparent) {
              const dst = tCursors[gType] ?? 0;
              packed[dst] = id;
              tCursors[gType] = dst + 1;
            } else {
              const dst = cursors[gType] ?? 0;
              packed[dst] = id;
              cursors[gType] = dst + 1;
            }
          }
        }
        this.buffers.writeBuffer(this.instanceIdBuffer, packed, 0, 0, totalVisible);
      }
    }

    u[36] = dynamicLightCount;
    u[37] = shadowCasterCount;
    u[38] = 0;
    u[39] = 0;
    this.writeLightViewProj(u[28] ?? 0.5, u[29] ?? 0.85, u[30] ?? 0.45);
    this.buffers.writeBuffer(this.uniformBuffer, u);

    const encoder = this.gpu.device.createCommandEncoder();
    this.recordShadowPass(encoder, counts, offsets);
    const swapchainView = this.gpu.context.getCurrentTexture().createView();
    const useVolumetric = this.volumetricEnabled;
    const useBloom = this.bloomEnabled;
    const usePost = useVolumetric || useBloom;
    if (useVolumetric) {
      this.ensureVolumetricPipeline();
      if (!this.volumetricBindGroup) {
        this.rebuildVolumetricBindGroup();
      }
    }
    if (useBloom) {
      this.ensureBloomPipeline();
      if (!this.bloomBindGroupScene || !this.bloomBindGroupPost) {
        this.rebuildBloomBindGroups();
      }
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.msaaColorView!,
          resolveTarget: usePost ? this.sceneColorView! : swapchainView,
          clearValue: presetCfg.clearColor,
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: this.depthView!,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: useVolumetric ? 'store' : 'discard',
      },
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    this.drawPbrMeshes(pass, counts, offsets);

    pass.setPipeline(this.transparentPipeline);
    pass.setBindGroup(0, this.bindGroup);
    this.drawPbrMeshes(pass, tCounts, tOffsets);

    const lineCount = counts[GeometryTypeId.Line] ?? 0;
    if (lineCount > 0) {
      const lineMesh = this.meshes.get(GeometryTypeId.Line);
      if (lineMesh) {
        pass.setPipeline(this.linePipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.setVertexBuffer(0, lineMesh.vertexBuffer);
        pass.setIndexBuffer(lineMesh.indexBuffer, lineMesh.indexFormat);
        pass.drawIndexed(lineMesh.indexCount, lineCount, 0, 0, offsets[GeometryTypeId.Line] ?? 0);
      }
    }

    pass.end();

    if (useVolumetric && this.volumetricPipeline && this.volumetricBindGroup && this.volumetricUniformBuffer) {
      const vp = camera ? camera.viewProjMatrix : u.subarray(0, 16);
      mat4.invert(this.invViewProjScratch, vp);
      const vs = this.volumetricScratch;
      vs.set(this.invViewProjScratch, 0);
      vs[16] = u[16] ?? 0;
      vs[17] = u[17] ?? 0;
      vs[18] = u[18] ?? 0;
      vs[19] = elapsed;

      const vc = this.volumetricConfig;
      vs[20] = u[28] ?? 0.5;
      vs[21] = u[29] ?? 0.85;
      vs[22] = u[30] ?? 0.45;
      vs[23] = vc.fogDensity;

      vs[24] = vc.fogColorRgb[0];
      vs[25] = vc.fogColorRgb[1];
      vs[26] = vc.fogColorRgb[2];
      vs[27] = vc.fogHeight;

      vs[28] = vc.cloudDensity;
      vs[29] = vc.cloudCoverage;
      vs[30] = vc.cloudMode === 'spherical' ? 1 : 0;
      vs[31] = vc.cloudScale;

      vs[32] = vc.planetCenter[0];
      vs[33] = vc.planetCenter[1];
      vs[34] = vc.planetCenter[2];
      vs[35] = vc.planetRadius;

      vs[36] = vc.cloudAltitude[0];
      vs[37] = vc.cloudAltitude[1];
      vs[38] = vc.cloudSpeed;
      vs[39] = vc.scatter;

      // Project sun position to screen UV & NDC depth for radial god rays
      const lx = u[28] ?? 0.5;
      const ly = u[29] ?? 0.85;
      const lz = u[30] ?? 0.45;
      const sunLen = Math.hypot(lx, ly, lz) || 1;
      const sx = sunLen > 2.5 ? lx : (u[16] ?? 0) + (lx / sunLen) * 50;
      const sy = sunLen > 2.5 ? ly : (u[17] ?? 0) + (ly / sunLen) * 50;
      const sz = sunLen > 2.5 ? lz : (u[18] ?? 0) + (lz / sunLen) * 50;
      const clipX = (vp[0] ?? 0) * sx + (vp[4] ?? 0) * sy + (vp[8] ?? 0) * sz + (vp[12] ?? 0);
      const clipY = (vp[1] ?? 0) * sx + (vp[5] ?? 0) * sy + (vp[9] ?? 0) * sz + (vp[13] ?? 0);
      const clipZ = (vp[2] ?? 0) * sx + (vp[6] ?? 0) * sy + (vp[10] ?? 0) * sz + (vp[14] ?? 0);
      const clipW = (vp[3] ?? 0) * sx + (vp[7] ?? 0) * sy + (vp[11] ?? 0) * sz + (vp[15] ?? 0);
      if (clipW > 0.001) {
        vs[40] = (clipX / clipW) * 0.5 + 0.5;
        vs[41] = 0.5 - (clipY / clipW) * 0.5;
        vs[42] = Math.max(0.0001, Math.min(1.0, clipZ / clipW));
        vs[43] = vc.godRays;
      } else {
        vs[40] = 0.5;
        vs[41] = 0.5;
        vs[42] = -1.0;
        vs[43] = 0.0;
      }

      this.buffers.writeBuffer(this.volumetricUniformBuffer, vs);

      const vPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: useBloom ? this.postColorView! : swapchainView,
            clearValue: presetCfg.clearColor,
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      vPass.setPipeline(this.volumetricPipeline);
      vPass.setBindGroup(0, this.volumetricBindGroup);
      vPass.draw(3);
      vPass.end();
    }

    if (useBloom && this.bloomPipeline && this.bloomUniformBuffer) {
      const bloomBindGroup = useVolumetric ? this.bloomBindGroupPost : this.bloomBindGroupScene;
      if (bloomBindGroup) {
        const bs = this.bloomScratch;
        const bc = this.bloomConfig;
        bs[0] = 1.0 / Math.max(1, this.rtWidth);
        bs[1] = 1.0 / Math.max(1, this.rtHeight);
        bs[2] = bc.threshold;
        bs[3] = bc.intensity;
        bs[4] = bc.radius;
        bs[5] = 0;
        bs[6] = 0;
        bs[7] = 0;
        this.buffers.writeBuffer(this.bloomUniformBuffer, bs);

        const bPass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: swapchainView,
              clearValue: presetCfg.clearColor,
              loadOp: 'clear',
              storeOp: 'store',
            },
          ],
        });
        bPass.setPipeline(this.bloomPipeline);
        bPass.setBindGroup(0, bloomBindGroup);
        bPass.draw(3);
        bPass.end();
      }
    }

    this.gpu.device.queue.submit([encoder.finish()]);
  }

  private writeLightViewProj(lx: number, ly: number, lz: number): void {
    const len = Math.max(1e-4, Math.hypot(lx, ly, lz));
    const dist = 18;
    mat4.lookAt(
      this.lightViewScratch,
      (lx / len) * dist,
      (ly / len) * dist,
      (lz / len) * dist,
      0,
      0,
      0,
    );
    mat4.ortho(this.lightProjScratch, -8, 8, -8, 8, 2, 36);
    mat4.multiply(this.lightViewProjScratch, this.lightProjScratch, this.lightViewScratch);
    this.uniformScratch.set(this.lightViewProjScratch, 168);
  }

  private recordShadowPass(encoder: GPUCommandEncoder, counts: Uint32Array, offsets: Uint32Array): void {
    const shadowPass = encoder.beginRenderPass({
      depthStencilAttachment: {
        view: this.shadowView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
      colorAttachments: [],
    });
    shadowPass.setPipeline(this.shadowPipeline);
    shadowPass.setBindGroup(0, this.shadowBindGroup);
    this.drawPbrMeshes(shadowPass, counts, offsets);
    shadowPass.end();
  }

  private drawPbrMeshes(pass: GPURenderPassEncoder, counts: Uint32Array, offsets: Uint32Array): void {
    for (let i = 0; i < MESH_TYPES.length; i++) {
      const gType = MESH_TYPES[i]!;
      const count = counts[gType] ?? 0;
      if (count === 0) continue;
      const mesh = this.meshes.get(gType);
      if (!mesh) continue;

      pass.setVertexBuffer(0, mesh.vertexBuffer);
      pass.setIndexBuffer(mesh.indexBuffer, mesh.indexFormat);
      pass.drawIndexed(mesh.indexCount, count, 0, 0, offsets[gType] ?? 0);
    }

    const modelCount = counts[GeometryTypeId.Model] ?? 0;
    if (modelCount > 0 && this.customMeshes.size > 0) {
      const modelOffset = offsets[GeometryTypeId.Model] ?? 0;
      const packed = this.packedInstanceIds;
      const meshIds = this.store.meshIds;
      for (let i = 0; i < modelCount; i++) {
        const slot = modelOffset + i;
        const entityId = packed[slot]!;
        const meshId = meshIds[entityId] ?? -1;
        if (meshId < 0) continue;
        const mesh = this.customMeshes.get(meshId);
        if (!mesh) continue;
        pass.setVertexBuffer(0, mesh.vertexBuffer);
        pass.setIndexBuffer(mesh.indexBuffer, mesh.indexFormat);
        pass.drawIndexed(mesh.indexCount, 1, 0, 0, slot);
      }
    }
  }

  private ensureCapturePipelines(): void {
    const device = this.gpu.device;
    if (!this.offscreenPbrPipeline) {
      const pbrModule = device.createShaderModule({ label: 'pbr-offscreen-shader', code: pbrShaderCode });
      this.offscreenPbrPipeline = device.createRenderPipeline({
        label: 'pbr-offscreen-pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
        vertex: {
          module: pbrModule,
          entryPoint: 'vs_main',
          buffers: [
            {
              arrayStride: 8 * 4,
              attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x3' },
                { shaderLocation: 1, offset: 3 * 4, format: 'float32x3' },
                { shaderLocation: 2, offset: 6 * 4, format: 'float32x2' },
              ],
            },
          ],
        },
        fragment: {
          module: pbrModule,
          entryPoint: 'fs_main',
          targets: [{ format: 'rgba8unorm' }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      });
    }

    if (!this.idDepthPipeline) {
      this.captureUniformBuffer = this.buffers.createUniformBuffer(this.captureUniformScratch.byteLength, 'capture-uniforms');
      this.idDepthBindGroupLayout = device.createBindGroupLayout({
        label: 'id-depth-bind-group-layout',
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        ],
      });
      const idModule = device.createShaderModule({ label: 'id-depth-shader', code: idDepthShaderCode });
      this.idDepthPipeline = device.createRenderPipeline({
        label: 'id-depth-mrt-pipeline',
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.idDepthBindGroupLayout] }),
        vertex: {
          module: idModule,
          entryPoint: 'vs_main',
          buffers: [
            {
              arrayStride: 8 * 4,
              attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x3' },
                { shaderLocation: 1, offset: 3 * 4, format: 'float32x3' },
                { shaderLocation: 2, offset: 6 * 4, format: 'float32x2' },
              ],
            },
          ],
        },
        fragment: {
          module: idModule,
          entryPoint: 'fs_main',
          targets: [{ format: 'r32uint' }, { format: 'r32float' }],
        },
        primitive: { topology: 'triangle-list', cullMode: 'back' },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      });
    }
  }

  async renderCapturePass(
    width: number,
    height: number,
    camera: Camera | undefined,
    layers: readonly SnapshotLayer[],
  ): Promise<CapturePassResult> {
    if (this.gpu.isLost) {
      return { width, height, rgbPixels: null, objectIds: null, depth: null };
    }

    const needRgb = layers.includes('rgb') || layers.includes('annotated');
    const needIds = layers.includes('ids');
    const needDepth = layers.includes('depth');

    this.ensureCapturePipelines();
    this.ensureStorageCapacity(this.store.capacity);

    const device = this.gpu.device;
    const presetCfg = PRESETS[this.preset];
    const u = this.uniformScratch;
    const cu = this.captureUniformScratch;

    if (camera) {
      const vp = camera.updateMatrices(width / height);
      u.set(vp, 0);
      u[16] = camera.position[0] ?? 0;
      u[17] = camera.position[1] ?? 2;
      u[18] = camera.position[2] ?? 5;
      u[19] = 1;

      cu.set(vp, 0);
      cu[16] = u[16]; cu[17] = u[17]; cu[18] = u[18]; cu[19] = 1;
      cu[20] = camera.near; cu[21] = camera.far; cu[22] = 0; cu[23] = 0;
    } else {
      u[0] = 1; u[5] = 1; u[10] = 1; u[15] = 1;
      u[16] = 0; u[17] = 2; u[18] = 5; u[19] = 1;
      cu.set(u.subarray(0, 20), 0);
      cu[20] = 0.05; cu[21] = 500; cu[22] = 0; cu[23] = 0;
    }

    u.set(presetCfg.skyColor, 20);
    u.set(presetCfg.groundColor, 24);
    u.set(presetCfg.keyLightDir, 28);
    u.set(presetCfg.keyLightColor, 32);

    const n = this.store.highWaterMark;
    const counts = this.bucketCounts;
    const offsets = this.bucketOffsets;
    const cursors = this.bucketCursors;
    const tCounts = this.transCounts;
    const tOffsets = this.transOffsets;
    const tCursors = this.transCursors;
    counts.fill(0);
    tCounts.fill(0);

    let dynamicLightCount = 0;
    let shadowCasterCount = 0;

    if (n > 0) {
      this.buffers.writeBuffer(this.modelBuffer, this.store.modelMatrices, 0, 0, n * 16);
      this.buffers.writeBuffer(this.colorBuffer, this.store.colors, 0, 0, n * 4);
      this.buffers.writeBuffer(this.pbrBuffer, this.store.pbrParams, 0, 0, n * 4);

      const alive = this.store.alive;
      const vis = this.store.visible;
      const gTypes = this.store.geometryTypes;
      const mats = this.store.modelMatrices;
      const cols = this.store.colors;
      const lParams = this.store.lightParams;
      const bMin = this.store.boundsMin;
      const bMax = this.store.boundsMax;

      for (let id = 0; id < n; id++) {
        if (alive[id] === 0 || vis[id] === 0) continue;
        const gType = gTypes[id] ?? 0;
        const alpha = cols[id * 4 + 3] ?? 1;
        const isTransparent = alpha < 0.995 && gType !== GeometryTypeId.Line && gType !== GeometryTypeId.Light;
        if (gType < 12) {
          if (isTransparent) tCounts[gType] = (tCounts[gType] ?? 0) + 1;
          else counts[gType] = (counts[gType] ?? 0) + 1;
        }

        if (gType === GeometryTypeId.Light && dynamicLightCount < 8) {
          const base = 40 + dynamicLightCount * 8;
          const mOff = id * 16;
          const cOff = id * 4;
          u[base + 0] = mats[mOff + 12] ?? 0;
          u[base + 1] = mats[mOff + 13] ?? 0;
          u[base + 2] = mats[mOff + 14] ?? 0;
          u[base + 3] = lParams[cOff] ?? 0;
          u[base + 4] = cols[cOff] ?? 1;
          u[base + 5] = cols[cOff + 1] ?? 1;
          u[base + 6] = cols[cOff + 2] ?? 1;
          u[base + 7] = lParams[cOff + 1] ?? 1;
          dynamicLightCount++;
        } else if (
          !isTransparent &&
          (gType === GeometryTypeId.Box ||
            gType === GeometryTypeId.Sphere ||
            gType === GeometryTypeId.Cylinder ||
            gType === GeometryTypeId.Cone ||
            gType === GeometryTypeId.Torus) &&
          shadowCasterCount < 16
        ) {
          const base = 104 + shadowCasterCount * 4;
          const pIdx = id * 3;
          const minX = bMin[pIdx] ?? 0;
          const minY = bMin[pIdx + 1] ?? 0;
          const minZ = bMin[pIdx + 2] ?? 0;
          const maxX = bMax[pIdx] ?? 0;
          const maxZ = bMax[pIdx + 2] ?? 0;
          u[base + 0] = (minX + maxX) * 0.5;
          u[base + 1] = minY;
          u[base + 2] = (minZ + maxZ) * 0.5;
          u[base + 3] = Math.max(maxX - minX, maxZ - minZ) * 0.5;
          shadowCasterCount++;
        }
      }

      let totalVisible = 0;
      for (let t = 0; t < 12; t++) {
        offsets[t] = totalVisible;
        cursors[t] = totalVisible;
        totalVisible += counts[t] ?? 0;
      }
      for (let t = 0; t < 12; t++) {
        tOffsets[t] = totalVisible;
        tCursors[t] = totalVisible;
        totalVisible += tCounts[t] ?? 0;
      }

      if (totalVisible > 0) {
        const packed = this.packedInstanceIds;
        for (let id = 0; id < n; id++) {
          if (alive[id] === 0 || vis[id] === 0) continue;
          const gType = gTypes[id] ?? 0;
          if (gType < 12) {
            const alpha = cols[id * 4 + 3] ?? 1;
            const isTransparent = alpha < 0.995 && gType !== GeometryTypeId.Line && gType !== GeometryTypeId.Light;
            if (isTransparent) {
              const dst = tCursors[gType] ?? 0;
              packed[dst] = id;
              tCursors[gType] = dst + 1;
            } else {
              const dst = cursors[gType] ?? 0;
              packed[dst] = id;
              cursors[gType] = dst + 1;
            }
          }
        }
        this.buffers.writeBuffer(this.instanceIdBuffer, packed, 0, 0, totalVisible);
      }
    }

    u[36] = dynamicLightCount;
    u[37] = shadowCasterCount;
    u[38] = 0;
    u[39] = 0;
    this.writeLightViewProj(u[28] ?? 0.5, u[29] ?? 0.85, u[30] ?? 0.45);
    this.buffers.writeBuffer(this.uniformBuffer, u);
    this.buffers.writeBuffer(this.captureUniformBuffer!, cu);

    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    const readbackSize = bytesPerRow * height;

    const captureDepthTex = this.buffers.createTexture({
      label: 'capture-depth',
      size: [width, height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    let rgbTex: GPUTexture | null = null;
    let rgbReadBuf: GPUBuffer | null = null;
    let idTex: GPUTexture | null = null;
    let idReadBuf: GPUBuffer | null = null;
    let depthTex: GPUTexture | null = null;
    let depthReadBuf: GPUBuffer | null = null;

    try {
      const encoder = device.createCommandEncoder();

      if (needRgb) {
        this.recordShadowPass(encoder, counts, offsets);
        rgbTex = this.buffers.createTexture({
          label: 'capture-rgb',
          size: [width, height],
          format: 'rgba8unorm',
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        rgbReadBuf = this.buffers.createBuffer({
          label: 'capture-rgb-readback',
          size: readbackSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: rgbTex.createView(), clearValue: presetCfg.clearColor, loadOp: 'clear', storeOp: 'store' }],
          depthStencilAttachment: { view: captureDepthTex.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'discard' },
        });
        pass.setPipeline(this.offscreenPbrPipeline!);
        pass.setBindGroup(0, this.bindGroup);
        this.drawPbrMeshes(pass, counts, offsets);
        pass.end();
        encoder.copyTextureToBuffer({ texture: rgbTex }, { buffer: rgbReadBuf, bytesPerRow }, { width, height });
      }

      if (needIds || needDepth) {
        idTex = this.buffers.createTexture({
          label: 'capture-ids',
          size: [width, height],
          format: 'r32uint',
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        depthTex = this.buffers.createTexture({
          label: 'capture-linear-depth',
          size: [width, height],
          format: 'r32float',
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });

        const idBindGroup = device.createBindGroup({
          layout: this.idDepthBindGroupLayout!,
          entries: [
            { binding: 0, resource: { buffer: this.captureUniformBuffer! } },
            { binding: 1, resource: { buffer: this.modelBuffer } },
            { binding: 2, resource: { buffer: this.instanceIdBuffer } },
          ],
        });

        const pass = encoder.beginRenderPass({
          colorAttachments: [
            { view: idTex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
            { view: depthTex.createView(), clearValue: { r: 1, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' },
          ],
          depthStencilAttachment: { view: captureDepthTex.createView(), depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'discard' },
        });
        pass.setPipeline(this.idDepthPipeline!);
        pass.setBindGroup(0, idBindGroup);
        this.drawPbrMeshes(pass, counts, offsets);
        pass.end();

        if (needIds) {
          idReadBuf = this.buffers.createBuffer({
            label: 'capture-ids-readback',
            size: readbackSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          });
          encoder.copyTextureToBuffer({ texture: idTex }, { buffer: idReadBuf, bytesPerRow }, { width, height });
        }
        if (needDepth) {
          depthReadBuf = this.buffers.createBuffer({
            label: 'capture-depth-readback',
            size: readbackSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
          });
          encoder.copyTextureToBuffer({ texture: depthTex }, { buffer: depthReadBuf, bytesPerRow }, { width, height });
        }
      }

      device.queue.submit([encoder.finish()]);

      let rgbPixels: Uint8ClampedArray | null = null;
      let objectIds: Uint32Array | null = null;
      let depth: Float32Array | null = null;

      if (rgbReadBuf) {
        await rgbReadBuf.mapAsync(GPUMapMode.READ);
        const mapped = new Uint8Array(rgbReadBuf.getMappedRange());
        rgbPixels = new Uint8ClampedArray(width * height * 4);
        for (let y = 0; y < height; y++) {
          const srcRow = y * bytesPerRow;
          rgbPixels.set(mapped.subarray(srcRow, srcRow + width * 4), y * width * 4);
        }
        rgbReadBuf.unmap();
      }

      if (idReadBuf) {
        await idReadBuf.mapAsync(GPUMapMode.READ);
        const mapped = new Uint32Array(idReadBuf.getMappedRange());
        const u32Stride = bytesPerRow / 4;
        objectIds = new Uint32Array(width * height);
        for (let y = 0; y < height; y++) {
          const srcRow = y * u32Stride;
          objectIds.set(mapped.subarray(srcRow, srcRow + width), y * width);
        }
        idReadBuf.unmap();
      }

      if (depthReadBuf) {
        await depthReadBuf.mapAsync(GPUMapMode.READ);
        const mapped = new Float32Array(depthReadBuf.getMappedRange());
        const f32Stride = bytesPerRow / 4;
        depth = new Float32Array(width * height);
        for (let y = 0; y < height; y++) {
          const srcRow = y * f32Stride;
          depth.set(mapped.subarray(srcRow, srcRow + width), y * width);
        }
        depthReadBuf.unmap();
      }

      return { width, height, rgbPixels, objectIds, depth };
    } finally {
      this.buffers.destroyTexture(captureDepthTex);
      if (rgbTex) this.buffers.destroyTexture(rgbTex);
      if (idTex) this.buffers.destroyTexture(idTex);
      if (depthTex) this.buffers.destroyTexture(depthTex);
      if (rgbReadBuf) this.buffers.destroyBuffer(rgbReadBuf);
      if (idReadBuf) this.buffers.destroyBuffer(idReadBuf);
      if (depthReadBuf) this.buffers.destroyBuffer(depthReadBuf);
    }
  }

  dispose(): void {
    this.buffers.dispose();
  }
}
