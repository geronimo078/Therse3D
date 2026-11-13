import type { GPUDeviceContext } from './Device';
import { GeometryTypeId, type WorldStore } from '../core/WorldStore';
import { BufferManager } from './BufferManager';
import {
  createBoxGeometry,
  createCylinderGeometry,
  createPlaneGeometry,
  createSphereGeometry,
  type GeometryMeshData,
} from '../entities/Primitives';
import type { EnginePreset } from '../core/Engine';
import type { Camera } from '../controls/OrbitControls';
import pbrShaderCode from './shaders/pbr.wgsl?raw';
import idDepthShaderCode from './shaders/id_depth.wgsl?raw';

interface GeometryGpuMesh {
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  indexCount: number;
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
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private offscreenPbrPipeline: GPURenderPipeline | null = null;
  private idDepthPipeline: GPURenderPipeline | null = null;
  private idDepthBindGroupLayout: GPUBindGroupLayout | null = null;
  private captureUniformBuffer: GPUBuffer | null = null;
  private readonly captureUniformScratch = new Float32Array(24); // mat4(16) + camPos(4) + nearFar(4)
  private readonly meshes = new Map<GeometryTypeId, GeometryGpuMesh>();

  private readonly uniformBuffer: GPUBuffer;
  // mat4(16) + 5*vec4(20) + counts(4) + 8*DynamicLight(64) + 16*vec4 shadowCasters(64) = 168 floats
  private readonly uniformScratch = new Float32Array(168);

  private allocatedCapacity = 0;
  private modelBuffer!: GPUBuffer;
  private colorBuffer!: GPUBuffer;
  private pbrBuffer!: GPUBuffer;
  private instanceIdBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;

  // Pre-allocated CPU scratch buffers for zero-allocation instancing bucketing
  private readonly bucketCounts = new Uint32Array(8);
  private readonly bucketOffsets = new Uint32Array(8);
  private readonly bucketCursors = new Uint32Array(8);
  private packedInstanceIds!: Uint32Array;

  // Render target textures (recreated on canvas resize)
  private msaaColorTexture: GPUTexture | null = null;
  private msaaColorView: GPUTextureView | null = null;
  private depthTexture: GPUTexture | null = null;
  private depthView: GPUTextureView | null = null;
  private rtWidth = 0;
  private rtHeight = 0;

  constructor(gpu: GPUDeviceContext, store: WorldStore, preset: EnginePreset = 'studio') {
    this.gpu = gpu;
    this.store = store;
    this.preset = preset;
    this.buffers = new BufferManager(gpu.device);
    this.pbrShaderSource = pbrShaderCode;
    this.idDepthShaderSource = idDepthShaderCode;

    this.uniformBuffer = this.buffers.createUniformBuffer(this.uniformScratch.byteLength, 'scene-uniforms');

    this.bindGroupLayout = gpu.device.createBindGroupLayout({
      label: 'pbr-bind-group-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
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

    this.uploadPrimitiveMesh(GeometryTypeId.Box, createBoxGeometry(), 'box');
    this.uploadPrimitiveMesh(GeometryTypeId.Sphere, createSphereGeometry(), 'sphere');
    this.uploadPrimitiveMesh(GeometryTypeId.Cylinder, createCylinderGeometry(), 'cylinder');
    this.uploadPrimitiveMesh(GeometryTypeId.Plane, createPlaneGeometry(), 'plane');

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
    });
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
      ],
    });
  }

  private ensureRenderTargets(width: number, height: number): void {
    if (this.rtWidth === width && this.rtHeight === height && this.depthView && this.msaaColorView) {
      return;
    }

    if (this.msaaColorTexture) this.buffers.destroyTexture(this.msaaColorTexture);
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

    this.depthTexture = this.buffers.createTexture({
      label: 'depth-24plus',
      size: [width, height],
      sampleCount: MSAA_SAMPLE_COUNT,
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthView = this.depthTexture.createView();
  }

  renderFrame(camera?: Camera): void {
    if (this.gpu.isLost) return;

    const canvas = this.gpu.canvas;
    const width = Math.max(1, canvas.width);
    const height = Math.max(1, canvas.height);

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
    counts.fill(0);

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
        if (gType < 8) {
          counts[gType] = (counts[gType] ?? 0) + 1;
        }

        if (gType === GeometryTypeId.Light && dynamicLightCount < 8) {
          const base = 40 + dynamicLightCount * 8;
          const mOff = id * 16;
          const cOff = id * 4;
          u[base + 0] = mats[mOff + 12] ?? 0;
          u[base + 1] = mats[mOff + 13] ?? 0;
          u[base + 2] = mats[mOff + 14] ?? 0;
          u[base + 3] = lParams[cOff] ?? 0; // kind: 0 = point, 1 = directional
          u[base + 4] = cols[cOff] ?? 1;
          u[base + 5] = cols[cOff + 1] ?? 1;
          u[base + 6] = cols[cOff + 2] ?? 1;
          u[base + 7] = lParams[cOff + 1] ?? 1; // intensity
          dynamicLightCount++;
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
      for (let t = 0; t < 8; t++) {
        offsets[t] = totalVisible;
        cursors[t] = totalVisible;
        totalVisible += counts[t] ?? 0;
      }

      if (totalVisible > 0) {
        const packed = this.packedInstanceIds;
        for (let id = 0; id < n; id++) {
          if (alive[id] === 0 || vis[id] === 0) continue;
          const gType = gTypes[id] ?? 0;
          if (gType < 8) {
            const dst = cursors[gType] ?? 0;
            packed[dst] = id;
            cursors[gType] = dst + 1;
          }
        }
        this.buffers.writeBuffer(this.instanceIdBuffer, packed, 0, 0, totalVisible);
      }
    }

    u[36] = dynamicLightCount;
    u[37] = shadowCasterCount;
    u[38] = 0;
    u[39] = 0;
    this.buffers.writeBuffer(this.uniformBuffer, u);

    const encoder = this.gpu.device.createCommandEncoder();
    const resolveView = this.gpu.context.getCurrentTexture().createView();

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.msaaColorView!,
          resolveTarget: resolveView,
          clearValue: presetCfg.clearColor,
          loadOp: 'clear',
          storeOp: 'discard',
        },
      ],
      depthStencilAttachment: {
        view: this.depthView!,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'discard',
      },
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);

    for (let i = 0; i < MESH_TYPES.length; i++) {
      const gType = MESH_TYPES[i]!;
      const count = counts[gType] ?? 0;
      if (count === 0) continue;
      const mesh = this.meshes.get(gType);
      if (!mesh) continue;

      pass.setVertexBuffer(0, mesh.vertexBuffer);
      pass.setIndexBuffer(mesh.indexBuffer, 'uint16');
      pass.drawIndexed(mesh.indexCount, count, 0, 0, offsets[gType] ?? 0);
    }

    pass.end();
    this.gpu.device.queue.submit([encoder.finish()]);
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
    counts.fill(0);

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
        if (gType < 8) counts[gType] = (counts[gType] ?? 0) + 1;

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
      for (let t = 0; t < 8; t++) {
        offsets[t] = totalVisible;
        cursors[t] = totalVisible;
        totalVisible += counts[t] ?? 0;
      }

      if (totalVisible > 0) {
        const packed = this.packedInstanceIds;
        for (let id = 0; id < n; id++) {
          if (alive[id] === 0 || vis[id] === 0) continue;
          const gType = gTypes[id] ?? 0;
          if (gType < 8) {
            const dst = cursors[gType] ?? 0;
            packed[dst] = id;
            cursors[gType] = dst + 1;
          }
        }
        this.buffers.writeBuffer(this.instanceIdBuffer, packed, 0, 0, totalVisible);
      }
    }

    u[36] = dynamicLightCount;
    u[37] = shadowCasterCount;
    u[38] = 0;
    u[39] = 0;
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
        for (let i = 0; i < MESH_TYPES.length; i++) {
          const gType = MESH_TYPES[i]!;
          const count = counts[gType] ?? 0;
          if (count === 0) continue;
          const mesh = this.meshes.get(gType);
          if (!mesh) continue;
          pass.setVertexBuffer(0, mesh.vertexBuffer);
          pass.setIndexBuffer(mesh.indexBuffer, 'uint16');
          pass.drawIndexed(mesh.indexCount, count, 0, 0, offsets[gType] ?? 0);
        }
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
        for (let i = 0; i < MESH_TYPES.length; i++) {
          const gType = MESH_TYPES[i]!;
          const count = counts[gType] ?? 0;
          if (count === 0) continue;
          const mesh = this.meshes.get(gType);
          if (!mesh) continue;
          pass.setVertexBuffer(0, mesh.vertexBuffer);
          pass.setIndexBuffer(mesh.indexBuffer, 'uint16');
          pass.drawIndexed(mesh.indexCount, count, 0, 0, offsets[gType] ?? 0);
        }
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
