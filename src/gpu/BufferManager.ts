/**
 * GPU-Buffer & GPU-Texture Resource-Manager mit deterministischer Memory-Freigabe.
 */
export class BufferManager {
  private readonly device: GPUDevice;
  private readonly buffers = new Set<GPUBuffer>();
  private readonly textures = new Set<GPUTexture>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
    const buffer = this.device.createBuffer(descriptor);
    this.buffers.add(buffer);
    return buffer;
  }

  createStorageBuffer(byteLength: number, label?: string): GPUBuffer {
    const descriptor: GPUBufferDescriptor = {
      size: Math.max(16, byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    };
    if (label !== undefined) {
      descriptor.label = label;
    }
    return this.createBuffer(descriptor);
  }

  createUniformBuffer(byteLength: number, label?: string): GPUBuffer {
    const descriptor: GPUBufferDescriptor = {
      size: Math.max(16, byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    };
    if (label !== undefined) {
      descriptor.label = label;
    }
    return this.createBuffer(descriptor);
  }

  createVertexBuffer(data: Float32Array, label?: string): GPUBuffer {
    const descriptor: GPUBufferDescriptor = {
      size: Math.max(16, data.byteLength),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    };
    if (label !== undefined) {
      descriptor.label = label;
    }
    const buffer = this.createBuffer(descriptor);
    this.writeBuffer(buffer, data);
    return buffer;
  }

  createIndexBuffer(data: Uint16Array | Uint32Array, label?: string): GPUBuffer {
    const alignedSize = Math.max(16, (data.byteLength + 3) & ~3);
    const descriptor: GPUBufferDescriptor = {
      size: alignedSize,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    };
    if (label !== undefined) {
      descriptor.label = label;
    }
    const buffer = this.createBuffer(descriptor);
    this.writeBuffer(buffer, data);
    return buffer;
  }

  createTexture(descriptor: GPUTextureDescriptor): GPUTexture {
    const texture = this.device.createTexture(descriptor);
    this.textures.add(texture);
    return texture;
  }

  writeBuffer(
    buffer: GPUBuffer,
    data: ArrayBufferView | ArrayBuffer,
    dstOffset = 0,
    dataOffset?: number,
    size?: number,
  ): void {
    const source = data as GPUAllowSharedBufferSource;
    if (dataOffset !== undefined && size !== undefined) {
      this.device.queue.writeBuffer(buffer, dstOffset, source, dataOffset, size);
    } else if (dataOffset !== undefined) {
      this.device.queue.writeBuffer(buffer, dstOffset, source, dataOffset);
    } else {
      this.device.queue.writeBuffer(buffer, dstOffset, source);
    }
  }

  destroyBuffer(buffer: GPUBuffer): void {
    if (this.buffers.delete(buffer)) {
      buffer.destroy();
    }
  }

  destroyTexture(texture: GPUTexture): void {
    if (this.textures.delete(texture)) {
      texture.destroy();
    }
  }

  get trackedBufferCount(): number {
    return this.buffers.size;
  }

  get trackedTextureCount(): number {
    return this.textures.size;
  }

  dispose(): void {
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
    this.buffers.clear();
    for (const texture of this.textures) {
      texture.destroy();
    }
    this.textures.clear();
  }
}
