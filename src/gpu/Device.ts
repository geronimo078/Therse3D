export type QualityMode = 'auto' | 'high' | 'low';

export interface GPUDeviceInitOptions {
  autoResize?: boolean;
  quality?: QualityMode;
  maxDpr?: number;
  onDeviceLost?: (info: GPUDeviceLostInfo) => void;
}

/**
 * WebGPU Adapter/Device-Initialisierung, Feature-Checks, Low-Spec-Erkennung & device.lost-Handling.
 */
export class GPUDeviceContext {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly canvas: HTMLCanvasElement;
  readonly context: GPUCanvasContext;
  readonly format: GPUTextureFormat;
  readonly isLowSpec: boolean;
  readonly maxDpr: number;

  isLost = false;
  private resizeObserver: ResizeObserver | null = null;

  private constructor(
    adapter: GPUAdapter,
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    isLowSpec: boolean,
    options: GPUDeviceInitOptions,
  ) {
    this.adapter = adapter;
    this.device = device;
    this.canvas = canvas;
    this.context = context;
    this.format = format;
    this.isLowSpec = isLowSpec;
    this.maxDpr = options.maxDpr ?? (isLowSpec ? 1 : 2);

    this.setupDeviceLost(options.onDeviceLost);
    if (options.autoResize ?? true) {
      this.setupResize();
    }
  }

  static async init(
    canvas: HTMLCanvasElement,
    options: GPUDeviceInitOptions = {},
  ): Promise<GPUDeviceContext> {
    if (typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu) {
      const hint =
        typeof window !== 'undefined' && !window.isSecureContext
          ? 'WebGPU requires a secure context (http://localhost or https://).'
          : 'WebGPU is not enabled in this browser (try Chrome/Edge 113+ or enable chrome://flags/#enable-unsafe-webgpu).';
      throw new Error(`WebGPU unavailable: ${hint}`);
    }

    const requestedQuality = options.quality ?? 'auto';
    let adapter = await navigator.gpu.requestAdapter({
      powerPreference: requestedQuality === 'low' ? 'low-power' : 'high-performance',
    });
    if (!adapter) {
      adapter = await navigator.gpu.requestAdapter();
    }
    if (!adapter) {
      adapter = await navigator.gpu.requestAdapter({ forceFallbackAdapter: true });
    }
    if (!adapter) {
      throw new Error(
        'No WebGPU adapter found. Check chrome://gpu (or edge://gpu) -> "WebGPU" status, ensure Hardware Acceleration is enabled, or enable chrome://flags/#enable-unsafe-webgpu.',
      );
    }

    const device = await adapter.requestDevice();
    device.addEventListener('uncapturederror', (event) => {
      console.error('therse3d WebGPU uncaptured error:', event.error);
    });

    const isLowSpec =
      requestedQuality === 'low' ||
      (requestedQuality === 'auto' && detectLowSpecAdapter(adapter, device));

    const context = canvas.getContext('webgpu');
    if (!context) {
      throw new Error('Failed to acquire WebGPU canvas context.');
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format,
      alphaMode: 'premultiplied',
    });

    return new GPUDeviceContext(adapter, device, canvas, context, format, isLowSpec, options);
  }

  private setupDeviceLost(onDeviceLost?: (info: GPUDeviceLostInfo) => void): void {
    void this.device.lost.then((info) => {
      this.isLost = true;
      if (onDeviceLost) {
        onDeviceLost(info);
      } else if (info.reason !== 'destroyed') {
        console.warn(`therse3d WebGPU device lost (${info.reason}): ${info.message}`);
      }
    });
  }

  private setupResize(): void {
    const maxDim = this.device.limits.maxTextureDimension2D || 8192;
    const updateSize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, this.maxDpr);
      const width = Math.min(maxDim, Math.max(1, Math.floor(this.canvas.clientWidth * dpr)));
      const height = Math.min(maxDim, Math.max(1, Math.floor(this.canvas.clientHeight * dpr)));
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
      }
    };

    updateSize();
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(updateSize);
      this.resizeObserver.observe(this.canvas);
    }
  }

  dispose(): void {
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.device.destroy();
  }
}

function detectLowSpecAdapter(adapter: GPUAdapter, device: GPUDevice): boolean {
  const anyAdapter = adapter as GPUAdapter & {
    isFallbackAdapter?: boolean;
    info?: { architecture?: string; description?: string; vendor?: string };
  };
  if (anyAdapter.isFallbackAdapter) return true;
  if (device.limits.maxTextureDimension2D < 8192) return true;
  if (device.limits.maxStorageBufferBindingSize < 134_217_728) return true;

  const vendor = (anyAdapter.info?.vendor ?? '').toLowerCase();
  const arch = (anyAdapter.info?.architecture ?? '').toLowerCase();
  const desc = (anyAdapter.info?.description ?? '').toLowerCase();

  if (desc.includes('swiftshader') || desc.includes('warp') || desc.includes('llvmpipe')) {
    return true;
  }
  // Older AMD GCN / integrated APU GPUs (e.g. Radeon Vega / Graphics / R5/R7)
  if (arch.includes('gcn')) return true;
  if (
    vendor.includes('amd') &&
    (desc.includes('vega') || (desc.includes('radeon') && desc.includes('graphics') && !desc.includes('rx ')))
  ) {
    return true;
  }
  // Older Intel integrated GPUs (HD / UHD Graphics)
  if (
    vendor.includes('intel') &&
    (arch.includes('gen-8') || arch.includes('gen-9') || desc.includes('uhd graphics') || desc.includes('hd graphics'))
  ) {
    return true;
  }
  return false;
}
