export interface GPUDeviceInitOptions {
  autoResize?: boolean;
  onDeviceLost?: (info: GPUDeviceLostInfo) => void;
}

/**
 * WebGPU Adapter/Device-Initialisierung, Feature-Checks, ResizeObserver & device.lost-Handling.
 */
export class GPUDeviceContext {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly canvas: HTMLCanvasElement;
  readonly context: GPUCanvasContext;
  readonly format: GPUTextureFormat;

  isLost = false;
  private resizeObserver: ResizeObserver | null = null;

  private constructor(
    adapter: GPUAdapter,
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    options: GPUDeviceInitOptions,
  ) {
    this.adapter = adapter;
    this.device = device;
    this.canvas = canvas;
    this.context = context;
    this.format = format;

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

    let adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
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

    return new GPUDeviceContext(adapter, device, canvas, context, format, options);
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
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
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
