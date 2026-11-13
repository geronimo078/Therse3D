import type { Renderer, SnapshotLayer } from '../gpu/Renderer';
import type { Camera } from '../controls/OrbitControls';
import type { Inspector, SceneDescription } from './Inspector';

export interface SnapshotOptions {
  layers?: readonly SnapshotLayer[];
  resolution?: [number, number];
}

export interface SnapshotResult {
  width: number;
  height: number;
  rgb: ImageBitmap | null;
  rgbPixels: Uint8ClampedArray | null;
  annotated: ImageBitmap | null;
  annotatedDataUrl: string | null;
  depth: Float32Array | null;
  objectIds: Uint32Array | null;
}

/**
 * Renders RGB, Annotated Bounding-Box Overlay, Depth Map, and Object-ID Masks via MRT.
 */
export class Snapshot {
  private readonly renderer: Renderer;

  constructor(renderer: Renderer) {
    this.renderer = renderer;
  }

  async capture(
    options: SnapshotOptions = {},
    camera?: Camera,
    inspector?: Inspector,
    resolveName?: (entityId: number) => string | undefined,
  ): Promise<SnapshotResult> {
    const layers = options.layers ?? ['rgb', 'annotated', 'depth', 'ids'];
    const canvas = this.renderer.gpu.canvas;
    const width = Math.max(1, Math.floor(options.resolution?.[0] ?? canvas.width));
    const height = Math.max(1, Math.floor(options.resolution?.[1] ?? canvas.height));

    const pass = await this.renderer.renderCapturePass(width, height, camera, layers);

    let rgb: ImageBitmap | null = null;
    let annotated: ImageBitmap | null = null;
    let annotatedDataUrl: string | null = null;

    if (pass.rgbPixels && typeof ImageData !== 'undefined' && typeof createImageBitmap === 'function') {
      const imgData = new ImageData(
        new Uint8ClampedArray(pass.rgbPixels),
        width,
        height,
      );
      if (layers.includes('rgb')) {
        rgb = await createImageBitmap(imgData);
      }

      if (layers.includes('annotated') && typeof document !== 'undefined') {
        const overlayCanvas = document.createElement('canvas');
        overlayCanvas.width = width;
        overlayCanvas.height = height;
        const ctx = overlayCanvas.getContext('2d');
        if (ctx) {
          ctx.putImageData(imgData, 0, 0);

          if (inspector && camera) {
            const desc: SceneDescription = inspector.describe({
              viewProjMatrix: camera.updateMatrices(width / height),
              ...(resolveName ? { resolveName } : {}),
            });
            this.drawAnnotations(ctx, width, height, desc);
          }

          annotatedDataUrl = overlayCanvas.toDataURL('image/png');
          annotated = await createImageBitmap(overlayCanvas);
        }
      }
    }

    return {
      width,
      height,
      rgb,
      rgbPixels: layers.includes('rgb') ? pass.rgbPixels : null,
      annotated,
      annotatedDataUrl,
      depth: layers.includes('depth') ? pass.depth : null,
      objectIds: layers.includes('ids') ? pass.objectIds : null,
    };
  }

  private drawAnnotations(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    desc: SceneDescription,
  ): void {
    ctx.lineWidth = 2;
    ctx.font = '12px monospace';

    for (const entity of desc.entities) {
      if (!entity.visible) continue;
      const [xMin, yMin, xMax, yMax] = entity.screenBox;
      const x = xMin * width;
      const y = yMin * height;
      const w = Math.max(2, (xMax - xMin) * width);
      const h = Math.max(2, (yMax - yMin) * height);

      ctx.strokeStyle = '#00e5ff';
      ctx.strokeRect(x, y, w, h);

      const label = `#${entity.id}`;
      const textWidth = ctx.measureText(label).width;
      const tagY = Math.max(0, y - 18);
      ctx.fillStyle = 'rgba(0, 20, 35, 0.8)';
      ctx.fillRect(x, tagY, textWidth + 8, 16);
      ctx.fillStyle = '#00e5ff';
      ctx.fillText(label, x + 4, tagY + 12);
    }
  }
}
