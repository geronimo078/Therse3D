import type { Engine } from '../core/Engine';
import type { CameraAnglePreset, SceneSyncState } from '../core/SceneSync';
import type { SnapshotOptions } from './Snapshot';

export interface McpBridgeOptions {
  url?: string;
  channel?: string;
}

export type McpToolName =
  | 'therse_sync_scene'
  | 'therse_inspect'
  | 'therse_snapshot'
  | 'therse_camera_focus';

export interface CameraFocusArgs {
  target?: string;
  entityId?: string;
  view?: CameraAnglePreset;
  distance?: number;
}

export interface McpToolRequest {
  id?: string | number;
  tool?: McpToolName | string;
  args?: unknown;
  /** Legacy direct sync payload support */
  type?: string;
  state?: SceneSyncState;
}

/**
 * Adapter layer (WebSocket / BroadcastChannel / direct tool dispatch) for external AI agents.
 */
export class McpBridge {
  private readonly engine: Engine;
  private socket: WebSocket | null = null;
  private broadcast: BroadcastChannel | null = null;

  constructor(engine: Engine) {
    this.engine = engine;
  }

  connect(options: McpBridgeOptions = {}): void {
    if (options.url && typeof WebSocket !== 'undefined') {
      this.socket = new WebSocket(options.url);
      this.socket.addEventListener('message', (event) => {
        void this.onRawMessage(String(event.data), (reply) => {
          if (this.socket?.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(reply));
          }
        });
      });
    }

    if (options.channel && typeof BroadcastChannel !== 'undefined') {
      this.broadcast = new BroadcastChannel(options.channel);
      this.broadcast.addEventListener('message', (event) => {
        void this.onRawMessage(event.data, (reply) => {
          this.broadcast?.postMessage(reply);
        });
      });
    }
  }

  private async onRawMessage(
    raw: unknown,
    sendReply: (payload: unknown) => void,
  ): Promise<void> {
    let req: McpToolRequest;
    try {
      req = typeof raw === 'string' ? (JSON.parse(raw) as McpToolRequest) : (raw as McpToolRequest);
    } catch {
      return;
    }

    if (req.type === 'sync' && req.state) {
      const report = this.engine.sync(req.state);
      sendReply({ id: req.id, result: report });
      return;
    }

    if (req.tool) {
      const result = await this.handleToolCall(req.tool, req.args);
      sendReply({ id: req.id, tool: req.tool, result });
    }
  }

  async handleToolCall(tool: McpToolName | string, args: unknown = {}): Promise<unknown> {
    switch (tool) {
      case 'therse_sync_scene': {
        const report = this.engine.sync(args as SceneSyncState);
        const scene = this.engine.describe();
        return { report, scene };
      }

      case 'therse_inspect': {
        return this.engine.describe();
      }

      case 'therse_snapshot': {
        const opts = (args && typeof args === 'object' ? args : {}) as SnapshotOptions;
        const snap = await this.engine.snapshot({
          layers: opts.layers ?? ['rgb', 'annotated'],
          ...(opts.resolution ? { resolution: opts.resolution } : {}),
        });
        return {
          width: snap.width,
          height: snap.height,
          annotatedDataUrl: snap.annotatedDataUrl,
        };
      }

      case 'therse_camera_focus': {
        const focusArgs = (args && typeof args === 'object' ? args : {}) as CameraFocusArgs;
        const targetName = focusArgs.target ?? focusArgs.entityId;
        this.engine.focusCamera({
          ...(targetName !== undefined ? { target: targetName } : {}),
          ...(focusArgs.view !== undefined ? { angle: focusArgs.view } : {}),
          ...(focusArgs.distance !== undefined ? { distance: focusArgs.distance } : {}),
        });
        return { ok: true, scene: this.engine.describe() };
      }

      default:
        return { error: `Unknown tool "${tool}".` };
    }
  }

  disconnect(): void {
    this.socket?.close();
    this.socket = null;
    this.broadcast?.close();
    this.broadcast = null;
  }
}
