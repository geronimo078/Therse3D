import { describe, expect, it, vi } from 'vitest';
import { Schema, THERSE3D_SCENE_SCHEMA, THERSE3D_TOOL_DEFINITIONS } from './Schema';
import { McpBridge } from './McpBridge';
import { GltfLoader } from '../entities/GltfLoader';
import type { Engine } from '../core/Engine';

function buildMinimalTriangleGlb(): ArrayBuffer {
  // 3 vertices * 3 floats (pos) = 36 bytes + 3 * 2 bytes (u16 indices) = 6 bytes -> padded to 44 bytes
  const bin = new ArrayBuffer(44);
  const f32 = new Float32Array(bin, 0, 9);
  f32.set([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const u16 = new Uint16Array(bin, 36, 3);
  u16.set([0, 1, 2]);

  const jsonObj = {
    asset: { version: '2.0' },
    meshes: [
      {
        name: 'tri',
        primitives: [{ attributes: { POSITION: 0 }, indices: 1, material: 0 }],
      },
    ],
    materials: [
      {
        pbrMetallicRoughness: {
          baseColorFactor: [1, 0.5, 0.25, 1],
          metallicFactor: 0.4,
          roughnessFactor: 0.3,
        },
      },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5123, count: 3, type: 'SCALAR' },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36 },
      { buffer: 0, byteOffset: 36, byteLength: 6 },
    ],
  };

  const rawJson = new TextEncoder().encode(JSON.stringify(jsonObj));
  const paddedJsonLen = (rawJson.byteLength + 3) & ~3;
  const totalLen = 12 + 8 + paddedJsonLen + 8 + bin.byteLength;
  const glb = new ArrayBuffer(totalLen);
  const dv = new DataView(glb);
  const bytes = new Uint8Array(glb);

  // Header
  dv.setUint32(0, 0x46546c67, true); // glTF
  dv.setUint32(4, 2, true);
  dv.setUint32(8, totalLen, true);

  // JSON chunk
  dv.setUint32(12, paddedJsonLen, true);
  dv.setUint32(16, 0x4e4f534a, true);
  bytes.set(rawJson, 20);
  for (let i = rawJson.byteLength; i < paddedJsonLen; i++) {
    bytes[20 + i] = 0x20;
  }

  // BIN chunk
  const binHeaderOffset = 20 + paddedJsonLen;
  dv.setUint32(binHeaderOffset, bin.byteLength, true);
  dv.setUint32(binHeaderOffset + 4, 0x004e4942, true);
  bytes.set(new Uint8Array(bin), binHeaderOffset + 8);

  return glb;
}

describe('Schema & PromptContext (Step 6)', () => {
  it('exports scene schema, 4 tool definitions, and compact prompt context', () => {
    expect(THERSE3D_SCENE_SCHEMA.required).toEqual(['entities']);
    expect(THERSE3D_TOOL_DEFINITIONS.map((t) => t.name)).toEqual([
      'therse_sync_scene',
      'therse_inspect',
      'therse_snapshot',
      'therse_camera_focus',
    ]);

    const prompt = Schema.getPromptContext();
    expect(prompt).toContain('therse3d');
    expect(prompt).toContain('oak-wood');
    expect(prompt).toContain('therse_sync_scene');
  });
});

describe('McpBridge (Step 6)', () => {
  it('dispatches all 4 standardized agent tools cleanly', async () => {
    const mockEngine = {
      sync: vi.fn().mockReturnValue({ applied: 1, added: 1, updated: 0, removed: 0, warnings: [] }),
      describe: vi.fn().mockReturnValue({ entityCount: 1, entities: [], warnings: [] }),
      snapshot: vi.fn().mockResolvedValue({ width: 800, height: 600, annotatedDataUrl: 'data:image/png;base64,abc' }),
      focusCamera: vi.fn(),
    } as unknown as Engine;

    const bridge = new McpBridge(mockEngine);

    const syncRes = (await bridge.handleToolCall('therse_sync_scene', { entities: [] })) as { report: { applied: number } };
    expect(syncRes.report.applied).toBe(1);

    await bridge.handleToolCall('therse_inspect');
    expect(mockEngine.describe).toHaveBeenCalled();

    const snapRes = (await bridge.handleToolCall('therse_snapshot')) as { annotatedDataUrl: string };
    expect(snapRes.annotatedDataUrl).toContain('data:image/png');

    await bridge.handleToolCall('therse_camera_focus', { target: 'orb', view: 'isometric', distance: 3 });
    expect(mockEngine.focusCamera).toHaveBeenCalledWith({ target: 'orb', angle: 'isometric', distance: 3 });

    const unknownRes = (await bridge.handleToolCall('unknown_tool')) as { error: string };
    expect(unknownRes.error).toContain('Unknown tool');
  });
});

describe('GltfLoader (Step 6)', () => {
  it('parses a valid binary .glb mesh and PBR material', () => {
    const loader = new GltfLoader();
    const result = loader.parseGlb(buildMinimalTriangleGlb(), 'tri.glb');

    expect(result.warnings).toEqual([]);
    expect(result.meshCount).toBe(1);
    const mesh = result.meshes[0]!;
    expect(mesh.name).toBe('tri');
    expect(mesh.vertices.length).toBe(3 * 8);
    expect(Array.from(mesh.indices)).toEqual([0, 1, 2]);
    expect(mesh.material.color[1]).toBeCloseTo(0.5);
    expect(mesh.material.metalness).toBeCloseTo(0.4);
  });

  it('returns auto-healing warnings on invalid .glb input', () => {
    const loader = new GltfLoader();
    const res = loader.parseGlb(new ArrayBuffer(4), 'bad.glb');
    expect(res.meshCount).toBe(0);
    expect(res.warnings.length).toBe(1);
  });
});

