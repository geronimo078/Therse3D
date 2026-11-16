import { describe, expect, it } from 'vitest';
import { GltfLoader } from '../entities/GltfLoader';
import { DirtyFlag, MAX_MORPH_TARGETS, WorldStore } from '../core/WorldStore';
import { SceneSync } from '../core/SceneSync';
import { Inspector } from './Inspector';

function buildGlbWithMorphTargets(): ArrayBuffer {
  // Single triangle (3 vertices): base positions, normals, and 2 morph targets ('smile', 'blink')
  const positions = new Float32Array([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ]);
  const normals = new Float32Array([
    0, 0, 1,
    0, 0, 1,
    0, 0, 1,
  ]);
  const smilePos = new Float32Array([
    0, 0.2, 0,
    0, 0.3, 0,
    0, 0.1, 0,
  ]);
  const smileNorm = new Float32Array([
    0, 0.1, 0,
    0, 0.1, 0,
    0, 0.1, 0,
  ]);
  const blinkPos = new Float32Array([
    0, -0.4, 0,
    0, -0.4, 0,
    0, -0.4, 0,
  ]);

  const binArrays = [positions, normals, smilePos, smileNorm, blinkPos];
  const binBytes = binArrays.reduce((sum, arr) => sum + arr.byteLength, 0);
  const binPaddedLen = (binBytes + 3) & ~3;
  const binBuf = new Uint8Array(binPaddedLen);
  let byteOffset = 0;
  const offsets: number[] = [];
  for (const arr of binArrays) {
    offsets.push(byteOffset);
    binBuf.set(new Uint8Array(arr.buffer), byteOffset);
    byteOffset += arr.byteLength;
  }

  const gltfJson = {
    asset: { version: '2.0' },
    meshes: [
      {
        name: 'AvatarFace',
        extras: { targetNames: ['smile', 'blink'] },
        primitives: [
          {
            attributes: { POSITION: 0, NORMAL: 1 },
            targets: [
              { POSITION: 2, NORMAL: 3 },
              { POSITION: 4 },
            ],
          },
        ],
      },
    ],
    bufferViews: [
      { buffer: 0, byteOffset: offsets[0], byteLength: positions.byteLength },
      { buffer: 0, byteOffset: offsets[1], byteLength: normals.byteLength },
      { buffer: 0, byteOffset: offsets[2], byteLength: smilePos.byteLength },
      { buffer: 0, byteOffset: offsets[3], byteLength: smileNorm.byteLength },
      { buffer: 0, byteOffset: offsets[4], byteLength: blinkPos.byteLength },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 3, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 4, componentType: 5126, count: 3, type: 'VEC3' },
    ],
    buffers: [{ byteLength: binPaddedLen }],
  };

  const jsonBytesRaw = new TextEncoder().encode(JSON.stringify(gltfJson));
  const jsonPaddedLen = (jsonBytesRaw.length + 3) & ~3;
  const jsonBuf = new Uint8Array(jsonPaddedLen);
  jsonBuf.fill(0x20); // space padding per glTF 2.0 spec
  jsonBuf.set(jsonBytesRaw, 0);

  const totalLen = 12 + 8 + jsonPaddedLen + 8 + binPaddedLen;
  const out = new ArrayBuffer(totalLen);
  const dv = new DataView(out);
  const bytes = new Uint8Array(out);

  dv.setUint32(0, 0x46546c67, true); // 'glTF'
  dv.setUint32(4, 2, true);
  dv.setUint32(8, totalLen, true);

  // JSON chunk
  dv.setUint32(12, jsonPaddedLen, true);
  dv.setUint32(16, 0x4e4f534a, true);
  bytes.set(jsonBuf, 20);

  // BIN chunk
  const binChunkOffset = 20 + jsonPaddedLen;
  dv.setUint32(binChunkOffset, binPaddedLen, true);
  dv.setUint32(binChunkOffset + 4, 0x004e4942, true);
  bytes.set(binBuf, binChunkOffset + 8);

  return out;
}

describe('Morph Targets / Blendshapes for AI Avatars', () => {
  it('parses glTF 2.0 morph targets and targetNames from .glb', () => {
    const loader = new GltfLoader();
    const glb = buildGlbWithMorphTargets();
    const result = loader.parseGlb(glb, 'avatar_face.glb');

    expect(result.warnings).toEqual([]);
    expect(result.meshCount).toBe(1);
    const mesh = result.meshes[0]!;
    expect(mesh.name).toBe('AvatarFace');
    expect(mesh.morphTargets).toHaveLength(2);
    expect(mesh.morphTargets![0]!.name).toBe('smile');
    expect(mesh.morphTargets![0]!.positionDeltas[1]).toBeCloseTo(0.2, 5);
    expect(mesh.morphTargets![0]!.normalDeltas![1]).toBeCloseTo(0.1, 5);
    expect(mesh.morphTargets![1]!.name).toBe('blink');
    expect(mesh.morphTargets![1]!.normalDeltas).toBeNull();
  });

  it('stores, clamps, and marks DirtyFlag.Morph in WorldStore', () => {
    const store = new WorldStore(8);
    const id = store.allocate();
    store.dirtyFlags[id] = DirtyFlag.None;

    store.setMorphWeight(id, 0, 0.75);
    store.setMorphWeight(id, 1, 5.0); // clamped to 2
    store.setMorphWeight(id, 2, Number.NaN); // sanitized to 0

    expect((store.dirtyFlags[id]! & DirtyFlag.Morph) !== 0).toBe(true);
    expect(store.morphWeights[id * MAX_MORPH_TARGETS + 0]).toBeCloseTo(0.75, 5);
    expect(store.morphWeights[id * MAX_MORPH_TARGETS + 1]).toBeCloseTo(2.0, 5);
    expect(store.morphWeights[id * MAX_MORPH_TARGETS + 2]).toBe(0);
  });

  it('syncs named morphWeights, warns on unknown targets, and tweens smoothly', () => {
    const store = new WorldStore(16);
    const sync = new SceneSync(store);
    const targetNames = ['smile', 'blink', 'jawOpen'];
    const hooks = {
      resolveMorphTargetNames: () => targetNames,
    };

    const res1 = sync.apply(
      {
        entities: [
          {
            id: 'face',
            type: 'model',
            morphWeights: { smile: 0.2, unknownExpr: 1.0 },
          },
        ],
      },
      hooks,
    );

    const faceId = sync.getEntityId('face')!;
    expect(res1.warnings.some((w) => w.includes('Unknown morphTarget "unknownExpr"'))).toBe(true);
    expect(store.morphWeights[faceId * MAX_MORPH_TARGETS + 0]).toBeCloseTo(0.2, 5);

    // Patch with transition: 1.0s to smile=1.0, jawOpen=0.6
    sync.patch(
      {
        transition: 1.0,
        upsert: [
          {
            id: 'face',
            type: 'model',
            morphWeights: { smile: 1.0, jawOpen: 0.6 },
          },
        ],
      },
      hooks,
    );

    // At t=0, weight should still be start weight (0.2)
    expect(store.morphWeights[faceId * MAX_MORPH_TARGETS + 0]).toBeCloseTo(0.2, 5);

    // Advance 0.5s -> smoothstep(0.5) = 0.5 -> smile = 0.6, jawOpen = 0.3
    sync.updateTweens(0.5);
    expect(store.morphWeights[faceId * MAX_MORPH_TARGETS + 0]).toBeCloseTo(0.6, 4);
    expect(store.morphWeights[faceId * MAX_MORPH_TARGETS + 2]).toBeCloseTo(0.3, 4);

    // Advance remaining 0.5s -> final weights
    sync.updateTweens(0.5);
    expect(store.morphWeights[faceId * MAX_MORPH_TARGETS + 0]).toBeCloseTo(1.0, 5);
    expect(store.morphWeights[faceId * MAX_MORPH_TARGETS + 2]).toBeCloseTo(0.6, 5);

    // Inspector should report available morphTargets and active non-zero morphWeights
    const inspector = new Inspector(store);
    const desc = inspector.describe({
      resolveName: (id) => (id === faceId ? 'face' : undefined),
      resolveMorphTargetNames: (id) => (id === faceId ? targetNames : undefined),
    });
    expect(desc.entities[0]?.morphTargets).toEqual(['smile', 'blink', 'jawOpen']);
    expect(desc.entities[0]?.morphWeights).toEqual({ smile: 1, jawOpen: 0.6 });
  });
});
