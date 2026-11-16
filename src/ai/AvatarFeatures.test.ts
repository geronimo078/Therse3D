import { describe, expect, it } from 'vitest';
import { GltfLoader } from '../entities/GltfLoader';
import { GeometryTypeId, LightKindId, WorldStore } from '../core/WorldStore';
import { SceneSync } from '../core/SceneSync';
import { Sprite, Text, createSpriteGeometry } from '../entities/Primitives';
import { Inspector } from './Inspector';

function buildGlbWithSkinAndTexture(): ArrayBuffer {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const joints = new Uint8Array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
  const weights = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
  const ibm = new Float32Array([
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
  ]);
  const fakePng = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

  const chunks = [positions, joints, weights, ibm, fakePng];
  const rawBytes = chunks.reduce((s, c) => s + c.byteLength, 0);
  const binPaddedLen = (rawBytes + 3) & ~3;
  const binBuf = new Uint8Array(binPaddedLen);
  const offsets: number[] = [];
  let off = 0;
  for (const c of chunks) {
    offsets.push(off);
    binBuf.set(new Uint8Array(c.buffer, c.byteOffset, c.byteLength), off);
    off += c.byteLength;
  }

  const gltfJson = {
    asset: { version: '2.0' },
    meshes: [
      {
        name: 'RiggedAvatar',
        primitives: [
          {
            attributes: { POSITION: 0, JOINTS_0: 1, WEIGHTS_0: 2 },
            material: 0,
          },
        ],
      },
    ],
    materials: [
      {
        pbrMetallicRoughness: {
          baseColorTexture: { index: 0 },
        },
      },
    ],
    textures: [{ source: 0 }],
    images: [{ bufferView: 4, mimeType: 'image/png' }],
    nodes: [
      { name: 'Root', mesh: 0, skin: 0 },
      { name: 'Neck', children: [2], translation: [0, 1, 0] },
      { name: 'Head', translation: [0, 0.3, 0] },
    ],
    skins: [{ inverseBindMatrices: 3, joints: [1, 2] }],
    bufferViews: [
      { buffer: 0, byteOffset: offsets[0], byteLength: positions.byteLength },
      { buffer: 0, byteOffset: offsets[1], byteLength: joints.byteLength },
      { buffer: 0, byteOffset: offsets[2], byteLength: weights.byteLength },
      { buffer: 0, byteOffset: offsets[3], byteLength: ibm.byteLength },
      { buffer: 0, byteOffset: offsets[4], byteLength: fakePng.byteLength },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' },
      { bufferView: 1, componentType: 5121, count: 3, type: 'VEC4' },
      { bufferView: 2, componentType: 5126, count: 3, type: 'VEC4' },
      { bufferView: 3, componentType: 5126, count: 2, type: 'MAT4' },
    ],
    buffers: [{ byteLength: binPaddedLen }],
  };

  const jsonBytes = new TextEncoder().encode(JSON.stringify(gltfJson));
  const jsonPaddedLen = (jsonBytes.length + 3) & ~3;
  const jsonBuf = new Uint8Array(jsonPaddedLen);
  jsonBuf.fill(0x20);
  jsonBuf.set(jsonBytes);

  const totalLen = 12 + 8 + jsonPaddedLen + 8 + binPaddedLen;
  const out = new ArrayBuffer(totalLen);
  const dv = new DataView(out);
  const bytes = new Uint8Array(out);
  dv.setUint32(0, 0x46546c67, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, totalLen, true);
  dv.setUint32(12, jsonPaddedLen, true);
  dv.setUint32(16, 0x4e4f534a, true);
  bytes.set(jsonBuf, 20);
  const binOff = 20 + jsonPaddedLen;
  dv.setUint32(binOff, binPaddedLen, true);
  dv.setUint32(binOff + 4, 0x004e4942, true);
  bytes.set(binBuf, binOff + 8);
  return out;
}

describe('glTF Skinning, Embedded Textures, Sprites & SpotLights', () => {
  it('parses skeletal skin joints/weights and embedded baseColorImage from .glb', () => {
    const loader = new GltfLoader();
    const res = loader.parseGlb(buildGlbWithSkinAndTexture(), 'avatar.glb');
    expect(res.warnings).toEqual([]);
    expect(res.meshCount).toBe(1);

    const mesh = res.meshes[0]!;
    expect(mesh.baseColorImage?.mimeType).toBe('image/png');
    expect(mesh.baseColorImage?.data.length).toBe(8);
    expect(mesh.skin).toBeDefined();
    expect(mesh.skin!.joints.map((j) => j.name)).toEqual(['Neck', 'Head']);
    expect(mesh.skin!.joints[1]!.parentIndex).toBe(0);
    expect(mesh.skin!.jointIndices[4]).toBe(1);
    expect(mesh.skin!.jointWeights[0]).toBeCloseTo(1, 5);
  });

  it('supports Sprite/Text descriptors, spotAngle lights, and bone reporting in Inspector', () => {
    const spriteDesc = Sprite({ id: 'badge', text: 'Hello' });
    const textDesc = Text({ id: 'label', text: 'AI' });
    expect(spriteDesc.type).toBe('sprite');
    expect(textDesc.type).toBe('sprite');
    expect(createSpriteGeometry().indices.length).toBe(6);

    const store = new WorldStore(16);
    const sync = new SceneSync(store);
    const res = sync.apply({
      entities: [
        { id: 'tag', type: 'sprite', text: 'Avatar', position: [0, 2, 0] },
        { id: 'spot1', type: 'light', kind: 'spot', spotAngle: 30, position: [0, 3, 2] },
      ],
    });
    expect(res.warnings).toEqual([]);
    const tagId = sync.getEntityId('tag')!;
    const spotId = sync.getEntityId('spot1')!;
    expect(store.geometryTypes[tagId]).toBe(GeometryTypeId.Sprite);
    expect(store.lightParams[spotId * 4]).toBe(LightKindId.Spot);
    expect(store.lightParams[spotId * 4 + 3]).toBeCloseTo(Math.cos(Math.PI / 6), 4);

    const inspector = new Inspector(store);
    const desc = inspector.describe({
      resolveName: (id) => (id === tagId ? 'tag' : 'spot1'),
      resolveBoneNames: (id) => (id === tagId ? ['Neck', 'Head'] : []),
    });
    expect(desc.entities[0]?.bones).toEqual(['Neck', 'Head']);
  });
});
