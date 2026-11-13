import type { MaterialDefinition } from './Materials';

export interface GltfParsedMesh {
  name: string;
  vertices: Float32Array;
  indices: Uint32Array;
  material: MaterialDefinition;
}

export interface GltfLoadResult {
  meshCount: number;
  meshes: GltfParsedMesh[];
  warnings: string[];
}

interface GltfJson {
  meshes?: Array<{
    name?: string;
    primitives?: Array<{
      attributes?: Record<string, number>;
      indices?: number;
      material?: number;
    }>;
  }>;
  materials?: Array<{
    pbrMetallicRoughness?: {
      baseColorFactor?: [number, number, number, number];
      metallicFactor?: number;
      roughnessFactor?: number;
    };
  }>;
  accessors?: Array<{
    bufferView?: number;
    byteOffset?: number;
    componentType: number;
    count: number;
    type: string;
  }>;
  bufferViews?: Array<{
    buffer: number;
    byteOffset?: number;
    byteLength: number;
    byteStride?: number;
  }>;
}

const GLB_MAGIC = 0x46546c67; // 'glTF'
const CHUNK_JSON = 0x4e4f534a; // 'JSON'
const CHUNK_BIN = 0x004e4942; // 'BIN\0'

const COMPONENT_UNSIGNED_SHORT = 5123;
const COMPONENT_UNSIGNED_INT = 5125;
const COMPONENT_FLOAT = 5126;

/**
 * Lean Binary glTF 2.0 (.glb) parser with PBR material extraction.
 */
export class GltfLoader {
  async load(url: string): Promise<GltfLoadResult> {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        return {
          meshCount: 0,
          meshes: [],
          warnings: [`Failed to fetch glTF asset "${url}" (${response.status}).`],
        };
      }
      const buffer = await response.arrayBuffer();
      return this.parseGlb(buffer, url);
    } catch (err) {
      return {
        meshCount: 0,
        meshes: [],
        warnings: [`Error loading glTF "${url}": ${err instanceof Error ? err.message : String(err)}`],
      };
    }
  }

  parseGlb(buffer: ArrayBuffer, label = 'memory.glb'): GltfLoadResult {
    const warnings: string[] = [];
    if (buffer.byteLength < 12) {
      warnings.push(`Invalid .glb header for "${label}" (too small).`);
      return { meshCount: 0, meshes: [], warnings };
    }

    const view = new DataView(buffer);
    const magic = view.getUint32(0, true);
    const version = view.getUint32(4, true);
    if (magic !== GLB_MAGIC || version !== 2) {
      warnings.push(`Unsupported glTF/GLB header in "${label}".`);
      return { meshCount: 0, meshes: [], warnings };
    }

    let offset = 12;
    let json: GltfJson | null = null;
    let binChunk: Uint8Array | null = null;

    while (offset + 8 <= buffer.byteLength) {
      const chunkLen = view.getUint32(offset, true);
      const chunkType = view.getUint32(offset + 4, true);
      const chunkStart = offset + 8;
      if (chunkStart + chunkLen > buffer.byteLength) {
        warnings.push(`Truncated GLB chunk in "${label}".`);
        break;
      }

      if (chunkType === CHUNK_JSON) {
        const jsonBytes = new Uint8Array(buffer, chunkStart, chunkLen);
        const text = new TextDecoder().decode(jsonBytes);
        json = JSON.parse(text) as GltfJson;
      } else if (chunkType === CHUNK_BIN) {
        binChunk = new Uint8Array(buffer, chunkStart, chunkLen);
      }

      offset = chunkStart + chunkLen;
    }

    if (!json || !Array.isArray(json.meshes)) {
      warnings.push(`No meshes found in "${label}".`);
      return { meshCount: 0, meshes: [], warnings };
    }

    const meshes: GltfParsedMesh[] = [];
    const accessors = json.accessors ?? [];
    const bufferViews = json.bufferViews ?? [];

    for (let m = 0; m < json.meshes.length; m++) {
      const mesh = json.meshes[m]!;
      const prims = mesh.primitives ?? [];
      for (let p = 0; p < prims.length; p++) {
        const prim = prims[p]!;
        const posIdx = prim.attributes?.['POSITION'];
        if (posIdx === undefined || !binChunk) continue;

        const posData = readFloatAccessor(accessors, bufferViews, binChunk, posIdx, 3);
        if (!posData) continue;

        const vertCount = posData.length / 3;
        const normIdx = prim.attributes?.['NORMAL'];
        const uvIdx = prim.attributes?.['TEXCOORD_0'];
        const normData = normIdx !== undefined ? readFloatAccessor(accessors, bufferViews, binChunk, normIdx, 3) : null;
        const uvData = uvIdx !== undefined ? readFloatAccessor(accessors, bufferViews, binChunk, uvIdx, 2) : null;

        const vertices = new Float32Array(vertCount * 8);
        for (let i = 0; i < vertCount; i++) {
          const dst = i * 8;
          vertices[dst + 0] = posData[i * 3 + 0] ?? 0;
          vertices[dst + 1] = posData[i * 3 + 1] ?? 0;
          vertices[dst + 2] = posData[i * 3 + 2] ?? 0;
          vertices[dst + 3] = normData ? (normData[i * 3 + 0] ?? 0) : 0;
          vertices[dst + 4] = normData ? (normData[i * 3 + 1] ?? 1) : 1;
          vertices[dst + 5] = normData ? (normData[i * 3 + 2] ?? 0) : 0;
          vertices[dst + 6] = uvData ? (uvData[i * 2 + 0] ?? 0) : 0;
          vertices[dst + 7] = uvData ? (uvData[i * 2 + 1] ?? 0) : 0;
        }

        let indices: Uint32Array;
        if (prim.indices !== undefined) {
          indices = readIndexAccessor(accessors, bufferViews, binChunk, prim.indices) ?? sequentialIndices(vertCount);
        } else {
          indices = sequentialIndices(vertCount);
        }

        const matDef = prim.material !== undefined ? json.materials?.[prim.material]?.pbrMetallicRoughness : undefined;
        const color = matDef?.baseColorFactor ?? [0.85, 0.85, 0.88, 1.0];
        const material: MaterialDefinition = {
          color: [color[0] ?? 1, color[1] ?? 1, color[2] ?? 1, color[3] ?? 1],
          metalness: matDef?.metallicFactor ?? 0.0,
          roughness: matDef?.roughnessFactor ?? 0.5,
          emissive: 0.0,
          materialId: 0,
        };

        meshes.push({
          name: mesh.name ?? `mesh_${m}_${p}`,
          vertices,
          indices,
          material,
        });
      }
    }

    return { meshCount: meshes.length, meshes, warnings };
  }
}

function readFloatAccessor(
  accessors: NonNullable<GltfJson['accessors']>,
  bufferViews: NonNullable<GltfJson['bufferViews']>,
  bin: Uint8Array,
  accessorIdx: number,
  components: number,
): Float32Array | null {
  const acc = accessors[accessorIdx];
  if (!acc || acc.bufferView === undefined || acc.componentType !== COMPONENT_FLOAT) return null;
  const bv = bufferViews[acc.bufferView];
  if (!bv) return null;

  const byteOffset = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
  const stride = bv.byteStride ?? components * 4;
  const out = new Float32Array(acc.count * components);
  const dv = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);

  for (let i = 0; i < acc.count; i++) {
    const base = byteOffset + i * stride;
    for (let c = 0; c < components; c++) {
      out[i * components + c] = dv.getFloat32(base + c * 4, true);
    }
  }
  return out;
}

function readIndexAccessor(
  accessors: NonNullable<GltfJson['accessors']>,
  bufferViews: NonNullable<GltfJson['bufferViews']>,
  bin: Uint8Array,
  accessorIdx: number,
): Uint32Array | null {
  const acc = accessors[accessorIdx];
  if (!acc || acc.bufferView === undefined) return null;
  const bv = bufferViews[acc.bufferView];
  if (!bv) return null;

  const byteOffset = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
  const dv = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  const out = new Uint32Array(acc.count);

  if (acc.componentType === COMPONENT_UNSIGNED_SHORT) {
    for (let i = 0; i < acc.count; i++) {
      out[i] = dv.getUint16(byteOffset + i * 2, true);
    }
  } else if (acc.componentType === COMPONENT_UNSIGNED_INT) {
    for (let i = 0; i < acc.count; i++) {
      out[i] = dv.getUint32(byteOffset + i * 4, true);
    }
  } else {
    return null;
  }
  return out;
}

function sequentialIndices(count: number): Uint32Array {
  const out = new Uint32Array(count);
  for (let i = 0; i < count; i++) out[i] = i;
  return out;
}
