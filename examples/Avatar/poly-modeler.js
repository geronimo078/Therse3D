/**
 * Blender-style Editable Polygon Mesh & GLB Exporter for therse3d.
 * Supports raw polygon creation (verts/faces), extrusion, subdivision smoothing,
 * symmetric proportional sculpting (mirrorX), and binary .glb export.
 */

export class EditablePolyMesh {
  constructor() {
    /** @type {Array<[number, number, number]>} */
    this.verts = [];
    /** @type {Array<number[]>} */
    this.faces = []; // triangles [a,b,c] or quads [a,b,c,d]
  }

  clone() {
    const copy = new EditablePolyMesh();
    copy.verts = this.verts.map((v) => [v[0], v[1], v[2]]);
    copy.faces = this.faces.map((f) => [...f]);
    return copy;
  }

  clear() {
    this.verts = [];
    this.faces = [];
  }

  /**
   * Adds raw vertices and polygon faces (indices relative to this batch).
   */
  addRaw(verts, faces) {
    if (!Array.isArray(verts) || !Array.isArray(faces)) return;
    const base = this.verts.length;
    let cx = 0, cy = 0, cz = 0, count = 0;
    for (const v of verts) {
      if (Array.isArray(v) && v.length >= 3) {
        const px = Number(v[0]) || 0;
        const py = Number(v[1]) || 0;
        const pz = Number(v[2]) || 0;
        this.verts.push([px, py, pz]);
        cx += px; cy += py; cz += pz; count++;
      }
    }
    if (count === 0) return;
    cx /= count; cy /= count; cz /= count;

    const maxIdx = this.verts.length - 1;
    for (const f of faces) {
      if (Array.isArray(f) && f.length >= 3) {
        const mapped = f.map((i) => base + (Number(i) | 0));
        if (mapped.every((i) => i >= base && i <= maxIdx)) {
          // Ensure outward CCW winding relative to batch centroid
          const a = this.verts[mapped[0]];
          const b = this.verts[mapped[1]];
          const c = this.verts[mapped[2]];
          const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
          const acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
          const nx = aby * acz - abz * acy;
          const ny = abz * acx - abx * acz;
          const nz = abx * acy - aby * acx;
          const fcx = (a[0] + b[0] + c[0]) / 3 - cx;
          const fcy = (a[1] + b[1] + c[1]) / 3 - cy;
          const fcz = (a[2] + b[2] + c[2]) / 3 - cz;
          if (nx * fcx + ny * fcy + nz * fcz < 0) {
            mapped.reverse();
          }
          this.faces.push(mapped);
        }
      }
    }
  }

  /**
   * Creates a clean quad-sphere/icosphere polygon cage at center with radius.
   */
  createPolySphere(center = [0, 1.4, 0], radius = [0.24, 0.28, 0.25], segU = 12, segV = 8) {
    const rx = Array.isArray(radius) ? (Number(radius[0]) || 0.25) : (Number(radius) || 0.25);
    const ry = Array.isArray(radius) ? (Number(radius[1]) || rx) : rx;
    const rz = Array.isArray(radius) ? (Number(radius[2]) || rx) : rx;
    const cx = Number(center[0]) || 0;
    const cy = Number(center[1]) || 0;
    const cz = Number(center[2]) || 0;

    const uCount = Math.max(6, Math.min(32, Math.round(segU)));
    const vCount = Math.max(4, Math.min(24, Math.round(segV)));

    const topIdx = this.verts.length;
    this.verts.push([cx, cy + ry, cz]);

    const rings = [];
    for (let iv = 1; iv < vCount; iv++) {
      const phi = (iv / vCount) * Math.PI;
      const sinP = Math.sin(phi);
      const cosP = Math.cos(phi);
      const ring = [];
      for (let iu = 0; iu < uCount; iu++) {
        const theta = (iu / uCount) * Math.PI * 2;
        ring.push(this.verts.length);
        this.verts.push([
          cx + sinP * Math.sin(theta) * rx,
          cy + cosP * ry,
          cz + sinP * Math.cos(theta) * rz,
        ]);
      }
      rings.push(ring);
    }

    const botIdx = this.verts.length;
    this.verts.push([cx, cy - ry, cz]);

    // Top cap triangles (CCW from outside)
    for (let iu = 0; iu < uCount; iu++) {
      const next = (iu + 1) % uCount;
      this.faces.push([topIdx, rings[0][iu], rings[0][next]]);
    }
    // Middle quads (CCW from outside)
    for (let r = 0; r < rings.length - 1; r++) {
      const r0 = rings[r];
      const r1 = rings[r + 1];
      for (let iu = 0; iu < uCount; iu++) {
        const next = (iu + 1) % uCount;
        this.faces.push([r0[iu], r1[iu], r1[next], r0[next]]);
      }
    }
    // Bottom cap triangles (CCW from outside)
    const lastRing = rings[rings.length - 1];
    for (let iu = 0; iu < uCount; iu++) {
      const next = (iu + 1) % uCount;
      this.faces.push([lastRing[iu], botIdx, lastRing[next]]);
    }
  }

  /**
   * Creates a low-poly manifold box/torso cage for extrusion & subdivision.
   */
  createPolyCage(center = [0, 0.9, 0], size = [0.46, 0.56, 0.28]) {
    const hx = (Number(size[0]) || 0.4) * 0.5;
    const hy = (Number(size[1]) || 0.5) * 0.5;
    const hz = (Number(size[2]) || 0.3) * 0.5;
    const cx = Number(center[0]) || 0;
    const cy = Number(center[1]) || 0;
    const cz = Number(center[2]) || 0;

    this.addRaw(
      [
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
      ],
      [
        [0, 1, 2, 3], // front
        [5, 4, 7, 6], // back
        [4, 0, 3, 7], // left
        [1, 5, 6, 2], // right
        [3, 2, 6, 7], // top
        [4, 5, 1, 0], // bottom
      ],
    );
  }

  _matchSelector(p, sel) {
    if (!sel || typeof sel !== 'object') return true;
    if (sel.minX !== undefined && p[0] < sel.minX) return false;
    if (sel.maxX !== undefined && p[0] > sel.maxX) return false;
    if (sel.minY !== undefined && p[1] < sel.minY) return false;
    if (sel.maxY !== undefined && p[1] > sel.maxY) return false;
    if (sel.minZ !== undefined && p[2] < sel.minZ) return false;
    if (sel.maxZ !== undefined && p[2] > sel.maxZ) return false;
    return true;
  }

  /**
   * Extrudes faces matching a spatial selector (like Blender 'E' + 'S' + 'G').
   */
  extrudeRegion(selector, offset = [0, 0.2, 0], scale = [1, 1, 1]) {
    const ox = Number(offset[0]) || 0;
    const oy = Number(offset[1]) || 0;
    const oz = Number(offset[2]) || 0;
    const sx = Array.isArray(scale) ? (Number(scale[0]) || 1) : (Number(scale) || 1);
    const sy = Array.isArray(scale) ? (Number(scale[1]) || sx) : sx;
    const sz = Array.isArray(scale) ? (Number(scale[2]) || sx) : sx;

    const matchingFaceIndices = [];
    for (let fi = 0; fi < this.faces.length; fi++) {
      const f = this.faces[fi];
      let cx = 0, cy = 0, cz = 0;
      for (const vi of f) {
        cx += this.verts[vi][0];
        cy += this.verts[vi][1];
        cz += this.verts[vi][2];
      }
      cx /= f.length;
      cy /= f.length;
      cz /= f.length;
      if (this._matchSelector([cx, cy, cz], selector)) {
        matchingFaceIndices.push(fi);
      }
    }

    if (matchingFaceIndices.length === 0) return { matchedFaces: 0 };

    // Compute region centroid for scaling
    let rcx = 0, rcy = 0, rcz = 0;
    const oldVertSet = new Set();
    for (const fi of matchingFaceIndices) {
      for (const vi of this.faces[fi]) oldVertSet.add(vi);
    }
    for (const vi of oldVertSet) {
      rcx += this.verts[vi][0];
      rcy += this.verts[vi][1];
      rcz += this.verts[vi][2];
    }
    rcx /= oldVertSet.size;
    rcy /= oldVertSet.size;
    rcz /= oldVertSet.size;

    const vertMap = new Map();
    for (const vi of oldVertSet) {
      const v = this.verts[vi];
      const nv = [
        rcx + (v[0] - rcx) * sx + ox,
        rcy + (v[1] - rcy) * sy + oy,
        rcz + (v[2] - rcz) * sz + oz,
      ];
      vertMap.set(vi, this.verts.length);
      this.verts.push(nv);
    }

    // Create side quads and move cap faces to extruded vertices
    for (const fi of matchingFaceIndices) {
      const oldFace = this.faces[fi];
      const newFace = oldFace.map((vi) => vertMap.get(vi));
      for (let k = 0; k < oldFace.length; k++) {
        const a = oldFace[k];
        const b = oldFace[(k + 1) % oldFace.length];
        const na = newFace[k];
        const nb = newFace[(k + 1) % newFace.length];
        this.faces.push([a, b, nb, na]);
      }
      this.faces[fi] = newFace;
    }
    return { matchedFaces: matchingFaceIndices.length };
  }

  /**
   * Proportional sculpt deformation (like Blender Sculpt Grab / Proportional Edit).
   */
  sculptDeform(center = [0, 1.3, 0.2], radius = 0.25, delta = [0, 0, 0.08], mirrorX = true) {
    const r = Math.max(0.02, Number(radius) || 0.2);
    const dx = Number(delta[0]) || 0;
    const dy = Number(delta[1]) || 0;
    const dz = Number(delta[2]) || 0;
    let affected = 0;
    let nearestDist = Infinity;
    let nearestVert = null;

    const applyAt = (cx, cy, cz, signX) => {
      for (let i = 0; i < this.verts.length; i++) {
        const v = this.verts[i];
        const dist = Math.hypot(v[0] - cx, v[1] - cy, v[2] - cz);
        if (signX === 1 && dist < nearestDist) {
          nearestDist = dist;
          nearestVert = [
            Math.round(v[0] * 100) / 100,
            Math.round(v[1] * 100) / 100,
            Math.round(v[2] * 100) / 100,
          ];
        }
        if (dist < r) {
          const t = 1 - dist / r;
          const weight = t * t * (3 - 2 * t);
          v[0] += dx * signX * weight;
          v[1] += dy * weight;
          v[2] += dz * weight;
          affected++;
        }
      }
    };

    const cx = Number(center[0]) || 0;
    const cy = Number(center[1]) || 0;
    const cz = Number(center[2]) || 0;
    applyAt(cx, cy, cz, 1);
    if (mirrorX && Math.abs(cx) > 0.01) {
      applyAt(-cx, cy, cz, -1);
    }
    return { affected, nearestVert };
  }

  /**
   * Moves or scales vertices matching a spatial selector.
   */
  transformRegion(selector, delta = [0, 0, 0], scale = [1, 1, 1]) {
    const matched = [];
    for (let i = 0; i < this.verts.length; i++) {
      if (this._matchSelector(this.verts[i], selector)) matched.push(i);
    }
    if (matched.length === 0) return;

    let cx = 0, cy = 0, cz = 0;
    for (const i of matched) {
      cx += this.verts[i][0];
      cy += this.verts[i][1];
      cz += this.verts[i][2];
    }
    cx /= matched.length;
    cy /= matched.length;
    cz /= matched.length;

    const dx = Number(delta[0]) || 0;
    const dy = Number(delta[1]) || 0;
    const dz = Number(delta[2]) || 0;
    const sx = Array.isArray(scale) ? (Number(scale[0]) || 1) : (Number(scale) || 1);
    const sy = Array.isArray(scale) ? (Number(scale[1]) || sx) : sx;
    const sz = Array.isArray(scale) ? (Number(scale[2]) || sx) : sx;

    for (const i of matched) {
      const v = this.verts[i];
      v[0] = cx + (v[0] - cx) * sx + dx;
      v[1] = cy + (v[1] - cy) * sy + dy;
      v[2] = cz + (v[2] - cz) * sz + dz;
    }
  }

  /**
   * Subdivides all faces (quads -> 4 quads, tris -> 4 tris) with optional Laplacian smoothing.
   */
  subdivide(iterations = 1, smooth = true) {
    const iters = Math.max(1, Math.min(2, Math.round(iterations)));
    for (let it = 0; it < iters; it++) {
      if (this.faces.length > 6000) break; // safety cap for interactive WebGPU
      const edgeMidMap = new Map();
      const getMid = (a, b) => {
        const key = a < b ? `${a}_${b}` : `${b}_${a}`;
        const existing = edgeMidMap.get(key);
        if (existing !== undefined) return existing;
        const va = this.verts[a];
        const vb = this.verts[b];
        const idx = this.verts.length;
        this.verts.push([
          (va[0] + vb[0]) * 0.5,
          (va[1] + vb[1]) * 0.5,
          (va[2] + vb[2]) * 0.5,
        ]);
        edgeMidMap.set(key, idx);
        return idx;
      };

      const nextFaces = [];
      for (const f of this.faces) {
        if (f.length === 4) {
          const [a, b, c, d] = f;
          const ab = getMid(a, b);
          const bc = getMid(b, c);
          const cd = getMid(c, d);
          const da = getMid(d, a);
          const centerIdx = this.verts.length;
          this.verts.push([
            (this.verts[a][0] + this.verts[b][0] + this.verts[c][0] + this.verts[d][0]) * 0.25,
            (this.verts[a][1] + this.verts[b][1] + this.verts[c][1] + this.verts[d][1]) * 0.25,
            (this.verts[a][2] + this.verts[b][2] + this.verts[c][2] + this.verts[d][2]) * 0.25,
          ]);
          nextFaces.push([a, ab, centerIdx, da]);
          nextFaces.push([ab, b, bc, centerIdx]);
          nextFaces.push([centerIdx, bc, c, cd]);
          nextFaces.push([da, centerIdx, cd, d]);
        } else if (f.length === 3) {
          const [a, b, c] = f;
          const ab = getMid(a, b);
          const bc = getMid(b, c);
          const ca = getMid(c, a);
          nextFaces.push([a, ab, ca]);
          nextFaces.push([ab, b, bc]);
          nextFaces.push([ca, bc, c]);
          nextFaces.push([ab, bc, ca]);
        }
      }
      this.faces = nextFaces;

      if (smooth) {
        const neighbors = Array.from({ length: this.verts.length }, () => new Set());
        for (const f of this.faces) {
          for (let k = 0; k < f.length; k++) {
            const a = f[k];
            const b = f[(k + 1) % f.length];
            neighbors[a].add(b);
            neighbors[b].add(a);
          }
        }
        const smoothed = this.verts.map((v, i) => {
          const nb = neighbors[i];
          if (nb.size === 0) return [v[0], v[1], v[2]];
          let ax = 0, ay = 0, az = 0;
          for (const j of nb) {
            ax += this.verts[j][0];
            ay += this.verts[j][1];
            az += this.verts[j][2];
          }
          ax /= nb.size;
          ay /= nb.size;
          az /= nb.size;
          return [
            v[0] * 0.55 + ax * 0.45,
            v[1] * 0.55 + ay * 0.45,
            v[2] * 0.55 + az * 0.45,
          ];
        });
        this.verts = smoothed;
      }
    }
  }

  /**
   * Adds a smooth tapered polygon limb/horn/tail along 3D points (optionally mirrored across X).
   */
  addPolyTube(points = [[0.24, 1.0, 0], [0.46, 0.8, 0.05], [0.52, 0.5, 0.12]], radii = [0.08, 0.06, 0.04], mirrorX = true, radialSegs = 10) {
    if (!Array.isArray(points) || points.length < 2) return 0;
    const segU = Math.max(6, Math.min(20, Math.round(radialSegs)));
    const segV = points.length - 1;

    const addSingle = (pts) => {
      const rings = [];
      let prevTx = 0, prevTy = 1, prevTz = 0;
      let nx = 1, ny = 0, nz = 0;

      for (let iv = 0; iv <= segV; iv++) {
        const p = pts[iv];
        const prev = pts[Math.max(0, iv - 1)];
        const next = pts[Math.min(segV, iv + 1)];
        let tx = (next[0] ?? 0) - (prev[0] ?? 0);
        let ty = (next[1] ?? 0) - (prev[1] ?? 0);
        let tz = (next[2] ?? 0) - (prev[2] ?? 0);
        const tLen = Math.hypot(tx, ty, tz) || 1;
        tx /= tLen; ty /= tLen; tz /= tLen;

        if (iv === 0) {
          const rx = Math.abs(ty) < 0.9 ? 0 : 1;
          const ry = Math.abs(ty) < 0.9 ? 1 : 0;
          nx = ty * 0 - tz * ry;
          ny = tz * rx - tx * 0;
          nz = tx * ry - ty * rx;
          const nLen = Math.hypot(nx, ny, nz) || 1;
          nx /= nLen; ny /= nLen; nz /= nLen;
        } else {
          // Parallel transport previous normal onto plane perpendicular to new tangent
          const dot = nx * tx + ny * ty + nz * tz;
          nx -= dot * tx;
          ny -= dot * ty;
          nz -= dot * tz;
          const nLen = Math.hypot(nx, ny, nz);
          if (nLen < 1e-4) {
            const rx = Math.abs(ty) < 0.9 ? 0 : 1;
            const ry = Math.abs(ty) < 0.9 ? 1 : 0;
            nx = ty * 0 - tz * ry;
            ny = tz * rx - tx * 0;
            nz = tx * ry - ty * rx;
            const len2 = Math.hypot(nx, ny, nz) || 1;
            nx /= len2; ny /= len2; nz /= len2;
          } else {
            nx /= nLen; ny /= nLen; nz /= nLen;
          }
        }
        prevTx = tx; prevTy = ty; prevTz = tz;
        void prevTx; void prevTy; void prevTz;

        // binormal = tangent x normal -> (normal, binormal, tangent) is right-handed
        const bx = ty * nz - tz * ny;
        const by = tz * nx - tx * nz;
        const bz = tx * ny - ty * nx;

        const r = Array.isArray(radii)
          ? Math.max(0.005, Number(radii[Math.min(iv, radii.length - 1)]) || 0.05)
          : Math.max(0.005, Number(radii) || 0.05);

        const ring = [];
        for (let iu = 0; iu < segU; iu++) {
          const ang = (iu / segU) * Math.PI * 2;
          const ca = Math.cos(ang) * r;
          const sa = Math.sin(ang) * r;
          ring.push(this.verts.length);
          this.verts.push([
            (p[0] ?? 0) + nx * ca + bx * sa,
            (p[1] ?? 0) + ny * ca + by * sa,
            (p[2] ?? 0) + nz * ca + bz * sa,
          ]);
        }
        rings.push(ring);
      }

      for (let iv = 0; iv < segV; iv++) {
        const r0 = rings[iv];
        const r1 = rings[iv + 1];
        for (let iu = 0; iu < segU; iu++) {
          const next = (iu + 1) % segU;
          this.faces.push([r0[iu], r0[next], r1[next], r1[iu]]);
        }
      }
      // Cap ends (start cap faces -tangent, end cap faces +tangent)
      this.faces.push([...rings[0]].reverse());
      this.faces.push([...rings[segV]]);
    };

    addSingle(points);
    if (mirrorX) {
      addSingle(points.map((p) => [-(Number(p[0]) || 0), Number(p[1]) || 0, Number(p[2]) || 0]));
    }
    return mirrorX ? 2 : 1;
  }

  /**
   * Executes an array of Blender-style modeling ops and returns per-op feedback for the AI.
   */
  applyOps(ops) {
    const opFeedback = [];
    if (!Array.isArray(ops)) return opFeedback;
    for (const op of ops) {
      if (!op || typeof op !== 'object') continue;
      // If the AI sends deform/extrude/subdivide ops on an empty mesh, auto-seed a manifold humanoid base cage
      if (
        this.verts.length === 0 &&
        (op.op === 'sculpt_deform' || op.op === 'extrude_region' || op.op === 'transform_region' || op.op === 'subdivide')
      ) {
        this.createPolyCage([0, 0.75, 0], [0.44, 0.56, 0.26]);
        this.createPolySphere([0, 1.32, 0], [0.22, 0.26, 0.24], 12, 8);
        opFeedback.push('auto_seeded base cage because mesh was empty');
      }
      switch (op.op) {
        case 'clear':
          this.clear();
          opFeedback.push('cleared mesh');
          break;
        case 'add_raw':
        case 'add_verts_faces':
          this.addRaw(op.verts, op.faces);
          opFeedback.push(`add_verts_faces -> total ${this.verts.length} verts`);
          break;
        case 'create_poly_sphere':
          this.createPolySphere(op.center, op.radius, op.segU, op.segV);
          opFeedback.push(`create_poly_sphere at ${JSON.stringify(op.center ?? [0, 1.32, 0])}`);
          break;
        case 'create_poly_cage':
          this.createPolyCage(op.center, op.size);
          opFeedback.push(`create_poly_cage at ${JSON.stringify(op.center ?? [0, 0.75, 0])}`);
          break;
        case 'add_poly_tube': {
          const count = this.addPolyTube(op.points, op.radii, op.mirrorX ?? true, op.segments ?? 10);
          opFeedback.push(`add_poly_tube added ${count} tube(s)`);
          break;
        }
        case 'extrude_region': {
          const res = this.extrudeRegion(op.selector, op.offset, op.scale);
          opFeedback.push(`extrude_region matched ${res.matchedFaces} faces`);
          break;
        }
        case 'sculpt_deform': {
          const res = this.sculptDeform(op.center, op.radius, op.delta, op.mirrorX ?? true);
          opFeedback.push(
            res.affected > 0
              ? `sculpt_deform moved ${res.affected} verts`
              : `sculpt_deform at ${JSON.stringify(op.center)} affected 0 verts (nearest vert is at ${JSON.stringify(res.nearestVert)})`,
          );
          break;
        }
        case 'transform_region':
          this.transformRegion(op.selector, op.delta, op.scale);
          opFeedback.push('transform_region applied');
          break;
        case 'subdivide':
          this.subdivide(op.iterations ?? 1, op.smooth ?? true);
          opFeedback.push(`subdivide -> ${this.verts.length} verts, ${this.faces.length} faces`);
          break;
      }
    }
    return opFeedback;
  }

  getBounds() {
    if (this.verts.length === 0) {
      return { min: [0, 0, 0], max: [0, 0, 0] };
    }
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const v of this.verts) {
      if (v[0] < min[0]) min[0] = v[0];
      if (v[1] < min[1]) min[1] = v[1];
      if (v[2] < min[2]) min[2] = v[2];
      if (v[0] > max[0]) max[0] = v[0];
      if (v[1] > max[1]) max[1] = v[1];
      if (v[2] > max[2]) max[2] = v[2];
    }
    return {
      min: min.map((n) => Math.round(n * 100) / 100),
      max: max.map((n) => Math.round(n * 100) / 100),
    };
  }

  /**
   * Ensures every connected mesh component has positive signed volume (CCW outward normals).
   */
  ensureOutwardWinding() {
    if (this.faces.length === 0 || this.verts.length === 0) return;
    const vertToFaces = Array.from({ length: this.verts.length }, () => []);
    for (let fi = 0; fi < this.faces.length; fi++) {
      for (const vi of this.faces[fi]) {
        vertToFaces[vi]?.push(fi);
      }
    }

    const visited = new Uint8Array(this.faces.length);
    for (let start = 0; start < this.faces.length; start++) {
      if (visited[start]) continue;
      const compFaces = [];
      const queue = [start];
      visited[start] = 1;
      while (queue.length > 0) {
        const fi = queue.pop();
        compFaces.push(fi);
        for (const vi of this.faces[fi]) {
          for (const adj of vertToFaces[vi]) {
            if (!visited[adj]) {
              visited[adj] = 1;
              queue.push(adj);
            }
          }
        }
      }

      // Compute signed volume of this connected component
      let vol6 = 0;
      for (const fi of compFaces) {
        const f = this.faces[fi];
        for (let k = 1; k < f.length - 1; k++) {
          const a = this.verts[f[0]];
          const b = this.verts[f[k]];
          const c = this.verts[f[k + 1]];
          vol6 +=
            a[0] * (b[1] * c[2] - b[2] * c[1]) +
            a[1] * (b[2] * c[0] - b[0] * c[2]) +
            a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
      }

      if (vol6 < 0) {
        for (const fi of compFaces) {
          this.faces[fi].reverse();
        }
      }
    }
  }

  /**
   * Compiles the polygon mesh to therse3d interleaved GeometryMeshData.
   */
  toGeometryMeshData() {
    this.ensureOutwardWinding();
    const triIndices = [];
    for (const f of this.faces) {
      if (f.length === 3) {
        triIndices.push(f[0], f[1], f[2]);
      } else if (f.length >= 4) {
        for (let k = 1; k < f.length - 1; k++) {
          triIndices.push(f[0], f[k], f[k + 1]);
        }
      }
    }

    const vertCount = this.verts.length;
    const normals = new Float32Array(vertCount * 3);
    for (let i = 0; i < triIndices.length; i += 3) {
      const i0 = triIndices[i];
      const i1 = triIndices[i + 1];
      const i2 = triIndices[i + 2];
      const a = this.verts[i0];
      const b = this.verts[i1];
      const c = this.verts[i2];
      const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
      const acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
      const nx = aby * acz - abz * acy;
      const ny = abz * acx - abx * acz;
      const nz = abx * acy - aby * acx;
      normals[i0 * 3] += nx; normals[i0 * 3 + 1] += ny; normals[i0 * 3 + 2] += nz;
      normals[i1 * 3] += nx; normals[i1 * 3 + 1] += ny; normals[i1 * 3 + 2] += nz;
      normals[i2 * 3] += nx; normals[i2 * 3 + 1] += ny; normals[i2 * 3 + 2] += nz;
    }

    const vertices = new Float32Array(vertCount * 8);
    for (let i = 0; i < vertCount; i++) {
      const v = this.verts[i];
      const nx = normals[i * 3];
      const ny = normals[i * 3 + 1];
      const nz = normals[i * 3 + 2];
      const len = Math.hypot(nx, ny, nz) || 1;
      const o = i * 8;
      vertices[o] = v[0];
      vertices[o + 1] = v[1];
      vertices[o + 2] = v[2];
      vertices[o + 3] = nx / len;
      vertices[o + 4] = ny / len;
      vertices[o + 5] = nz / len;
      vertices[o + 6] = 0.5 + Math.atan2(v[0], v[2]) / (Math.PI * 2);
      vertices[o + 7] = Math.max(0, Math.min(1, v[1] * 0.5));
    }

    return {
      vertices,
      indices: new Uint32Array(triIndices),
    };
  }

  /**
   * Exports the modeled polygon mesh as a valid binary glTF 2.0 (.glb) Blob.
   */
  toGlbBlob() {
    const meshData = this.toGeometryMeshData();
    const vertCount = this.verts.length;
    const posArr = new Float32Array(vertCount * 3);
    const normArr = new Float32Array(vertCount * 3);
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    for (let i = 0; i < vertCount; i++) {
      const o = i * 8;
      const x = meshData.vertices[o];
      const y = meshData.vertices[o + 1];
      const z = meshData.vertices[o + 2];
      posArr[i * 3] = x; posArr[i * 3 + 1] = y; posArr[i * 3 + 2] = z;
      normArr[i * 3] = meshData.vertices[o + 3];
      normArr[i * 3 + 1] = meshData.vertices[o + 4];
      normArr[i * 3 + 2] = meshData.vertices[o + 5];
      if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
      if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
    }
    const idxArr = meshData.indices;

    const rawBinBytes = posArr.byteLength + normArr.byteLength + idxArr.byteLength;
    const binPaddedLen = (rawBinBytes + 3) & ~3;
    const binBuf = new Uint8Array(binPaddedLen);
    binBuf.set(new Uint8Array(posArr.buffer), 0);
    binBuf.set(new Uint8Array(normArr.buffer), posArr.byteLength);
    binBuf.set(new Uint8Array(idxArr.buffer), posArr.byteLength + normArr.byteLength);

    const gltf = {
      asset: { version: '2.0', generator: 'therse3d-ai-poly-modeler' },
      scene: 0,
      scenes: [{ nodes: [0] }],
      nodes: [{ mesh: 0, name: 'AI_Avatar_Mesh' }],
      meshes: [{
        name: 'AI_Avatar_Mesh',
        primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, indices: 2 }],
      }],
      buffers: [{ byteLength: binPaddedLen }],
      bufferViews: [
        { buffer: 0, byteOffset: 0, byteLength: posArr.byteLength, target: 34962 },
        { buffer: 0, byteOffset: posArr.byteLength, byteLength: normArr.byteLength, target: 34962 },
        { buffer: 0, byteOffset: posArr.byteLength + normArr.byteLength, byteLength: idxArr.byteLength, target: 34963 },
      ],
      accessors: [
        { bufferView: 0, componentType: 5126, count: vertCount, type: 'VEC3', min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
        { buffer: 1, componentType: 5126, count: vertCount, type: 'VEC3' },
        { bufferView: 2, componentType: 5125, count: idxArr.length, type: 'SCALAR' },
      ],
    };

    const jsonBytes = new TextEncoder().encode(JSON.stringify(gltf));
    const jsonPaddedLen = (jsonBytes.length + 3) & ~3;
    const jsonBuf = new Uint8Array(jsonPaddedLen);
    jsonBuf.fill(0x20);
    jsonBuf.set(jsonBytes);

    const totalLen = 12 + 8 + jsonPaddedLen + 8 + binPaddedLen;
    const out = new ArrayBuffer(totalLen);
    const dv = new DataView(out);
    const bytes = new Uint8Array(out);
    dv.setUint32(0, 0x46546c67, true); // 'glTF'
    dv.setUint32(4, 2, true);
    dv.setUint32(8, totalLen, true);
    dv.setUint32(12, jsonPaddedLen, true);
    dv.setUint32(16, 0x4e4f534a, true); // 'JSON'
    bytes.set(jsonBuf, 20);
    const binOff = 20 + jsonPaddedLen;
    dv.setUint32(binOff, binPaddedLen, true);
    dv.setUint32(binOff + 4, 0x004e4942, true); // 'BIN\0'
    bytes.set(binBuf, binOff + 8);

    return new Blob([out], { type: 'model/gltf-binary' });
  }
}
