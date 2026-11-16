/**
 * Procedural high-detail sculpted humanoid avatar mesh & texture generator
 * for therse3d. Produces smooth vertex-normal GeometryMeshData (no blocky boxes).
 */

function computeSmoothNormals(positions, indices) {
  const normals = new Float32Array(positions.length);
  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i] * 3;
    const i1 = indices[i + 1] * 3;
    const i2 = indices[i + 2] * 3;

    const ax = positions[i0], ay = positions[i0 + 1], az = positions[i0 + 2];
    const bx = positions[i1], by = positions[i1 + 1], bz = positions[i1 + 2];
    const cx = positions[i2], cy = positions[i2 + 1], cz = positions[i2 + 2];

    const abx = bx - ax, aby = by - ay, abz = bz - az;
    const acx = cx - ax, acy = cy - ay, acz = cz - az;

    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;

    normals[i0] += nx; normals[i0 + 1] += ny; normals[i0 + 2] += nz;
    normals[i1] += nx; normals[i1 + 1] += ny; normals[i1 + 2] += nz;
    normals[i2] += nx; normals[i2 + 1] += ny; normals[i2 + 2] += nz;
  }

  for (let i = 0; i < normals.length; i += 3) {
    const x = normals[i], y = normals[i + 1], z = normals[i + 2];
    const len = Math.hypot(x, y, z) || 1;
    normals[i] = x / len;
    normals[i + 1] = y / len;
    normals[i + 2] = z / len;
  }
  return normals;
}

function packMeshData(positions, uvs, indices) {
  const normals = computeSmoothNormals(positions, indices);
  const vertCount = positions.length / 3;
  const vertices = new Float32Array(vertCount * 8);
  for (let i = 0; i < vertCount; i++) {
    const p = i * 3;
    const t = i * 2;
    const o = i * 8;
    vertices[o] = positions[p];
    vertices[o + 1] = positions[p + 1];
    vertices[o + 2] = positions[p + 2];
    vertices[o + 3] = normals[p];
    vertices[o + 4] = normals[p + 1];
    vertices[o + 5] = normals[p + 2];
    vertices[o + 6] = uvs[t];
    vertices[o + 7] = uvs[t + 1];
  }
  return { vertices, indices };
}

/**
 * Builds a smooth 3D surface of revolution from a 2D profile [[radius, y], ...].
 * Ideal for organic torsos, helmets, robes, vases, necks, and crowns.
 */
export function buildLatheMesh(profile, segments = 28) {
  const pts = Array.isArray(profile) && profile.length >= 2
    ? profile
    : [[0.2, -0.3], [0.28, 0], [0.2, 0.3]];
  const segU = Math.max(8, Math.min(64, Math.round(segments)));
  const segV = pts.length - 1;
  const vertCount = (segU + 1) * (segV + 1);
  const positions = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const indices = new Uint32Array(segU * segV * 6);

  let vIdx = 0;
  let uvIdx = 0;
  for (let iv = 0; iv <= segV; iv++) {
    const r = Math.max(0.002, Number(pts[iv][0]) || 0.002);
    const y = Number(pts[iv][1]) || 0;
    const v = iv / segV;
    for (let iu = 0; iu <= segU; iu++) {
      const u = iu / segU;
      const theta = u * Math.PI * 2;
      positions[vIdx++] = Math.cos(theta) * r;
      positions[vIdx++] = y;
      positions[vIdx++] = Math.sin(theta) * r;
      uvs[uvIdx++] = u;
      uvs[uvIdx++] = v;
    }
  }

  let idx = 0;
  const stride = segU + 1;
  for (let iv = 0; iv < segV; iv++) {
    for (let iu = 0; iu < segU; iu++) {
      const a = iv * stride + iu;
      const b = a + stride;
      const c = a + 1;
      const d = b + 1;
      indices[idx++] = a; indices[idx++] = c; indices[idx++] = b;
      indices[idx++] = c; indices[idx++] = d; indices[idx++] = b;
    }
  }
  return packMeshData(positions, uvs, indices);
}

/**
 * Builds a smooth 3D tube along control points [[x,y,z], ...] with variable radii.
 * Ideal for articulated limbs, tails, tentacles, horns, cables, and hair strands.
 */
export function buildTubeMesh(points, radii = 0.06, radialSegs = 14) {
  const pts = Array.isArray(points) && points.length >= 2
    ? points
    : [[0, 0, 0], [0, 0.5, 0]];
  const segU = Math.max(6, Math.min(32, Math.round(radialSegs)));
  const segV = pts.length - 1;
  const vertCount = (segU + 1) * (segV + 1);
  const positions = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const indices = new Uint32Array(segU * segV * 6);

  const getRadius = (i) => {
    if (typeof radii === 'number') return Math.max(0.003, radii);
    if (Array.isArray(radii) && radii.length > 0) {
      if (radii.length === pts.length) return Math.max(0.003, Number(radii[i]) || 0.03);
      const t = segV > 0 ? i / segV : 0;
      const r0 = Number(radii[0]) || 0.05;
      const r1 = Number(radii[radii.length - 1]) || 0.05;
      return Math.max(0.003, r0 + (r1 - r0) * t);
    }
    return 0.05;
  };

  let vIdx = 0;
  let uvIdx = 0;
  for (let iv = 0; iv <= segV; iv++) {
    const p = pts[iv];
    const prev = pts[Math.max(0, iv - 1)];
    const next = pts[Math.min(segV, iv + 1)];
    let tx = (next[0] ?? 0) - (prev[0] ?? 0);
    let ty = (next[1] ?? 0) - (prev[1] ?? 0);
    let tz = (next[2] ?? 0) - (prev[2] ?? 0);
    const tLen = Math.hypot(tx, ty, tz) || 1;
    tx /= tLen; ty /= tLen; tz /= tLen;

    // Reference vector orthogonal to tangent
    let rx = Math.abs(ty) < 0.9 ? 0 : 1;
    let ry = Math.abs(ty) < 0.9 ? 1 : 0;
    let rz = 0;
    // normal = normalize(cross(tangent, ref))
    let nx = ty * rz - tz * ry;
    let ny = tz * rx - tx * rz;
    let nz = tx * ry - ty * rx;
    const nLen = Math.hypot(nx, ny, nz) || 1;
    nx /= nLen; ny /= nLen; nz /= nLen;
    // binormal = cross(tangent, normal)
    const bx = ty * nz - tz * ny;
    const by = tz * nx - tx * nz;
    const bz = tx * ny - ty * nx;

    const r = getRadius(iv);
    const v = segV > 0 ? iv / segV : 0;
    for (let iu = 0; iu <= segU; iu++) {
      const u = iu / segU;
      const ang = u * Math.PI * 2;
      const ca = Math.cos(ang) * r;
      const sa = Math.sin(ang) * r;
      positions[vIdx++] = (p[0] ?? 0) + nx * ca + bx * sa;
      positions[vIdx++] = (p[1] ?? 0) + ny * ca + by * sa;
      positions[vIdx++] = (p[2] ?? 0) + nz * ca + bz * sa;
      uvs[uvIdx++] = u;
      uvs[uvIdx++] = v;
    }
  }

  let idx = 0;
  const stride = segU + 1;
  for (let iv = 0; iv < segV; iv++) {
    for (let iu = 0; iu < segU; iu++) {
      const a = iv * stride + iu;
      const b = a + stride;
      const c = a + 1;
      const d = b + 1;
      indices[idx++] = a; indices[idx++] = b; indices[idx++] = c;
      indices[idx++] = c; indices[idx++] = b; indices[idx++] = d;
    }
  }
  return packMeshData(positions, uvs, indices);
}

function gauss(x, mu, sigma) {
  const d = (x - mu) / sigma;
  return Math.exp(-0.5 * d * d);
}

/**
 * Sculpted humanoid head with cranium, brow ridge, eye sockets, nose bridge/tip,
 * cheekbones, lip contour, jawline, and chin.
 */
export function createSculptedHeadMesh(segU = 48, segV = 36) {
  const vertCount = (segU + 1) * (segV + 1);
  const positions = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const indices = new Uint32Array(segU * segV * 6);

  let vIdx = 0;
  let uvIdx = 0;

  for (let iv = 0; iv <= segV; iv++) {
    const v = iv / segV;
    const phi = v * Math.PI; // 0 (top) .. PI (bottom)
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);

    for (let iu = 0; iu <= segU; iu++) {
      const u = iu / segU;
      // Shift theta so +Z is center of face (theta = 0 at +Z)
      const theta = (u - 0.5) * Math.PI * 2;
      const nx = sinPhi * Math.sin(theta);
      const ny = cosPhi;
      const nz = sinPhi * Math.cos(theta);

      // Base cranial ellipsoid
      let x = nx * 0.34;
      let y = ny * 0.43;
      let z = nz * 0.38;

      // Taper lower face into jaw & chin, widen cranium slightly at back/top
      if (ny < 0.05) {
        const jawT = Math.min(1, (0.05 - ny) / 0.95);
        const taper = 1.0 - 0.42 * jawT * jawT;
        x *= taper;
        if (nz > -0.1) {
          z *= 1.0 - 0.18 * jawT * jawT;
        }
      } else {
        // Cranial fullness
        const crown = gauss(ny, 0.45, 0.45);
        x *= 1.0 + 0.06 * crown;
        z *= 1.0 + 0.08 * crown;
      }

      // Chin forward projection
      const chin = gauss(ny, -0.78, 0.18) * gauss(nx, 0.0, 0.24) * Math.max(0, nz);
      z += chin * 0.065;
      y -= chin * 0.02;

      // Front face sculptural features (nz > 0)
      if (nz > 0.1) {
        const frontMask = Math.pow(nz, 1.4);

        // Brow ridge
        const brow = gauss(ny, 0.26, 0.09) * gauss(Math.abs(nx), 0.18, 0.22) * frontMask;
        z += brow * 0.038;

        // Eye sockets (subtle inward recess)
        const eyeL = gauss(nx, -0.28, 0.14) * gauss(ny, 0.09, 0.11);
        const eyeR = gauss(nx, 0.28, 0.14) * gauss(ny, 0.09, 0.11);
        z -= (eyeL + eyeR) * 0.045 * frontMask;

        // Nose bridge & tip
        const noseBridge = gauss(nx, 0.0, 0.085) * gauss(ny, 0.04, 0.17) * frontMask;
        const noseTip = gauss(nx, 0.0, 0.09) * gauss(ny, -0.16, 0.085) * frontMask;
        z += noseBridge * 0.055 + noseTip * 0.082;

        // Cheekbones
        const cheek = gauss(Math.abs(nx), 0.42, 0.16) * gauss(ny, -0.14, 0.14) * frontMask;
        z += cheek * 0.032;
        x += Math.sign(nx) * cheek * 0.018;

        // Upper & lower lips
        const upperLip = gauss(nx, 0.0, 0.16) * gauss(ny, -0.39, 0.045) * frontMask;
        const lowerLip = gauss(nx, 0.0, 0.14) * gauss(ny, -0.49, 0.05) * frontMask;
        const mouthCrease = gauss(nx, 0.0, 0.16) * gauss(ny, -0.44, 0.022) * frontMask;
        z += upperLip * 0.032 + lowerLip * 0.028 - mouthCrease * 0.024;
      }

      // Flatten temples slightly
      const temple = gauss(Math.abs(nx), 0.92, 0.14) * gauss(ny, 0.12, 0.28);
      x *= 1.0 - 0.06 * temple;

      positions[vIdx++] = x;
      positions[vIdx++] = y;
      positions[vIdx++] = z;

      uvs[uvIdx++] = u;
      uvs[uvIdx++] = 1.0 - v;
    }
  }

  let idx = 0;
  const stride = segU + 1;
  for (let iv = 0; iv < segV; iv++) {
    for (let iu = 0; iu < segU; iu++) {
      const a = iv * stride + iu;
      const b = a + stride;
      const c = a + 1;
      const d = b + 1;
      indices[idx++] = a; indices[idx++] = b; indices[idx++] = c;
      indices[idx++] = c; indices[idx++] = b; indices[idx++] = d;
    }
  }

  return packMeshData(positions, uvs, indices);
}

/**
 * Smooth anatomical upper torso + shoulders + neck collar.
 */
export function createSculptedTorsoMesh(segU = 40, segV = 28) {
  const vertCount = (segU + 1) * (segV + 1);
  const positions = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const indices = new Uint32Array(segU * segV * 6);

  let vIdx = 0;
  let uvIdx = 0;

  for (let iv = 0; iv <= segV; iv++) {
    const v = iv / segV; // 0 (bottom waist) .. 1 (top neck)
    const y = -0.45 + v * 0.72; // -0.45 .. +0.27

    // Width & depth profile along height
    let rx = 0.24;
    let rz = 0.16;

    if (v < 0.55) {
      // Waist to chest/ribcage
      const t = v / 0.55;
      rx = 0.23 + 0.11 * Math.sin(t * Math.PI * 0.5);
      rz = 0.15 + 0.06 * Math.sin(t * Math.PI * 0.5);
    } else if (v < 0.78) {
      // Broad shoulders & upper chest tapering to trapezius
      const t = (v - 0.55) / 0.23;
      const shoulderPeak = Math.sin(t * Math.PI);
      rx = 0.34 + 0.14 * shoulderPeak - 0.21 * t * t;
      rz = 0.21 - 0.08 * t;
    } else {
      // Neck column
      const t = (v - 0.78) / 0.22;
      rx = 0.13 - 0.015 * Math.sin(t * Math.PI);
      rz = 0.13 - 0.015 * Math.sin(t * Math.PI);
    }

    for (let iu = 0; iu <= segU; iu++) {
      const u = iu / segU;
      const theta = u * Math.PI * 2;
      const cosT = Math.cos(theta);
      const sinT = Math.sin(theta);

      let px = cosT * rx;
      let pz = sinT * rz;
      let py = y;

      // Subtle pectoral & back contour
      if (v > 0.25 && v < 0.7 && sinT > 0) {
        const chest = gauss(v, 0.5, 0.14) * sinT;
        pz += chest * 0.04;
      }
      // Slight shoulder slope downward at outer X
      if (v > 0.52 && v < 0.8) {
        py -= Math.abs(cosT) * 0.045 * gauss(v, 0.66, 0.12);
      }

      positions[vIdx++] = px;
      positions[vIdx++] = py;
      positions[vIdx++] = pz;
      uvs[uvIdx++] = u;
      uvs[uvIdx++] = v;
    }
  }

  let idx = 0;
  const stride = segU + 1;
  for (let iv = 0; iv < segV; iv++) {
    for (let iu = 0; iu < segU; iu++) {
      const a = iv * stride + iu;
      const b = a + stride;
      const c = a + 1;
      const d = b + 1;
      indices[idx++] = a; indices[idx++] = c; indices[idx++] = b;
      indices[idx++] = c; indices[idx++] = d; indices[idx++] = b;
    }
  }

  return packMeshData(positions, uvs, indices);
}

/**
 * Conforming sculpted hair / cyber-helmet shell over the head.
 */
export function createSculptedHairMesh(style = 'swept', segU = 40, segV = 24) {
  const vertCount = (segU + 1) * (segV + 1);
  const positions = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const indices = new Uint32Array(segU * segV * 6);

  let vIdx = 0;
  let uvIdx = 0;

  const isHelmet = style === 'helmet';
  const isBob = style === 'bob';
  const maxPhi = isBob ? Math.PI * 0.78 : isHelmet ? Math.PI * 0.62 : Math.PI * 0.58;

  for (let iv = 0; iv <= segV; iv++) {
    const v = iv / segV;
    const phi = v * maxPhi;
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);

    for (let iu = 0; iu <= segU; iu++) {
      const u = iu / segU;
      const theta = (u - 0.5) * Math.PI * 2;
      const nx = sinPhi * Math.sin(theta);
      const ny = cosPhi;
      const nz = sinPhi * Math.cos(theta);

      // Cut out front face opening so eyes/nose/mouth remain visible
      let thickness = isHelmet ? 0.055 : 0.045;
      const frontAngle = Math.atan2(nx, Math.max(0.001, nz));
      if (nz > 0.15 && ny < 0.42) {
        const faceCut = Math.exp(-frontAngle * frontAngle * 4.5);
        thickness *= Math.max(0.0, 1.0 - faceCut * 1.15);
      }

      // Add organic swept hair volume / spikes / helmet crest
      if (style === 'swept') {
        const wave = Math.sin(theta * 5.0 + v * 4.0) * 0.018 * Math.sin(v * Math.PI);
        const crest = gauss(nx, 0.0, 0.3) * gauss(ny, 0.65, 0.3) * 0.045;
        thickness += wave + crest;
      } else if (isHelmet) {
        const ridge = gauss(nx, 0.0, 0.12) * 0.045;
        thickness += ridge;
      } else if (isBob) {
        thickness += 0.025 * Math.sin(v * Math.PI);
      }

      const rX = 0.355 + thickness;
      const rY = 0.445 + thickness;
      const rZ = 0.395 + thickness;

      positions[vIdx++] = nx * rX;
      positions[vIdx++] = ny * rY;
      positions[vIdx++] = nz * rZ;
      uvs[uvIdx++] = u;
      uvs[uvIdx++] = 1.0 - v;
    }
  }

  let idx = 0;
  const stride = segU + 1;
  for (let iv = 0; iv < segV; iv++) {
    for (let iu = 0; iu < segU; iu++) {
      const a = iv * stride + iu;
      const b = a + stride;
      const c = a + 1;
      const d = b + 1;
      indices[idx++] = a; indices[idx++] = b; indices[idx++] = c;
      indices[idx++] = c; indices[idx++] = b; indices[idx++] = d;
    }
  }

  return packMeshData(positions, uvs, indices);
}

/**
 * Curved tapered horn mesh for demon/samurai/oni avatars.
 */
export function createCurvedHornMesh(side = 1, segU = 16, segV = 18) {
  const vertCount = (segU + 1) * (segV + 1);
  const positions = new Float32Array(vertCount * 3);
  const uvs = new Float32Array(vertCount * 2);
  const indices = new Uint32Array(segU * segV * 6);

  let vIdx = 0;
  let uvIdx = 0;

  for (let iv = 0; iv <= segV; iv++) {
    const t = iv / segV;
    const radius = 0.048 * (1.0 - t * 0.92);
    const cx = side * (0.04 * t + 0.08 * t * t);
    const cy = 0.28 * t - 0.04 * t * t;
    const cz = 0.09 * Math.sin(t * Math.PI * 0.85);

    for (let iu = 0; iu <= segU; iu++) {
      const u = iu / segU;
      const ang = u * Math.PI * 2;
      positions[vIdx++] = cx + Math.cos(ang) * radius;
      positions[vIdx++] = cy;
      positions[vIdx++] = cz + Math.sin(ang) * radius;
      uvs[uvIdx++] = u;
      uvs[uvIdx++] = t;
    }
  }

  let idx = 0;
  const stride = segU + 1;
  for (let iv = 0; iv < segV; iv++) {
    for (let iu = 0; iu < segU; iu++) {
      const a = iv * stride + iu;
      const b = a + stride;
      const c = a + 1;
      const d = b + 1;
      indices[idx++] = a; indices[idx++] = b; indices[idx++] = c;
      indices[idx++] = c; indices[idx++] = b; indices[idx++] = d;
    }
  }

  return packMeshData(positions, uvs, indices);
}

/**
 * Procedural Canvas2D skin/android texture with subtle cheek blush & shading.
 */
export function createSkinTextureCanvas(baseColor = '#e8c4b0', isAndroid = false) {
  const size = 512;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = baseColor;
  ctx.fillRect(0, 0, size, size);

  // Warm subsurface cheek blush on front face (u ~ 0.35 & 0.65, v ~ 0.56)
  if (!isAndroid) {
    const drawBlush = (cx, cy) => {
      const g = ctx.createRadialGradient(cx, cy, 4, cx, cy, 55);
      g.addColorStop(0, 'rgba(244, 114, 128, 0.22)');
      g.addColorStop(1, 'rgba(244, 114, 128, 0.0)');
      ctx.fillStyle = g;
      ctx.fillRect(cx - 60, cy - 60, 120, 120);
    };
    drawBlush(size * 0.41, size * 0.56);
    drawBlush(size * 0.59, size * 0.56);

    // Subtle lip tint
    const lipG = ctx.createRadialGradient(size * 0.5, size * 0.66, 2, size * 0.5, size * 0.66, 28);
    lipG.addColorStop(0, 'rgba(210, 95, 110, 0.38)');
    lipG.addColorStop(1, 'rgba(210, 95, 110, 0.0)');
    ctx.fillStyle = lipG;
    ctx.fillRect(size * 0.4, size * 0.6, size * 0.2, size * 0.12);
  } else {
    // Sleek synthetic android panel lines
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(size * 0.42, size * 0.52);
    ctx.lineTo(size * 0.36, size * 0.72);
    ctx.moveTo(size * 0.58, size * 0.52);
    ctx.lineTo(size * 0.64, size * 0.72);
    ctx.stroke();
  }
  return c;
}

/**
 * Procedural Iris + Pupil + Highlight texture for realistic eyes.
 */
export function createEyeTextureCanvas(irisColor = '#38bdf8') {
  const size = 256;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 0, size, size);

  const cx = size * 0.5;
  const cy = size * 0.5;

  // Outer limbal ring
  ctx.beginPath();
  ctx.arc(cx, cy, 86, 0, Math.PI * 2);
  ctx.fillStyle = '#0f172a';
  ctx.fill();

  // Colored iris gradient
  const grad = ctx.createRadialGradient(cx, cy, 18, cx, cy, 80);
  grad.addColorStop(0, irisColor);
  grad.addColorStop(0.75, irisColor);
  grad.addColorStop(1, '#020617');
  ctx.beginPath();
  ctx.arc(cx, cy, 80, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();

  // Pupil
  ctx.beginPath();
  ctx.arc(cx, cy, 34, 0, Math.PI * 2);
  ctx.fillStyle = '#020617';
  ctx.fill();

  // Specular catchlight
  ctx.beginPath();
  ctx.arc(cx - 22, cy - 22, 13, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
  ctx.fill();

  return c;
}
