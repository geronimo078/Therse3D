/**
 * Zero-Allocation Mathe-Helfer (vec3, mat4, quat, aabb).
 * Alle Funktionen schreiben direkt in übergebene Ziel-Arrays/TypedArrays, um GC-Druck im Render-Loop zu vermeiden.
 */

export type Vec3 = Float32Array | [number, number, number];
export type Quat = Float32Array | [number, number, number, number];
export type Mat4 = Float32Array;

export const vec3 = {
  set(out: Vec3, x: number, y: number, z: number, offset = 0): Vec3 {
    out[offset] = x;
    out[offset + 1] = y;
    out[offset + 2] = z;
    return out;
  },

  copy(out: Vec3, a: Vec3, outOffset = 0, aOffset = 0): Vec3 {
    out[outOffset] = a[aOffset] ?? 0;
    out[outOffset + 1] = a[aOffset + 1] ?? 0;
    out[outOffset + 2] = a[aOffset + 2] ?? 0;
    return out;
  },

  add(out: Vec3, a: Vec3, b: Vec3): Vec3 {
    out[0] = (a[0] ?? 0) + (b[0] ?? 0);
    out[1] = (a[1] ?? 0) + (b[1] ?? 0);
    out[2] = (a[2] ?? 0) + (b[2] ?? 0);
    return out;
  },

  sub(out: Vec3, a: Vec3, b: Vec3): Vec3 {
    out[0] = (a[0] ?? 0) - (b[0] ?? 0);
    out[1] = (a[1] ?? 0) - (b[1] ?? 0);
    out[2] = (a[2] ?? 0) - (b[2] ?? 0);
    return out;
  },

  scale(out: Vec3, a: Vec3, s: number): Vec3 {
    out[0] = (a[0] ?? 0) * s;
    out[1] = (a[1] ?? 0) * s;
    out[2] = (a[2] ?? 0) * s;
    return out;
  },

  dot(a: Vec3, b: Vec3): number {
    return (a[0] ?? 0) * (b[0] ?? 0) + (a[1] ?? 0) * (b[1] ?? 0) + (a[2] ?? 0) * (b[2] ?? 0);
  },

  cross(out: Vec3, a: Vec3, b: Vec3): Vec3 {
    const ax = a[0] ?? 0, ay = a[1] ?? 0, az = a[2] ?? 0;
    const bx = b[0] ?? 0, by = b[1] ?? 0, bz = b[2] ?? 0;
    out[0] = ay * bz - az * by;
    out[1] = az * bx - ax * bz;
    out[2] = ax * by - ay * bx;
    return out;
  },

  length(a: Vec3): number {
    const x = a[0] ?? 0;
    const y = a[1] ?? 0;
    const z = a[2] ?? 0;
    return Math.hypot(x, y, z);
  },

  normalize(out: Vec3, a: Vec3): Vec3 {
    const x = a[0] ?? 0;
    const y = a[1] ?? 0;
    const z = a[2] ?? 0;
    const len = Math.hypot(x, y, z);
    if (len > 1e-8) {
      const inv = 1 / len;
      out[0] = x * inv;
      out[1] = y * inv;
      out[2] = z * inv;
    } else {
      out[0] = 0;
      out[1] = 0;
      out[2] = 0;
    }
    return out;
  },
};

export const quat = {
  identity(out: Quat, offset = 0): Quat {
    out[offset] = 0;
    out[offset + 1] = 0;
    out[offset + 2] = 0;
    out[offset + 3] = 1;
    return out;
  },

  fromEuler(out: Quat, rx: number, ry: number, rz: number, offset = 0): Quat {
    const hx = rx * 0.5;
    const hy = ry * 0.5;
    const hz = rz * 0.5;

    const sx = Math.sin(hx);
    const cx = Math.cos(hx);
    const sy = Math.sin(hy);
    const cy = Math.cos(hy);
    const sz = Math.sin(hz);
    const cz = Math.cos(hz);

    out[offset] = sx * cy * cz - cx * sy * sz;
    out[offset + 1] = cx * sy * cz + sx * cy * sz;
    out[offset + 2] = cx * cy * sz - sx * sy * cz;
    out[offset + 3] = cx * cy * cz + sx * sy * sz;
    return out;
  },

  fromLookAt(
    out: Quat,
    offset: number,
    eyeX: number,
    eyeY: number,
    eyeZ: number,
    targetX: number,
    targetY: number,
    targetZ: number,
  ): Quat {
    let fx = targetX - eyeX;
    let fy = targetY - eyeY;
    let fz = targetZ - eyeZ;
    let len = Math.hypot(fx, fy, fz);
    if (len < 1e-6) {
      return quat.identity(out, offset);
    }
    fx /= len; fy /= len; fz /= len;

    // Forward is +Z in local space; fallback up if parallel to Y
    const upX = 0;
    const upY = Math.abs(fy) > 0.999 ? 0 : 1;
    const upZ = Math.abs(fy) > 0.999 ? 1 : 0;

    // right = normalize(cross(up, forward))
    let rx = upY * fz - upZ * fy;
    let ry = upZ * fx - upX * fz;
    let rz = upX * fy - upY * fx;
    len = Math.hypot(rx, ry, rz);
    if (len < 1e-6) {
      return quat.identity(out, offset);
    }
    rx /= len; ry /= len; rz /= len;

    // trueUp = cross(forward, right)
    const ux = fy * rz - fz * ry;
    const uy = fz * rx - fx * rz;
    const uz = fx * ry - fy * rx;

    // Rotation matrix columns: [right, trueUp, forward] -> quaternion
    const trace = rx + uy + fz;
    if (trace > 0) {
      const s = 0.5 / Math.sqrt(trace + 1.0);
      out[offset + 3] = 0.25 / s;
      out[offset + 0] = (uz - ry) * s;
      out[offset + 1] = (fx - rz) * s;
      out[offset + 2] = (rx - uy) * s; // wait: (ry - ux) * s
    } else if (rx > uy && rx > fz) {
      const s = 2.0 * Math.sqrt(1.0 + rx - uy - fz);
      out[offset + 3] = (uz - ry) / s;
      out[offset + 0] = 0.25 * s;
      out[offset + 1] = (ux + ry) / s;
      out[offset + 2] = (fx + rz) / s;
    } else if (uy > fz) {
      const s = 2.0 * Math.sqrt(1.0 + uy - rx - fz);
      out[offset + 3] = (fx - rz) / s;
      out[offset + 0] = (ux + ry) / s;
      out[offset + 1] = 0.25 * s;
      out[offset + 2] = (fy + uz) / s;
    } else {
      const s = 2.0 * Math.sqrt(1.0 + fz - rx - uy);
      out[offset + 3] = (ry - ux) / s;
      out[offset + 0] = (fx + rz) / s;
      out[offset + 1] = (fy + uz) / s;
      out[offset + 2] = 0.25 * s;
    }

    // Fix trace > 0 z component: (ry - ux) * s
    if (trace > 0) {
      const s = 0.5 / Math.sqrt(trace + 1.0);
      out[offset + 2] = (ry - ux) * s;
    }
    return out;
  },
};

export const mat4 = {
  identity(out: Mat4, offset = 0): Mat4 {
    out[offset + 0] = 1; out[offset + 1] = 0; out[offset + 2] = 0; out[offset + 3] = 0;
    out[offset + 4] = 0; out[offset + 5] = 1; out[offset + 6] = 0; out[offset + 7] = 0;
    out[offset + 8] = 0; out[offset + 9] = 0; out[offset + 10] = 1; out[offset + 11] = 0;
    out[offset + 12] = 0; out[offset + 13] = 0; out[offset + 14] = 0; out[offset + 15] = 1;
    return out;
  },

  fromTranslationRotationScale(
    out: Mat4,
    outOffset: number,
    px: number,
    py: number,
    pz: number,
    qx: number,
    qy: number,
    qz: number,
    qw: number,
    sx: number,
    sy: number,
    sz: number,
  ): Mat4 {
    const x2 = qx + qx;
    const y2 = qy + qy;
    const z2 = qz + qz;
    const xx = qx * x2;
    const xy = qx * y2;
    const xz = qx * z2;
    const yy = qy * y2;
    const yz = qy * z2;
    const zz = qz * z2;
    const wx = qw * x2;
    const wy = qw * y2;
    const wz = qw * z2;

    out[outOffset + 0] = (1 - (yy + zz)) * sx;
    out[outOffset + 1] = (xy + wz) * sx;
    out[outOffset + 2] = (xz - wy) * sx;
    out[outOffset + 3] = 0;

    out[outOffset + 4] = (xy - wz) * sy;
    out[outOffset + 5] = (1 - (xx + zz)) * sy;
    out[outOffset + 6] = (yz + wx) * sy;
    out[outOffset + 7] = 0;

    out[outOffset + 8] = (xz + wy) * sz;
    out[outOffset + 9] = (yz - wx) * sz;
    out[outOffset + 10] = (1 - (xx + yy)) * sz;
    out[outOffset + 11] = 0;

    out[outOffset + 12] = px;
    out[outOffset + 13] = py;
    out[outOffset + 14] = pz;
    out[outOffset + 15] = 1;

    return out;
  },

  multiply(out: Mat4, a: Mat4, b: Mat4): Mat4 {
    const a00 = a[0] ?? 0, a01 = a[1] ?? 0, a02 = a[2] ?? 0, a03 = a[3] ?? 0;
    const a10 = a[4] ?? 0, a11 = a[5] ?? 0, a12 = a[6] ?? 0, a13 = a[7] ?? 0;
    const a20 = a[8] ?? 0, a21 = a[9] ?? 0, a22 = a[10] ?? 0, a23 = a[11] ?? 0;
    const a30 = a[12] ?? 0, a31 = a[13] ?? 0, a32 = a[14] ?? 0, a33 = a[15] ?? 0;

    let b0 = b[0] ?? 0, b1 = b[1] ?? 0, b2 = b[2] ?? 0, b3 = b[3] ?? 0;
    out[0] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

    b0 = b[4] ?? 0; b1 = b[5] ?? 0; b2 = b[6] ?? 0; b3 = b[7] ?? 0;
    out[4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[5] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[6] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[7] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

    b0 = b[8] ?? 0; b1 = b[9] ?? 0; b2 = b[10] ?? 0; b3 = b[11] ?? 0;
    out[8] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[9] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[10] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[11] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;

    b0 = b[12] ?? 0; b1 = b[13] ?? 0; b2 = b[14] ?? 0; b3 = b[15] ?? 0;
    out[12] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
    out[13] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
    out[14] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
    out[15] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
    return out;
  },

  perspective(out: Mat4, fovYRadians: number, aspect: number, near: number, far: number): Mat4 {
    const f = 1.0 / Math.tan(fovYRadians * 0.5);
    const nf = 1.0 / (near - far);
    out[0] = f / Math.max(aspect, 1e-6);
    out[1] = 0; out[2] = 0; out[3] = 0;
    out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0;
    out[8] = 0; out[9] = 0; out[10] = far * nf; out[11] = -1;
    out[12] = 0; out[13] = 0; out[14] = far * near * nf; out[15] = 0;
    return out;
  },

  lookAt(
    out: Mat4,
    eyeX: number, eyeY: number, eyeZ: number,
    targetX: number, targetY: number, targetZ: number,
    upX = 0, upY = 1, upZ = 0,
  ): Mat4 {
    let zx = eyeX - targetX;
    let zy = eyeY - targetY;
    let zz = eyeZ - targetZ;
    let len = Math.hypot(zx, zy, zz);
    if (len < 1e-8) {
      zz = 1;
      len = 1;
    }
    zx /= len; zy /= len; zz /= len;

    let xx = upY * zz - upZ * zy;
    let xy = upZ * zx - upX * zz;
    let xz = upX * zy - upY * zx;
    len = Math.hypot(xx, xy, xz);
    if (len > 1e-8) {
      xx /= len; xy /= len; xz /= len;
    } else {
      xx = 1; xy = 0; xz = 0;
    }

    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;

    out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
    out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
    out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
    out[12] = -(xx * eyeX + xy * eyeY + xz * eyeZ);
    out[13] = -(yx * eyeX + yy * eyeY + yz * eyeZ);
    out[14] = -(zx * eyeX + zy * eyeY + zz * eyeZ);
    out[15] = 1;
    return out;
  },
};

export const aabb = {
  fromCenterHalfExtents(
    minOut: Float32Array,
    maxOut: Float32Array,
    offset: number,
    cx: number,
    cy: number,
    cz: number,
    hx: number,
    hy: number,
    hz: number,
  ): void {
    minOut[offset] = cx - hx;
    minOut[offset + 1] = cy - hy;
    minOut[offset + 2] = cz - hz;
    maxOut[offset] = cx + hx;
    maxOut[offset + 1] = cy + hy;
    maxOut[offset + 2] = cz + hz;
  },
};
