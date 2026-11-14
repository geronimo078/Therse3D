import {
  Engine,
  Box,
  Sphere,
  Cylinder,
  Cone,
  Plane,
  Torus,
  Group,
  Light,
  quat,
} from '../../src/index.ts';

const scoreVal = document.getElementById('scoreVal');
const ringVal = document.getElementById('ringVal');
const toast = document.getElementById('toast');
const statusMsg = document.getElementById('statusMsg');
const btnReset = document.getElementById('btnReset');
const btnPause = document.getElementById('btnPause');

const COLOR_ACTIVE = [1.0, 0.78, 0.14, 1.0];
const COLOR_NEXT = [0.22, 0.74, 0.97, 1.0];
const COLOR_CLEARED = [0.13, 0.77, 0.37, 0.45];

try {
  const app = await Engine.create('#canvas', { preset: 'outdoor' });

  // Bright Sun & Volumetric Sky Clouds
  app.add(Light({
    id: 'sun-light',
    kind: 'directional',
    position: [32, 26, -24],
    color: '#fff6e0',
    intensity: 2.6,
  }));
  app.add(Sphere({
    id: 'sun-disc',
    position: [32, 26, -24],
    radius: 2.2,
    color: '#fff3c4',
    emissive: 3.5,
  }));

  app.setVolumetric({
    enabled: true,
    fogDensity: 0.007,
    fogColor: '#93c5fd',
    fogHeight: 18.0,
    cloudMode: 'planar',
    cloudAltitude: [14.0, 23.0],
    cloudDensity: 1.55,
    cloudCoverage: 0.46,
    cloudScale: 0.16,
    cloudSpeed: 0.35,
    scatter: 0.65,
    godRays: 0.65,
  });

  // Ocean Floor
  app.add(Plane({
    id: 'ocean',
    position: [0, -16, 0],
    size: [260, 1, 260],
    color: '#0e4c8a',
    metalness: 0.15,
    roughness: 0.25,
  }));

  // Scenic Archipelago Islands below the flight path
  const islands = [
    { x: 0, z: -28, r: 9.5, h: 3.2 },
    { x: -24, z: -65, r: 12.0, h: 4.5 },
    { x: 26, z: -95, r: 10.5, h: 3.8 },
    { x: -18, z: -140, r: 14.0, h: 5.2 },
    { x: 34, z: -38, r: 8.5, h: 2.6 },
    { x: -42, z: -15, r: 11.0, h: 3.5 },
    { x: 14, z: 22, r: 9.0, h: 2.9 },
  ];
  islands.forEach((isl, i) => {
    app.add(Cylinder({
      id: `island-sand-${i}`,
      position: [isl.x, -15.6, isl.z],
      size: [isl.r * 2, 0.8, isl.r * 2],
      color: '#e2d19e',
      roughness: 0.9,
    }));
    app.add(Cone({
      id: `island-hill-${i}`,
      position: [isl.x, -15.2 + isl.h * 0.5, isl.z],
      size: [isl.r * 1.35, isl.h, isl.r * 1.35],
      color: '#2f7a42',
      roughness: 0.85,
    }));
    // Decorative trees on each island
    for (let t = 0; t < 3; t++) {
      const ang = t * 2.1 + i;
      const dist = isl.r * 0.42;
      const tx = isl.x + Math.cos(ang) * dist;
      const tz = isl.z + Math.sin(ang) * dist;
      const ty = -14.8 + isl.h * 0.28;
      app.add(Cylinder({
        id: `isl-tree-trunk-${i}-${t}`,
        position: [tx, ty, tz],
        size: [0.32, 1.1, 0.32],
        material: 'oak-wood',
      }));
      app.add(Cone({
        id: `isl-tree-top-${i}-${t}`,
        position: [tx, ty + 1.0, tz],
        size: [1.3, 2.0, 1.3],
        color: '#1b4332',
        roughness: 0.85,
      }));
    }
  });

  // Player Airplane (facing -Z in local space)
  const plane = app.add(Group({ id: 'plane', position: [0, 2, 0] }));
  const fuselage = app.add(Cylinder({
    id: 'plane-fuselage',
    parent: 'plane',
    position: [0, 0, 0.05],
    size: [0.42, 1.75, 0.42],
    color: '#ef4444',
    metalness: 0.35,
    roughness: 0.3,
  }));
  quat.fromEuler(fuselage.rotation, Math.PI * 0.5, 0, 0);

  const nose = app.add(Cone({
    id: 'plane-nose',
    parent: 'plane',
    position: [0, 0, -1.05],
    size: [0.40, 0.60, 0.40],
    color: '#f8fafc',
    metalness: 0.4,
    roughness: 0.25,
  }));
  quat.fromEuler(nose.rotation, -Math.PI * 0.5, 0, 0);

  app.add(Sphere({
    id: 'plane-canopy',
    parent: 'plane',
    position: [0, 0.20, -0.18],
    scale: [0.22, 0.18, 0.42],
    material: 'frosted-glass',
    color: [0.55, 0.85, 1.0, 0.65],
  }));

  app.add(Box({
    id: 'plane-wing',
    parent: 'plane',
    position: [0, -0.02, -0.10],
    size: [2.7, 0.07, 0.55],
    color: '#f8fafc',
    metalness: 0.25,
    roughness: 0.35,
  }));

  app.add(Box({
    id: 'plane-tail-h',
    parent: 'plane',
    position: [0, 0.04, 0.82],
    size: [1.0, 0.05, 0.30],
    color: '#ef4444',
    roughness: 0.35,
  }));

  app.add(Box({
    id: 'plane-tail-v',
    parent: 'plane',
    position: [0, 0.28, 0.82],
    size: [0.06, 0.52, 0.32],
    color: '#ef4444',
    roughness: 0.35,
  }));

  const prop = app.add(Box({
    id: 'plane-prop',
    parent: 'plane',
    position: [0, 0, -1.36],
    size: [0.85, 0.08, 0.03],
    color: '#1e293b',
    metalness: 0.6,
    roughness: 0.3,
  }));

  // 12-Ring Aerial Slalom Course (closed loop!)
  const ringCount = 12;
  const rings = [];
  for (let i = 0; i < ringCount; i++) {
    const t = (i / ringCount) * Math.PI * 2;
    const rx = Math.sin(t) * 42 + Math.sin(t * 2) * 14;
    const rz = -35 - (1 - Math.cos(t)) * 38;
    const ry = 2.0 + Math.sin(t * 2 + 0.5) * 4.8;

    // Face ring tangent along the loop
    const tNext = ((i + 0.1) / ringCount) * Math.PI * 2;
    const nx = Math.sin(tNext) * 42 + Math.sin(tNext * 2) * 14;
    const nz = -35 - (1 - Math.cos(tNext)) * 38;
    const yaw = Math.atan2(-(nx - rx), -(nz - rz));

    const handle = app.add(Torus({
      id: `ring-${i}`,
      position: [rx, ry, rz],
      scale: [7.5, 2.2, 7.5],
      color: i === 0 ? COLOR_ACTIVE : COLOR_NEXT,
      emissive: i === 0 ? 2.2 : 0.9,
      roughness: 0.25,
    }));
    quat.fromEuler(handle.rotation, Math.PI * 0.5, yaw, 0);

    rings.push({ handle, x: rx, y: ry, z: rz, yaw, cleared: false });
  }

  // Game State
  let score = 0;
  let activeIdx = 0;
  let lap = 1;
  let paused = false;
  let toastTimer = 0;

  let px = 0;
  let py = 2;
  let pz = -12;
  let yaw = 0;
  let pitch = 0;
  let roll = 0;

  let camX = 0;
  let camY = 4.2;
  let camZ = -4;

  const keys = new Set();
  window.addEventListener('keydown', (e) => {
    keys.add(e.code);
    if (e.code === 'KeyR') resetCourse();
  });
  window.addEventListener('keyup', (e) => keys.delete(e.code));

  // Mouse Steering (Virtual Flight Stick) + Left Click Turbo Boost
  let mouseSteerActive = false;
  let mouseNormX = 0; // -1 (left) .. +1 (right)
  let mouseNormY = 0; // -1 (top) .. +1 (bottom)
  let mouseBoost = false;
  const canvasEl = app.gpu.canvas;
  canvasEl.addEventListener('mousemove', (e) => {
    const rect = canvasEl.getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / Math.max(1, rect.width)) * 2 - 1;
    const ny = ((e.clientY - rect.top) / Math.max(1, rect.height)) * 2 - 1;
    mouseNormX = Math.max(-1, Math.min(1, nx));
    mouseNormY = Math.max(-1, Math.min(1, ny));
    mouseSteerActive = true;
  });
  canvasEl.addEventListener('mouseleave', () => {
    mouseSteerActive = false;
    mouseBoost = false;
  });
  canvasEl.addEventListener('mousedown', (e) => {
    if (e.button === 0) mouseBoost = true;
  });
  window.addEventListener('mouseup', (e) => {
    if (e.button === 0) mouseBoost = false;
  });

  function updateRingColors() {
    for (let i = 0; i < rings.length; i++) {
      const r = rings[i];
      if (i === activeIdx) {
        r.handle.color = COLOR_ACTIVE;
        app.store.pbrParams[r.handle.id * 4 + 2] = 2.2;
      } else if (r.cleared) {
        r.handle.color = COLOR_CLEARED;
        app.store.pbrParams[r.handle.id * 4 + 2] = 0.4;
      } else {
        r.handle.color = COLOR_NEXT;
        app.store.pbrParams[r.handle.id * 4 + 2] = 0.9;
      }
    }
    ringVal.textContent = `${activeIdx + 1} / ${ringCount}`;
  }

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    toastTimer = 1.1;
  }

  function resetCourse() {
    score = 0;
    lap = 1;
    activeIdx = 0;
    px = 0;
    py = 2;
    pz = -12;
    yaw = 0;
    pitch = 0;
    roll = 0;
    camX = 0;
    camY = 4.2;
    camZ = -4;
    rings.forEach((r) => { r.cleared = false; });
    scoreVal.textContent = '0';
    statusMsg.textContent = `Lap ${lap}`;
    updateRingColors();
  }

  btnReset.addEventListener('click', resetCourse);
  btnPause.addEventListener('click', () => {
    paused = !paused;
    btnPause.textContent = paused ? 'Resume' : 'Pause';
  });

  updateRingColors();
  statusMsg.textContent = 'Lap 1';
  app.camera.position[0] = camX;
  app.camera.position[1] = camY;
  app.camera.position[2] = camZ;
  app.camera.target[0] = px;
  app.camera.target[1] = py + 0.4;
  app.camera.target[2] = pz - 5.0;

  app.onTick((dt, elapsed) => {
    if (toastTimer > 0) {
      toastTimer -= dt;
      if (toastTimer <= 0) toast.classList.remove('show');
    }

    if (paused) return;

    // Flight controls (Keyboard + Mouse Virtual Stick)
    const deadzone = 0.08;
    const applyDeadzone = (v) => (Math.abs(v) < deadzone ? 0 : (v - Math.sign(v) * deadzone) / (1 - deadzone));
    const mouseTurn = mouseSteerActive ? -applyDeadzone(mouseNormX) : 0;
    const mousePitch = mouseSteerActive ? -applyDeadzone(mouseNormY) : 0;

    const keyTurn =
      (keys.has('KeyA') || keys.has('ArrowLeft') ? 1 : 0) -
      (keys.has('KeyD') || keys.has('ArrowRight') ? 1 : 0);
    const keyPitch =
      (keys.has('KeyW') || keys.has('ArrowUp') ? 1 : 0) -
      (keys.has('KeyS') || keys.has('ArrowDown') ? 1 : 0);

    const turnInput = Math.max(-1, Math.min(1, keyTurn !== 0 ? keyTurn : mouseTurn));
    const pitchInput = Math.max(-1, Math.min(1, keyPitch !== 0 ? keyPitch : mousePitch));
    const boost = mouseBoost || keys.has('Space') || keys.has('ShiftLeft') || keys.has('ShiftRight');

    const speed = boost ? 24.0 : 14.5;
    yaw += turnInput * 1.45 * dt;
    pitch = Math.max(-0.75, Math.min(0.75, pitch + pitchInput * 1.25 * dt));
    if (pitchInput === 0) {
      pitch *= Math.exp(-1.5 * dt); // gentle auto-level
    }

    const targetRoll = turnInput * 0.65;
    roll += (targetRoll - roll) * (1 - Math.exp(-8 * dt));

    const cosP = Math.cos(pitch);
    const fx = -Math.sin(yaw) * cosP;
    const fy = Math.sin(pitch);
    const fz = -Math.cos(yaw) * cosP;

    const prevPx = px;
    const prevPy = py;
    const prevPz = pz;

    px += fx * speed * dt;
    py += fy * speed * dt;
    pz += fz * speed * dt;

    // Ocean safety bounce
    if (py < -13.5) {
      py = -13.5;
      pitch = Math.max(0.15, pitch);
    }
    if (py > 26) {
      py = 26;
      pitch = Math.min(-0.05, pitch);
    }

    plane.setPosition(px, py, pz);
    quat.fromEuler(plane.rotation, pitch, yaw, roll);
    quat.fromEuler(prop.rotation, 0, 0, elapsed * (boost ? 48 : 28));

    // Pulse active ring gently
    const targetRing = rings[activeIdx];
    const pulse = 7.5 + Math.sin(elapsed * 5.0) * 0.35;
    targetRing.handle.setScale(pulse, 2.2, pulse);

    // Check segment-to-ring pass across ALL uncleared rings
    const segX = px - prevPx;
    const segY = py - prevPy;
    const segZ = pz - prevPz;
    const segLenSq = segX * segX + segY * segY + segZ * segZ;
    const hitRadiusSq = 3.4 * 3.4;

    for (let i = 0; i < ringCount; i++) {
      const r = rings[i];
      if (r.cleared) continue;

      let u = 1;
      if (segLenSq > 1e-6) {
        u = ((r.x - prevPx) * segX + (r.y - prevPy) * segY + (r.z - prevPz) * segZ) / segLenSq;
        u = Math.max(0, Math.min(1, u));
      }
      const cx = prevPx + segX * u;
      const cy = prevPy + segY * u;
      const cz = prevPz + segZ * u;
      const dx = cx - r.x;
      const dy = cy - r.y;
      const dz = cz - r.z;

      if (dx * dx + dy * dy + dz * dz < hitRadiusSq) {
        score += 1;
        scoreVal.textContent = String(score);
        r.cleared = true;
        r.handle.setScale(7.5, 2.2, 7.5);

        const nextIdx = rings.findIndex((ring, idx) => idx > i && !ring.cleared);
        const fallbackIdx = nextIdx !== -1 ? nextIdx : rings.findIndex((ring) => !ring.cleared);
        if (fallbackIdx === -1) {
          activeIdx = 0;
          lap++;
          rings.forEach((ring) => { ring.cleared = false; });
          showToast(`Lap ${lap - 1} Complete! +1 Point`);
          statusMsg.textContent = `Lap ${lap}`;
        } else {
          activeIdx = fallbackIdx;
          showToast('+1 Point!');
        }
        updateRingColors();
      }
    }

    // Smooth Chase Camera
    const desiredCamX = px - fx * 6.8;
    const desiredCamY = py - fy * 4.5 + 2.1;
    const desiredCamZ = pz - fz * 6.8;
    const camLerp = 1 - Math.exp(-7.5 * dt);
    camX += (desiredCamX - camX) * camLerp;
    camY += (desiredCamY - camY) * camLerp;
    camZ += (desiredCamZ - camZ) * camLerp;

    app.camera.position[0] = camX;
    app.camera.position[1] = camY;
    app.camera.position[2] = camZ;
    app.camera.target[0] = px + fx * 5.0;
    app.camera.target[1] = py + fy * 3.0 + 0.4;
    app.camera.target[2] = pz + fz * 5.0;
  });
} catch (err) {
  statusMsg.textContent = err instanceof Error ? err.message : String(err);
}

