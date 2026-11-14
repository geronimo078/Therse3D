import { Engine, McpBridge, OrbitControls } from '../../src/index.ts';

const output = document.getElementById('output');
const snapImg = document.getElementById('snapImg');
const statusBadge = document.getElementById('statusBadge');
const statEntities = document.getElementById('statEntities');
const statAgv = document.getElementById('statAgv');
const statWarnings = document.getElementById('statWarnings');
const btnModeHazard = document.getElementById('btnModeHazard');

/**
 * Declarative factory scene using SpatialLayout (grid, on, beside) + studio-grid.
 * In Three.js this would require manual hierarchy, bounding box math, and light/material boilerplate.
 */
const FACTORY_SCENE = {
  preset: 'studio',
  transition: 0.45,
  camera: { target: 'rack-2', angle: 'isometric', distance: 13.5 },
  entities: [
    { id: 'floor', type: 'plane', size: [22, 1, 22], position: [0, 0, 0], material: 'studio-grid' },

    // 6 Warehouse racks arranged automatically via grid layout below
    { id: 'rack-1', type: 'box', size: [1.5, 2.2, 0.9], position: [0, 1.1, 0], material: 'brushed-aluminum' },
    { id: 'rack-2', type: 'box', size: [1.5, 2.2, 0.9], position: [0, 1.1, 0], material: 'brushed-aluminum' },
    { id: 'rack-3', type: 'box', size: [1.5, 2.2, 0.9], position: [0, 1.1, 0], material: 'brushed-aluminum' },
    { id: 'rack-4', type: 'box', size: [1.5, 2.2, 0.9], position: [0, 1.1, 0], material: 'brushed-aluminum' },
    { id: 'rack-5', type: 'box', size: [1.5, 2.2, 0.9], position: [0, 1.1, 0], material: 'brushed-aluminum' },
    { id: 'rack-6', type: 'box', size: [1.5, 2.2, 0.9], position: [0, 1.1, 0], material: 'brushed-aluminum' },

    // Cargo crates placed declaratively ON top of racks
    { id: 'cargo-1', type: 'box', size: [0.65, 0.55, 0.65], on: 'rack-1', align: 'center', material: 'oak-wood' },
    { id: 'cargo-2', type: 'box', size: [0.65, 0.55, 0.65], on: 'rack-3', align: 'center', material: 'oak-wood' },
    { id: 'cargo-3', type: 'box', size: [0.65, 0.55, 0.65], on: 'rack-4', align: 'center', material: 'polished-gold' },
    { id: 'cargo-4', type: 'box', size: [0.65, 0.55, 0.65], on: 'rack-6', align: 'center', material: 'oak-wood' },

    // Sorting conveyor & packing bench placed BESIDE it
    { id: 'conveyor-main', type: 'box', size: [5.4, 0.65, 1.2], position: [0, 0.325, 5.2], material: 'carbon-fiber' },
    {
      id: 'packing-bench',
      type: 'box',
      size: [1.6, 0.85, 1.2],
      beside: 'conveyor-main',
      side: 'right',
      margin: 0.4,
      material: 'oak-wood',
    },
    { id: 'sorter-orb', type: 'sphere', radius: 0.28, on: 'conveyor-main', align: 'center', material: 'emissive-neon' },

    // Quarantine / Safe Buffer Pad for AI obstacle relocation
    { id: 'quarantine-pad', type: 'cylinder', size: [1.6, 0.12, 1.6], position: [-6.4, 0.06, 5.0], material: 'matte-ceramic' },
    { id: 'dock-beacon', type: 'cylinder', size: [0.24, 1.2, 0.24], position: [-6.4, 0.6, 6.1], material: 'emissive-neon', color: '#22d3ee' },

    // Two autonomous AGV patrol bots
    { id: 'agv-1', type: 'box', size: [0.95, 0.42, 1.25], position: [-5.2, 0.21, 1.9], material: 'carbon-fiber', color: '#0ea5e9' },
    { id: 'agv-1-beacon', type: 'sphere', radius: 0.14, on: 'agv-1', align: 'center', material: 'emissive-neon', color: '#22c55e' },

    { id: 'agv-2', type: 'box', size: [0.95, 0.42, 1.25], position: [5.2, 0.21, -2.0], material: 'carbon-fiber', color: '#6366f1' },
    { id: 'agv-2-beacon', type: 'sphere', radius: 0.14, on: 'agv-2', align: 'center', material: 'emissive-neon', color: '#22c55e' },
  ],
  layouts: [
    { kind: 'grid', entities: ['rack-1', 'rack-2', 'rack-3', 'rack-4', 'rack-5', 'rack-6'], columns: 3, spacing: 3.6 },
  ],
};

function setLog(title, payload) {
  const body = typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2);
  output.textContent = `[${title}]\n${body}`;
}

function updateBadges(sceneDesc, warnings = [], blocked = false) {
  statEntities.textContent = String(sceneDesc?.entityCount ?? 0);
  statWarnings.textContent = String(warnings.length);
  statAgv.textContent = blocked ? 'BLOCKED' : 'ACTIVE';
  statAgv.style.color = blocked ? '#f87171' : '#4ade80';

  if (blocked || warnings.length > 0) {
    statusBadge.textContent = blocked ? 'HAZARD DETECTED' : `${warnings.length} WARNINGS`;
    statusBadge.className = 'badge warn';
  } else {
    statusBadge.textContent = 'NOMINAL';
    statusBadge.className = 'badge';
  }
}

async function init() {
  try {
    const app = await Engine.create('#canvas', { preset: 'studio' });
    app.use(OrbitControls({ distance: 13.5, target: [0, 1.0, 0] }));
    app.setBloom({ enabled: true, threshold: 0.72, intensity: 0.6, radius: 1.5 });
    const mcp = new McpBridge(app);

    let placeHazardMode = false;
    let hazardActive = false;
    let agvBlocked = false;
    let simTime = 0;

    const initialReport = app.sync(FACTORY_SCENE);
    updateBadges(app.describe(), initialReport.warnings, false);
    setLog('therse_sync_scene (initial)', {
      added: initialReport.added,
      applied: initialReport.applied,
      warnings: initialReport.warnings,
      tip: 'Click an entity to inspect, or click "Drop Lane Hazard" to test AI recovery.',
    });

    const placeHazardAt = async (x, z) => {
      hazardActive = true;
      const patchRes = await mcp.handleToolCall('therse_patch_scene', {
        transition: 0.3,
        entities: [
          {
            id: 'hazard-1',
            type: 'box',
            size: [0.85, 0.75, 0.85],
            position: [x, 0.375, z],
            material: 'emissive-neon',
            color: '#ef4444',
          },
        ],
      });
      updateBadges(patchRes.scene, patchRes.report.warnings, agvBlocked);
      setLog('therse_patch_scene (hazard injected)', {
        position: [Number(x.toFixed(2)), 0.38, Number(z.toFixed(2))],
        warnings: patchRes.report.warnings,
        nextStep: 'Click "AI Diagnose & Reroute" to let McpBridge inspect & resolve.',
      });
    };

    // Autonomous AGV patrol loop (zero allocations per frame)
    app.onTick((dt) => {
      const agv1 = app.get('agv-1');
      const agv1Beacon = app.get('agv-1-beacon');
      const agv2 = app.get('agv-2');
      const agv2Beacon = app.get('agv-2-beacon');
      const hazard = app.get('hazard-1');

      let blockedNow = false;
      if (hazardActive && hazard && agv1 && agv2) {
        const hx = hazard.position[0];
        const hz = hazard.position[2];
        const d1 = Math.hypot(agv1.position[0] - hx, agv1.position[2] - hz);
        const d2 = Math.hypot(agv2.position[0] - hx, agv2.position[2] - hz);
        if (d1 < 1.85 || d2 < 1.85) {
          blockedNow = true;
        }
      }

      if (blockedNow !== agvBlocked) {
        agvBlocked = blockedNow;
        updateBadges(app.describe(), [], agvBlocked);
        if (agvBlocked) {
          if (agv1Beacon) agv1Beacon.setColor('#ef4444');
          if (agv2Beacon) agv2Beacon.setColor('#ef4444');
        } else {
          if (agv1Beacon) agv1Beacon.setColor('#22c55e');
          if (agv2Beacon) agv2Beacon.setColor('#22c55e');
        }
      }

      if (!agvBlocked) {
        simTime += dt;
        if (agv1) {
          const x1 = Math.sin(simTime * 0.75) * 5.2;
          const z1 = 2.0 + Math.cos(simTime * 1.5) * 0.45;
          agv1.setPosition(x1, 0.21, z1);
          if (agv1Beacon) agv1Beacon.setPosition(x1, 0.56, z1);
        }
        if (agv2) {
          const x2 = Math.cos(simTime * 0.65) * 5.2;
          const z2 = -2.1 + Math.sin(simTime * 1.3) * 0.45;
          agv2.setPosition(x2, 0.21, z2);
          if (agv2Beacon) agv2Beacon.setPosition(x2, 0.56, z2);
        }
      }
    });

    // Toggle Click Mode
    btnModeHazard.addEventListener('click', () => {
      placeHazardMode = !placeHazardMode;
      btnModeHazard.textContent = placeHazardMode ? 'Click Mode: Place Hazard' : 'Click Mode: Inspect';
      btnModeHazard.classList.toggle('active', placeHazardMode);
    });

    // Drop Hazard directly in AGV-1 lane
    document.getElementById('btnDropHazard').addEventListener('click', async () => {
      const agv1 = app.get('agv-1');
      const targetX = agv1 ? Math.max(-4.5, Math.min(4.5, agv1.position[0] + 1.2)) : 0;
      await placeHazardAt(targetX, 2.0);
    });

    // AI Diagnose & Reroute via McpBridge tool calls
    document.getElementById('btnAiDiagnose').addEventListener('click', async () => {
      const scene = await mcp.handleToolCall('therse_inspect');
      const hazardDesc = scene.entities.find((e) => e.id === 'hazard-1');

      if (!hazardDesc) {
        setLog('AI Supervisor (therse_inspect)', {
          status: 'No hazard entity found.',
          entityCount: scene.entityCount,
          warnings: scene.warnings,
        });
        return;
      }

      // 1. Focus camera on the detected hazard
      await mcp.handleToolCall('therse_camera_focus', {
        target: 'hazard-1',
        view: 'front-elevated',
        distance: 6.2,
      });

      // 2. Relocate hazard onto quarantine-pad with avoidOverlap + smooth transition
      const patchRes = await mcp.handleToolCall('therse_patch_scene', {
        transition: 0.55,
        entities: [
          {
            id: 'hazard-1',
            on: 'quarantine-pad',
            align: 'center',
            avoidOverlap: true,
            material: 'polished-gold',
            color: '#f59e0b',
          },
        ],
      });

      hazardActive = false;
      agvBlocked = false;
      const agv1Beacon = app.get('agv-1-beacon');
      const agv2Beacon = app.get('agv-2-beacon');
      if (agv1Beacon) agv1Beacon.setColor('#22c55e');
      if (agv2Beacon) agv2Beacon.setColor('#22c55e');

      // 3. Capture annotated VLM snapshot after resolution
      const snap = await mcp.handleToolCall('therse_snapshot', {
        layers: ['rgb', 'annotated'],
        resolution: [640, 360],
      });
      if (snap.annotatedDataUrl) {
        snapImg.src = snap.annotatedDataUrl;
        snapImg.style.display = 'block';
      }

      updateBadges(patchRes.scene, patchRes.report.warnings, false);
      setLog('AI Supervisor Pipeline (McpBridge)', {
        steps: [
          '1. therse_inspect -> detected hazard-1',
          '2. therse_camera_focus -> target: hazard-1',
          '3. therse_patch_scene -> moved to quarantine-pad (avoidOverlap: true)',
          '4. therse_snapshot -> VLM annotated frame captured',
        ],
        hazardBefore: {
          position: hazardDesc.position,
          relations: hazardDesc.relations,
          screenRegion: hazardDesc.screenRegion,
        },
        patchWarnings: patchRes.report.warnings,
      });
    });

    // Auto-Healing Stress Test: deliberately send broken AI JSON
    document.getElementById('btnBrokenAi').addEventListener('click', async () => {
      const brokenRes = await mcp.handleToolCall('therse_patch_scene', {
        entities: [
          {
            id: 'ai-hallucinated-crate',
            type: 'box',
            size: [-3, 0, Number.NaN],
            on: 'nonexistent-rack-99',
            material: 'unobtanium-9000',
            position: [2.2, 0.5, 3.4],
          },
        ],
      });
      updateBadges(brokenRes.scene, brokenRes.report.warnings, agvBlocked);
      setLog('Auto-Healing Result (warnings[])', {
        note: 'Engine recovered safely from invalid AI JSON without crashing.',
        warnings: brokenRes.report.warnings,
        added: brokenRes.report.added,
      });
    });

    // Inspect
    document.getElementById('btnInspect').addEventListener('click', async () => {
      const desc = await mcp.handleToolCall('therse_inspect');
      updateBadges(desc, desc.warnings, agvBlocked);
      setLog('therse_inspect', {
        entityCount: desc.entityCount,
        warnings: desc.warnings,
        suggestedFixes: desc.suggestedFixes,
        sampleEntities: desc.entities.slice(0, 6).map((e) => ({
          id: e.id,
          screenRegion: e.screenRegion,
          relations: e.relations,
        })),
      });
    });

    // VLM Snapshot
    document.getElementById('btnSnap').addEventListener('click', async () => {
      const snap = await mcp.handleToolCall('therse_snapshot', {
        layers: ['rgb', 'annotated'],
        resolution: [640, 360],
      });
      if (snap.annotatedDataUrl) {
        snapImg.src = snap.annotatedDataUrl;
        snapImg.style.display = 'block';
      }
      setLog('therse_snapshot', {
        width: snap.width,
        height: snap.height,
        annotated: Boolean(snap.annotatedDataUrl),
      });
    });

    // Undo
    document.getElementById('btnUndo').addEventListener('click', async () => {
      const undoRes = await mcp.handleToolCall('therse_undo');
      updateBadges(undoRes.scene, undoRes.report.warnings, agvBlocked);
      setLog('therse_undo', {
        updated: undoRes.report.updated,
        removed: undoRes.report.removed,
        warnings: undoRes.report.warnings,
      });
    });

    // Reset
    document.getElementById('btnReset').addEventListener('click', async () => {
      hazardActive = false;
      agvBlocked = false;
      snapImg.style.display = 'none';
      const res = await mcp.handleToolCall('therse_sync_scene', FACTORY_SCENE);
      updateBadges(res.scene, res.report.warnings, false);
      setLog('therse_sync_scene (reset)', {
        entityCount: res.scene.entityCount,
        warnings: res.report.warnings,
      });
    });

    // Canvas Pick: Inspect entity OR Place Hazard (Shift+Click / Hazard Mode)
    app.gpu.canvas.addEventListener('click', async (e) => {
      const rect = app.gpu.canvas.getBoundingClientRect();
      const hit = app.pick(e.clientX - rect.left, e.clientY - rect.top);
      if (!hit) return;

      if (placeHazardMode || e.shiftKey) {
        await placeHazardAt(hit.point[0], hit.point[2]);
        return;
      }

      const desc = app.describe();
      const entity = desc.entities.find((item) => item.id === hit.id);
      const projected = app.project(hit.id);
      setLog(`app.pick -> "${hit.id}"`, {
        distance: Number(hit.distance.toFixed(2)),
        hitPoint: hit.point.map((v) => Number(v.toFixed(2))),
        screenPixel: projected ? [Math.round(projected.x), Math.round(projected.y)] : null,
        spatialDescription: entity ?? null,
      });
    });
  } catch (err) {
    setLog('Error', err instanceof Error ? err.message : String(err));
  }
}

void init();

