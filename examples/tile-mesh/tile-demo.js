import { Engine, OrbitControls, TileMeshPlugin } from '../../src/index.ts';

const output = document.getElementById('output');
const statTiles = document.getElementById('statTiles');
const statVerts = document.getElementById('statVerts');
const statTris = document.getElementById('statTris');
const btnSpherical = document.getElementById('btnSpherical');
const btnPlanar = document.getElementById('btnPlanar');
const btnReplaceTile = document.getElementById('btnReplaceTile');
const btnRemoveTile = document.getElementById('btnRemoveTile');
const btnAsyncTile = document.getElementById('btnAsyncTile');
const btnSkirts = document.getElementById('btnSkirts');
const btnBrushMode = document.getElementById('btnBrushMode');

function terrainHeight(lonDeg, latDeg) {
  const radLon = (lonDeg * Math.PI) / 180;
  const radLat = (latDeg * Math.PI) / 180;
  const ridge = Math.abs(Math.sin(radLon * 3.0 + Math.cos(radLat * 2.5)));
  const hills = Math.sin(lonDeg * 0.35) * Math.cos(latDeg * 0.35);
  return 0.18 * (1 - ridge) + 0.09 * hills;
}

function createTerrainTexture() {
  const size = 256;
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createLinearGradient(0, 0, size, size);
  g.addColorStop(0, '#1e5f74');
  g.addColorStop(0.5, '#2d8a68');
  g.addColorStop(1, '#4b9e5f');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= size; i += 32) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, size);
    ctx.moveTo(0, i);
    ctx.lineTo(size, i);
    ctx.stroke();
  }
  return c;
}

async function init() {
  try {
    const app = await Engine.create('#canvas', { preset: 'outdoor' });
    app.use(OrbitControls({ distance: 6.5, target: [0, 0, 0] }));

    const texId = app.registerTexture(createTerrainTexture());
    let currentMode = 'spherical';
    let skirtsOn = true;
    let selectedTileId = null;
    let brushMode = 'inspect'; // 'inspect' | 'raise' | 'crater'
    let tiles = null;

    const updateHudStats = (note = '') => {
      if (!tiles) return;
      const list = tiles.listTiles();
      let verts = 0;
      let tris = 0;
      const lodCounts = new Map();
      for (const t of list) {
        verts += t.vertexCount;
        tris += t.triangleCount;
        lodCounts.set(t.segments, (lodCounts.get(t.segments) ?? 0) + 1);
      }
      statTiles.textContent = String(list.length);
      statVerts.textContent = String(verts);
      statTris.textContent = String(tris);

      const lodSummary = Array.from(lodCounts.entries())
        .sort((a, b) => b[0] - a[0])
        .map(([seg, cnt]) => `${seg}x${seg}: ${cnt}`)
        .join(' | ');

      const sel = selectedTileId ? tiles.getTile(selectedTileId) : null;
      const selSample = sel
        ? tiles.sampleSurface(
            (sel.bounds.minLon + sel.bounds.maxLon) * 0.5,
            (sel.bounds.minLat + sel.bounds.maxLat) * 0.5,
          )
        : null;
      output.textContent =
        `Mode: ${currentMode.toUpperCase()} | Skirts: ${skirtsOn ? 'ON' : 'OFF'}\n` +
        `LODs: ${lodSummary || 'none'}\n` +
        (sel
          ? `Selected "${sel.id}" (rev ${sel.revision}): ${sel.segments}x${sel.segments}, ${sel.vertexCount} verts, ${sel.triangleCount} tris\n` +
            (selSample
              ? `Surface @ center: elev=${selSample.elevation.toFixed(3)}, slope=${selSample.slopeDeg.toFixed(1)}°, ${selSample.surfaceType}\n`
              : '')
          : 'Click any tile to inspect, or zoom in/out to watch LOD adapt.\n') +
        (note ? `\n${note}` : '');
    };

    const rebuildWorld = (mode) => {
      currentMode = mode;
      selectedTileId = null;
      if (tiles) tiles.dispose();

      const skirtDepth = skirtsOn ? (mode === 'spherical' ? 0.03 : 0.08) : 0.0;
      if (mode === 'spherical') {
        tiles = TileMeshPlugin(app, {
          mode: 'spherical',
          planetRadius: 2.0,
          planetCenter: [0, 0, 0],
          elevationScale: 0.32,
          skirtDepth,
          lodLevels: [
            { maxDistance: 4.5, segments: 24 },
            { maxDistance: 7.5, segments: 12 },
            { maxDistance: 12.0, segments: 6 },
            { maxDistance: Infinity, segments: 4 },
          ],
          defaultHeightSampler: terrainHeight,
        });

        // 6x4 global spherical tile grid covering full lon [-180..180], lat [-90..90]
        const lonSteps = 6;
        const latSteps = 4;
        const dLon = 360 / lonSteps;
        const dLat = 180 / latSteps;
        for (let iy = 0; iy < latSteps; iy++) {
          for (let ix = 0; ix < lonSteps; ix++) {
            const id = `sphere-${ix}-${iy}`;
            tiles.setTile({
              id,
              bounds: {
                minLon: -180 + ix * dLon,
                maxLon: -180 + (ix + 1) * dLon,
                minLat: -90 + iy * dLat,
                maxLat: -90 + (iy + 1) * dLat,
              },
              texture: texId,
              roughness: 0.72,
              metalness: 0.05,
            });
          }
        }
        app.focusCamera({ target: [0, 0, 0], angle: 'front-elevated', distance: 6.5 });
      } else {
        tiles = TileMeshPlugin(app, {
          mode: 'planar',
          originGeo: [0, 0],
          metersPerDegree: 0.55,
          elevationScale: 2.2,
          skirtDepth,
          lodLevels: [
            { maxDistance: 4.5, segments: 24 },
            { maxDistance: 8.5, segments: 12 },
            { maxDistance: Infinity, segments: 6 },
          ],
          defaultHeightSampler: terrainHeight,
        });

        // 4x4 planar tile grid
        const span = 5;
        for (let iy = 0; iy < 4; iy++) {
          for (let ix = 0; ix < 4; ix++) {
            const id = `flat-${ix}-${iy}`;
            tiles.setTile({
              id,
              bounds: {
                minLon: (ix - 2) * span,
                maxLon: (ix - 1) * span,
                minLat: (iy - 2) * span,
                maxLat: (iy - 1) * span,
              },
              texture: texId,
              roughness: 0.75,
              metalness: 0.05,
            });
          }
        }
        app.focusCamera({ target: [0, 0.2, 0], angle: 'front-elevated', distance: 8.5 });
      }
      updateHudStats();
    };

    rebuildWorld('spherical');

    // Refresh LOD stats in HUD whenever camera moves
    let hudTimer = 0;
    app.onTick((dt) => {
      hudTimer += dt;
      if (hudTimer >= 0.25) {
        hudTimer = 0;
        updateHudStats();
      }
    });

    btnSpherical.addEventListener('click', () => {
      btnSpherical.classList.add('active');
      btnPlanar.classList.remove('active');
      rebuildWorld('spherical');
    });

    btnPlanar.addEventListener('click', () => {
      btnPlanar.classList.add('active');
      btnSpherical.classList.remove('active');
      rebuildWorld('planar');
    });

    btnSkirts.addEventListener('click', () => {
      skirtsOn = !skirtsOn;
      btnSkirts.textContent = `Skirts: ${skirtsOn ? 'ON' : 'OFF'}`;
      btnSkirts.classList.toggle('active', skirtsOn);
      rebuildWorld(currentMode);
    });

    btnReplaceTile.addEventListener('click', () => {
      if (!tiles) return;
      if (!selectedTileId) {
        updateHudStats('Click a tile on the terrain first to select it!');
        return;
      }
      const targetId = selectedTileId;
      const rec = tiles.getTile(targetId);
      if (!rec) return;
      tiles.setTile({
        id: targetId,
        bounds: rec.bounds,
        color: '#f59e0b',
        roughness: 0.35,
        metalness: 0.35,
        heightSampler: (lon, lat) => terrainHeight(lon, lat) + 0.14,
      });
      updateHudStats(`Replaced tile "${targetId}" (old GPU mesh unloaded).`);
    });

    btnRemoveTile.addEventListener('click', () => {
      if (!tiles) return;
      if (!selectedTileId) {
        updateHudStats('Click a tile on the terrain first to select it!');
        return;
      }
      const targetId = selectedTileId;
      tiles.removeTile(targetId);
      selectedTileId = null;
      updateHudStats(`Removed tile "${targetId}" and destroyed its GPU buffers.`);
    });

    btnAsyncTile.addEventListener('click', async () => {
      if (!tiles) return;
      const id = currentMode === 'spherical' ? 'sphere-2-1' : 'flat-1-1';
      const existing = tiles.getTile(id);
      const bounds = existing?.bounds ?? (
        currentMode === 'spherical'
          ? { minLon: -60, maxLon: 0, minLat: -25, maxLat: 25 }
          : { minLon: -5, maxLon: 0, minLat: -5, maxLat: 0 }
      );
      await tiles.loadTile(id, async () => {
        await new Promise((r) => setTimeout(r, 150));
        return {
          bounds,
          color: '#38bdf8',
          heightSampler: (lon, lat) => terrainHeight(lon, lat) + 0.08,
        };
      });
      selectedTileId = id;
      updateHudStats(`Async streamed tile "${id}" via loadTile().`);
    });

    btnBrushMode.addEventListener('click', () => {
      const order = ['inspect', 'raise', 'crater'];
      brushMode = order[(order.indexOf(brushMode) + 1) % order.length];
      btnBrushMode.textContent =
        brushMode === 'inspect'
          ? 'Click: Inspect'
          : brushMode === 'raise'
            ? 'Click: Raise Hill (+)'
            : 'Click: Sculpt Crater (-)';
      btnBrushMode.classList.toggle('active', brushMode !== 'inspect');
    });

    app.gpu.canvas.addEventListener('click', (e) => {
      if (!tiles) return;
      const rect = app.gpu.canvas.getBoundingClientRect();
      const hit = tiles.pick(e.clientX - rect.left, e.clientY - rect.top);
      if (!hit) return;

      if (brushMode !== 'inspect') {
        const pt = hit.point;
        let lon = 0;
        let lat = 0;
        if (currentMode === 'spherical') {
          const r = Math.hypot(pt[0], pt[1], pt[2]) || 1;
          lat = (Math.asin(Math.max(-1, Math.min(1, pt[1] / r))) * 180) / Math.PI;
          lon = (Math.atan2(-pt[2], pt[0]) * 180) / Math.PI;
        } else {
          lon = pt[0] / 0.55;
          lat = -pt[2] / 0.55;
        }
        const radiusDeg = currentMode === 'spherical' ? 22 : 2.8;
        const delta = brushMode === 'raise' ? 0.16 : -0.16;
        const res = tiles.deformRegion({
          lon,
          lat,
          radiusDeg,
          deltaElevation: delta,
          materialMaskValue: brushMode === 'crater' ? 1.0 : 0.4,
        });
        selectedTileId = hit.id;
        updateHudStats(
          `${brushMode === 'raise' ? 'Raised' : 'Cratered'} @ (${lon.toFixed(1)}°, ${lat.toFixed(1)}°)\n` +
            `Patched ${res.patchedVertices} verts across [${res.updatedTiles.join(', ')}]`,
        );
        return;
      }

      if (selectedTileId && selectedTileId !== hit.id) {
        const prev = tiles.getTile(selectedTileId);
        if (prev) prev.entity.color = [1, 1, 1, 1];
      }
      selectedTileId = hit.id;
      const cur = tiles.getTile(selectedTileId);
      if (cur) cur.entity.color = [0.45, 0.9, 1.0, 1];
      updateHudStats(`Selected "${selectedTileId}" @ ${hit.distance.toFixed(2)}m`);
    });
  } catch (err) {
    output.textContent = err instanceof Error ? err.message : String(err);
  }
}

void init();
