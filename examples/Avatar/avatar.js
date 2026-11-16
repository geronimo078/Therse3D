import { Engine, OrbitControls } from '../../src/index.ts';
import { EditablePolyMesh } from './poly-modeler.js';

const providerEl = document.getElementById('provider');
const endpointEl = document.getElementById('endpoint');
const modelEl = document.getElementById('model');
const apiKeyEl = document.getElementById('apiKey');
const glbFileEl = document.getElementById('glbFile');
const useVisionEl = document.getElementById('useVision');
const promptEl = document.getElementById('prompt');
const btnAskAi = document.getElementById('btnAskAi');
const btnAutoBuild = document.getElementById('btnAutoBuild');
const btnContinuous = document.getElementById('btnContinuous');
const btnStopAuto = document.getElementById('btnStopAuto');
const btnExportGlb = document.getElementById('btnExportGlb');
const btnDownloadLog = document.getElementById('btnDownloadLog');
const btnClearStage = document.getElementById('btnClearStage');
const btnUndo = document.getElementById('btnUndo');
const btnSnapshot = document.getElementById('btnSnapshot');
const logEl = document.getElementById('log');
const rawJsonEl = document.getElementById('rawJson');
const snapshotBox = document.getElementById('snapshotBox');
const snapshotImg = document.getElementById('snapshotImg');

function appendLog(msg) {
  logEl.textContent = `${msg}\n${logEl.textContent}`.slice(0, 3600);
}

providerEl.addEventListener('change', () => {
  if (providerEl.value === 'ollama') {
    endpointEl.value = 'http://localhost:11434/v1/chat/completions';
    if (modelEl.value === 'local-model') modelEl.value = 'qwen2.5:7b';
  } else if (providerEl.value === 'lmstudio') {
    endpointEl.value = 'http://localhost:1234/v1/chat/completions';
    modelEl.value = 'local-model';
  }
});

const STAGE_SCENE = {
  preset: 'studio',
  transition: 0.25,
  camera: { target: [0, 1.0, 0], angle: 'front-elevated', distance: 3.1 },
  entities: [
    { id: 'stage', type: 'plane', size: [10, 1, 10], position: [0, 0, 0], material: 'studio-grid' },
    { id: 'key_spot', type: 'light', kind: 'point', position: [1.8, 2.6, 2.8], color: '#fff7ed', intensity: 4.5 },
    { id: 'fill_light', type: 'light', kind: 'point', position: [-1.6, 1.5, 2.4], color: '#e0f2fe', intensity: 2.5 },
    { id: 'rim_light', type: 'light', kind: 'point', position: [-2.0, 2.4, -2.0], color: '#38bdf8', intensity: 2.6 },
  ],
};

const MODEL_STEPS = [
  // Phase 1: Core Torso & Pelvis (1-4)
  'STEP 1/24 [torso]: Create the base torso polygon cage (part: "torso", center [0, 0.96, 0], size [0.40, 0.50, 0.24]) and subdivide (iterations: 1, smooth: true).',
  'STEP 2/24 [torso]: Sculpt the waist taper and ribcage breadth on part "torso" using transform_region / sculpt_deform (mirrorX: true).',
  'STEP 3/24 [torso]: Sculpt chest/pectoral volume (+Z) and spine/back contour (-Z) on part "torso" (mirrorX: true).',
  'STEP 4/24 [pelvis+neck]: Add a sculpted pelvis/hips (part: "pelvis", create_poly_cage at [0, 0.68, 0], size [0.36, 0.22, 0.22], subdivide) and neck (part: "neck", add_poly_tube from [0,1.18,0] to [0,1.32,0], radii [0.08,0.07], mirrorX: false).',
  // Phase 2: Head & Facial Structure (5-8)
  'STEP 5/24 [head]: Create the head polygon sphere (part: "head", create_poly_sphere at [0, 1.46, 0], radius [0.18, 0.22, 0.20], segU: 14, segV: 10) and subdivide (iterations: 1, smooth: true).',
  'STEP 6/24 [head]: Sculpt jawline taper and chin forward on part "head" (transform_region maxY: 1.38 scale [0.78, 1, 0.85] + sculpt_deform at [0, 1.28, 0.16]).',
  'STEP 7/24 [head]: Sculpt brow ridge (+Z at y~1.52) and eye socket recesses (-Z at [0.07, 1.47, 0.17], mirrorX: true) on part "head".',
  'STEP 8/24 [head]: Sculpt nose bridge/tip (+Z at [0, 1.44, 0.19]), cheekbones (mirrorX: true), and subtle mouth/lip contour on part "head".',
  // Phase 3: Limbs & Boots (9-12)
  'STEP 9/24 [arms]: Model symmetric upper and lower arms (part: "arms") using add_poly_tube from shoulder [0.22, 1.14, 0] -> elbow [0.34, 0.92, 0.03] -> wrist [0.38, 0.70, 0.10], radii [0.065, 0.052, 0.04], mirrorX: true.',
  'STEP 10/24 [hands]: Add sculpted hands/forearm gauntlets (part: "hands", create_poly_sphere at [0.39, 0.65, 0.12] and [-0.39, 0.65, 0.12], radius [0.045, 0.06, 0.04]).',
  'STEP 11/24 [legs]: Model symmetric thighs and calves (part: "legs") using add_poly_tube from hip [0.11, 0.64, 0] -> knee [0.12, 0.36, 0.03] -> ankle [0.12, 0.10, 0], radii [0.085, 0.065, 0.048], mirrorX: true.',
  'STEP 12/24 [boots]: Model sculpted boots/feet (part: "boots") using create_poly_cage at [0.12, 0.06, 0.04] & [-0.12, 0.06, 0.04], size [0.11, 0.12, 0.22], subdivide.',
  // Phase 4: Eyes, Face Details, Ears & Hair/Helmet (13-16)
  'STEP 13/24 [eyes]: Create glowing eyes (part: "eyes", emissive: 2.4, create_poly_sphere at [0.068, 1.47, 0.165] & [-0.068, 1.47, 0.165], radius [0.028, 0.024, 0.025]).',
  'STEP 14/24 [visor/brows]: Add expressive brow plates or translucent visor (part: "visor", create_poly_cage at [0, 1.48, 0.17], size [0.24, 0.06, 0.08], sculpt_deform).',
  'STEP 15/24 [ears]: Add sculpted ears or cyber-comms pods on head sides (part: "ears", create_poly_sphere at [0.19, 1.46, 0] & [-0.19, 1.46, 0], radius [0.035, 0.06, 0.045]).',
  'STEP 16/24 [hair/helmet]: Model hair/helmet shell over the cranium (part: "hair", create_poly_sphere at [0, 1.50, -0.02], radius [0.20, 0.21, 0.22], sculpt_deform crest/locks + subdivide).',
  // Phase 5: Armor, Clothing & Signature Silhouette (17-20)
  'STEP 17/24 [armor]: Model sculpted chest armor / collar (part: "armor", metalness: 0.7, create_poly_cage at [0, 1.02, 0.02], size [0.44, 0.32, 0.27], subdivide + sculpt_deform).',
  'STEP 18/24 [pauldrons]: Model layered shoulder pauldrons (part: "pauldrons", metalness: 0.75, create_poly_sphere at [0.25, 1.16, 0] & [-0.25, 1.16, 0], radius [0.11, 0.08, 0.11]).',
  'STEP 19/24 [core+belt]: Add glowing chest reactor/emblem (part: "core", emissive: 2.6, create_poly_sphere at [0, 1.03, 0.155], radius [0.055, 0.055, 0.025]) and waist belt (part: "belt", create_poly_cage at [0, 0.74, 0], size [0.40, 0.08, 0.25]).',
  'STEP 20/24 [signature]: Add signature silhouette elements (part: "wings" or "horns" or "cape") using add_poly_tube (mirrorX: true) or sculpted poly cages.',
  // Phase 6: Critique, Fine Sculpt & Final Polish (21-24)
  'STEP 21/24 [critique]: Inspect all parts & bounds — recenter or adjust any misaligned part using transform_region.',
  'STEP 22/24 [fine_sculpt]: Perform fine sculpt_deform detailing on "head" and "armor" (mirrorX: true).',
  'STEP 23/24 [materials]: Harmonize colors, roughness, metalness, and emissive contrast across parts.',
  'STEP 24/24 [final_polish]: Final subtle sculpt & set your character "speech" bubble!',
];

async function init() {
  try {
    const app = await Engine.create('#canvas', { preset: 'studio' });
    app.use(OrbitControls({ distance: 3.1, target: [0, 1.0, 0] }));
    app.setBloom({ enabled: true, threshold: 0.85, intensity: 0.55, radius: 1.4 });

    /** @type {Map<string, { mesh: EditablePolyMesh, color: string, roughness: number, metalness: number, emissive: number }>} */
    const parts = new Map();
    const history = [];
    const diagnosticTrace = [];
    let customGlbUrl = null;
    let buildStep = 0;
    let stopRequested = false;
    let lastFeedback = 'Stage is empty (0 parts, 0 verts).';

    function getOrCreatePart(name) {
      const key = (typeof name === 'string' && name.trim()) ? name.trim().toLowerCase().replace(/[^a-z0-9_-]/g, '_') : 'body';
      if (!parts.has(key)) {
        parts.set(key, {
          mesh: new EditablePolyMesh(),
          color: '#cbd5e1',
          roughness: 0.45,
          metalness: 0.2,
          emissive: 0,
        });
      }
      return { key, part: parts.get(key) };
    }

    function getTotalStats() {
      let verts = 0;
      let faces = 0;
      const summary = {};
      for (const [name, p] of parts.entries()) {
        if (p.mesh.verts.length === 0) continue;
        verts += p.mesh.verts.length;
        faces += p.mesh.faces.length;
        summary[name] = {
          verts: p.mesh.verts.length,
          faces: p.mesh.faces.length,
          bounds: p.mesh.getBounds(),
          color: p.color,
        };
      }
      return { partCount: Object.keys(summary).length, verts, faces, summary };
    }

    function snapshotPartsForUndo() {
      const snap = new Map();
      for (const [k, p] of parts.entries()) {
        snap.set(k, {
          mesh: p.mesh.clone(),
          color: p.color,
          roughness: p.roughness,
          metalness: p.metalness,
          emissive: p.emissive,
        });
      }
      return snap;
    }

    function syncPartsToEngine(speechText = null) {
      const entities = [...STAGE_SCENE.entities];
      for (const [name, p] of parts.entries()) {
        if (p.mesh.faces.length > 0 && p.mesh.verts.length >= 3) {
          entities.push({
            id: `ai_part_${name}`,
            type: 'model',
            position: [0, 0, 0],
            mesh: p.mesh.toGeometryMeshData(),
            color: p.color,
            roughness: p.roughness,
            metalness: p.metalness,
            ...(p.emissive > 0 ? { emissive: p.emissive } : {}),
          });
        }
      }
      if (speechText) {
        entities.push({
          id: 'speech_bubble',
          type: 'sprite',
          position: [0.85, 1.85, 0],
          scale: [1.15, 0.3, 1],
          text: speechText,
        });
      }
      return app.sync({
        preset: 'studio',
        transition: 0.15,
        entities,
      });
    }

    app.sync(STAGE_SCENE);
    appendLog('24-Step Multi-Part Polygon Modeler ready! Click "Auto-Model (24 Steps)" or "Model Step (+1)".');

    glbFileEl.addEventListener('change', () => {
      const file = glbFileEl.files?.[0];
      if (!file) return;
      if (customGlbUrl) URL.revokeObjectURL(customGlbUrl);
      customGlbUrl = URL.createObjectURL(file);
      const res = app.sync({
        ...STAGE_SCENE,
        camera: { target: 'glb_avatar', angle: 'front', distance: 2.5 },
        entities: [
          ...STAGE_SCENE.entities,
          { id: 'glb_avatar', type: 'model', url: customGlbUrl, position: [0, 1.1, 0] },
        ],
      });
      appendLog(`Loaded .glb "${file.name}". Warnings: ${res.warnings.length}`);
    });

    btnExportGlb.addEventListener('click', () => {
      const combined = new EditablePolyMesh();
      for (const p of parts.values()) {
        if (p.mesh.faces.length === 0) continue;
        const base = combined.verts.length;
        for (const v of p.mesh.verts) combined.verts.push([v[0], v[1], v[2]]);
        for (const f of p.mesh.faces) combined.faces.push(f.map((idx) => idx + base));
      }
      if (combined.faces.length === 0) {
        appendLog('No polygon mesh to export yet!');
        return;
      }
      const blob = combined.toGlbBlob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'ai-modeled-avatar.glb';
      a.click();
      URL.revokeObjectURL(url);
      appendLog(`Exported ai-modeled-avatar.glb (${combined.verts.length} verts, ${combined.faces.length} faces).`);
    });

    btnDownloadLog.addEventListener('click', () => {
      const payload = {
        exportedAt: new Date().toISOString(),
        userAgent: navigator.userAgent,
        sceneDescribe: app.describe(),
        stats: getTotalStats(),
        trace: diagnosticTrace,
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'avatar-diagnostic-log.json';
      a.click();
      URL.revokeObjectURL(url);
      appendLog(`Saved avatar-diagnostic-log.json (${diagnosticTrace.length} entries).`);
    });

    btnClearStage.addEventListener('click', () => {
      customGlbUrl = null;
      parts.clear();
      history.length = 0;
      buildStep = 0;
      lastFeedback = 'Stage cleared (0 parts, 0 verts).';
      app.sync(STAGE_SCENE);
      if (rawJsonEl) rawJsonEl.textContent = '';
      appendLog('All parts cleared (0 verts, 0 faces).');
    });

    btnUndo.addEventListener('click', () => {
      const prev = history.pop();
      if (!prev) {
        appendLog('Nothing to undo.');
        return;
      }
      parts.clear();
      for (const [k, v] of prev.entries()) parts.set(k, v);
      buildStep = Math.max(0, buildStep - 1);
      syncPartsToEngine();
      const st = getTotalStats();
      appendLog(`Undo -> ${st.partCount} parts, ${st.verts} verts, ${st.faces} faces.`);
    });

    btnSnapshot.addEventListener('click', async () => {
      const snap = await app.snapshot({ layers: ['annotated'], resolution: [512, 384] });
      if (snap.annotatedDataUrl) {
        snapshotImg.src = snap.annotatedDataUrl;
        snapshotBox.style.display = 'block';
      }
    });

    btnStopAuto.addEventListener('click', () => {
      stopRequested = true;
      btnStopAuto.disabled = true;
      appendLog('Stopping Auto-Model after current step...');
    });

    document.querySelectorAll('.expr-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const rotMap = {
          neutral: [0, 0, 0],
          happy: [-0.06, 0.14, 0.05],
          curious: [0.05, -0.24, -0.12],
          thinking: [-0.14, 0.22, 0.08],
        };
        const targetId = customGlbUrl ? 'glb_avatar' : app.get('ai_part_head') ? 'ai_part_head' : null;
        if (!targetId) {
          appendLog('Model a head part first!');
          return;
        }
        app.patch({
          transition: 0.35,
          upsert: [{ id: targetId, rotation: rotMap[btn.dataset.expr] ?? [0, 0, 0] }],
        });
      });
    });

    function extractJsonObject(text) {
      const noThink = String(text ?? '').replace(/<think>[\s\S]*?<\/think>/gi, '');
      const fenced = noThink.match(/```(?:json)?\s*([\s\S]*?)```/i);
      const candidate = (fenced ? fenced[1] : noThink).trim();

      const objStart = candidate.indexOf('{');
      const arrStart = candidate.indexOf('[');
      let slice = '';

      if (arrStart >= 0 && (objStart < 0 || arrStart < objStart)) {
        const arrEnd = candidate.lastIndexOf(']');
        slice = arrEnd > arrStart ? candidate.slice(arrStart, arrEnd + 1) : candidate.slice(arrStart);
        const cleanedArr = slice.replace(/\/\/[^\n\r]*/g, '').replace(/,\s*([}\]])/g, '$1');
        try {
          const arr = JSON.parse(cleanedArr);
          if (Array.isArray(arr)) return { ops: arr };
        } catch { /* fall through */ }
      }

      if (objStart < 0) {
        throw new Error('Model response did not contain a valid JSON object.');
      }

      const objEnd = candidate.lastIndexOf('}');
      slice = objEnd > objStart ? candidate.slice(objStart, objEnd + 1) : candidate.slice(objStart);
      let cleaned = slice.replace(/\/\/[^\n\r]*/g, '').replace(/,\s*([}\]])/g, '$1');

      try {
        return JSON.parse(cleaned);
      } catch {
        cleaned = cleaned.replace(/,\s*$/, '');
        const stack = [];
        let inStr = false;
        let esc = false;
        for (let i = 0; i < cleaned.length; i++) {
          const ch = cleaned[i];
          if (esc) { esc = false; continue; }
          if (ch === '\\') { esc = true; continue; }
          if (ch === '"') { inStr = !inStr; continue; }
          if (inStr) continue;
          if (ch === '{') stack.push('}');
          else if (ch === '[') stack.push(']');
          else if ((ch === '}' || ch === ']') && stack.length > 0) stack.pop();
        }
        if (inStr) cleaned += '"';
        cleaned = cleaned.replace(/,\s*$/, '') + stack.reverse().join('');
        return JSON.parse(cleaned);
      }
    }

    async function runModelingStep(stepGoalOverride = null) {
      const endpoint = endpointEl.value.trim();
      const model = modelEl.value.trim();
      const userInstruction = promptEl.value.trim();
      if (!endpoint || !model || !userInstruction) {
        appendLog('Please fill in Endpoint URL, Model Name, and Modeling Brief.');
        return false;
      }

      const statsBefore = getTotalStats();
      const stepHint = stepGoalOverride ?? MODEL_STEPS[Math.min(buildStep, MODEL_STEPS.length - 1)];

      const systemPrompt = [
        'You are a 3D Polygon Modeler (Blender Edit & Sculpt Mode) creating a multi-part humanoid/character avatar.',
        'Each body/armor/face part has its own "part" name, color, roughness, metalness, and optional emissive.',
        'Respond ONLY with a single valid JSON object (no markdown, no comments, no trailing commas):',
        '{ "part": "torso", "color": "#hex", "roughness": 0.45, "metalness": 0.2, "emissive": 0, "speech": "optional", "ops": [ ... ] }',
        '',
        'AVAILABLE OPS (each op can also override "part" if editing multiple parts):',
        '- { "op": "create_poly_cage", "part": "torso", "center": [0, 0.96, 0], "size": [0.40, 0.50, 0.24] }',
        '- { "op": "create_poly_sphere", "part": "head", "center": [0, 1.46, 0], "radius": [0.18, 0.22, 0.20], "segU": 14, "segV": 10 }',
        '- { "op": "add_poly_tube", "part": "arms", "points": [[0.22,1.14,0],[0.34,0.92,0.03],[0.38,0.70,0.10]], "radii": [0.065,0.052,0.04], "mirrorX": true }',
        '- { "op": "extrude_region", "part": "torso", "selector": { "minY": 1.15 }, "offset": [0, 0.12, 0], "scale": [0.7, 1, 0.7] }',
        '- { "op": "transform_region", "part": "head", "selector": { "maxY": 1.38 }, "delta": [0, 0, 0], "scale": [0.8, 1, 0.85] }',
        '- { "op": "subdivide", "part": "torso", "iterations": 1, "smooth": true }',
        '- { "op": "sculpt_deform", "part": "head", "center": [0.07, 1.47, 0.17], "radius": 0.08, "delta": [0, 0, -0.04], "mirrorX": true }',
        '',
        `CURRENT PARTS (${statsBefore.partCount} parts, ${statsBefore.verts} verts, ${statsBefore.faces} faces):`,
        JSON.stringify(statsBefore.summary),
        `PREVIOUS STEP FEEDBACK: ${lastFeedback}`,
        `CURRENT TASK: ${stepHint}`,
        'Coordinate space: floor Y=0, feet Y=0.0..0.12, legs Y=0.1..0.64, pelvis Y=0.6..0.78, torso Y=0.72..1.2, head Y=1.26..1.68, facing +Z, centered at X=0.',
      ].join('\n');

      const userContent = [{ type: 'text', text: `${userInstruction}\n\n${stepHint}` }];
      if (useVisionEl.checked) {
        const snap = await app.snapshot({ layers: ['annotated'], resolution: [512, 384] });
        if (snap.annotatedDataUrl) {
          snapshotImg.src = snap.annotatedDataUrl;
          snapshotBox.style.display = 'block';
          userContent.push({ type: 'image_url', image_url: { url: snap.annotatedDataUrl } });
        }
      }

      const headers = { 'Content-Type': 'application/json' };
      if (apiKeyEl.value.trim()) headers['Authorization'] = `Bearer ${apiKeyEl.value.trim()}`;

      const reqBody = {
        model,
        temperature: 0.5,
        max_tokens: 8192,
        reasoning_effort: 'low',
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userContent.length === 1 ? userContent[0].text : userContent },
        ],
      };

      const traceEntry = {
        time: new Date().toISOString(),
        stepIndex: buildStep + 1,
        endpoint,
        model,
        stepHint,
        statsBefore,
        httpStatus: null,
        rawReply: null,
        parsedJson: null,
        parseError: null,
        statsAfter: null,
        syncReport: null,
      };
      diagnosticTrace.push(traceEntry);

      let resp = await fetch(endpoint, { method: 'POST', headers, body: JSON.stringify(reqBody) });
      if (!resp.ok && resp.status === 400) {
        delete reqBody.response_format;
        delete reqBody.reasoning_effort;
        resp = await fetch(endpoint, { method: 'POST', headers, body: JSON.stringify(reqBody) });
      }
      traceEntry.httpStatus = resp.status;
      if (!resp.ok) {
        const errTxt = await resp.text();
        traceEntry.rawReply = errTxt;
        throw new Error(`HTTP ${resp.status}: ${errTxt}`);
      }

      const data = await resp.json();
      const msgObj = data?.choices?.[0]?.message;
      const rawReply = (msgObj?.content && msgObj.content.trim())
        ? msgObj.content
        : (msgObj?.reasoning || msgObj?.reasoning_content || '');
      traceEntry.rawReply = rawReply;
      if (rawJsonEl) rawJsonEl.textContent = rawReply;

      let parsed;
      try {
        parsed = extractJsonObject(rawReply);
        traceEntry.parsedJson = parsed;
      } catch (parseErr) {
        traceEntry.parseError = parseErr instanceof Error ? parseErr.message : String(parseErr);
        const details = rawJsonEl?.closest('details');
        if (details) details.open = true;
        throw parseErr;
      }
      if (rawJsonEl) rawJsonEl.textContent = JSON.stringify(parsed, null, 2);

      history.push(snapshotPartsForUndo());

      const defaultPartName = parsed.part || 'body';
      const { part: defaultPart } = getOrCreatePart(defaultPartName);
      if (typeof parsed.color === 'string') defaultPart.color = parsed.color;
      if (typeof parsed.roughness === 'number') defaultPart.roughness = parsed.roughness;
      if (typeof parsed.metalness === 'number') defaultPart.metalness = parsed.metalness;
      if (typeof parsed.emissive === 'number') defaultPart.emissive = parsed.emissive;

      const feedbackList = [];
      if (Array.isArray(parsed.ops)) {
        for (const op of parsed.ops) {
          if (!op || typeof op !== 'object') continue;
          const targetName = op.part || defaultPartName;
          const { key, part } = getOrCreatePart(targetName);
          if (typeof op.color === 'string') part.color = op.color;
          if (typeof op.roughness === 'number') part.roughness = op.roughness;
          if (typeof op.metalness === 'number') part.metalness = op.metalness;
          if (typeof op.emissive === 'number') part.emissive = op.emissive;
          const fb = part.mesh.applyOps([op]);
          if (fb.length > 0) feedbackList.push(`[${key}] ${fb.join(', ')}`);
        }
      } else if (Array.isArray(parsed.verts) && Array.isArray(parsed.faces)) {
        defaultPart.mesh.addRaw(parsed.verts, parsed.faces);
        feedbackList.push(`[${defaultPartName}] add_verts_faces`);
      } else {
        traceEntry.parseError = 'Missing "ops" or "verts"/"faces" in parsed JSON.';
        throw new Error('AI response contained no "ops" or "verts"/"faces".');
      }

      lastFeedback = feedbackList.join(' | ') || 'ops applied';
      const report = syncPartsToEngine(parsed.speech || null);
      const statsAfter = getTotalStats();
      traceEntry.statsAfter = statsAfter;
      traceEntry.syncReport = report;

      buildStep++;
      appendLog(
        `[Step ${buildStep}/${MODEL_STEPS.length}] ${statsAfter.partCount} parts, ${statsAfter.verts} verts, ${statsAfter.faces} faces\n${lastFeedback}`,
      );
      return true;
    }

    btnAskAi.addEventListener('click', async () => {
      btnAskAi.disabled = true;
      btnAutoBuild.disabled = true;
      btnContinuous.disabled = true;
      btnAskAi.textContent = 'Modeling...';
      try {
        await runModelingStep();
      } catch (err) {
        appendLog(`Error: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        btnAskAi.disabled = false;
        btnAutoBuild.disabled = false;
        btnContinuous.disabled = false;
        btnAskAi.textContent = 'Model Step (+1)';
      }
    });

    btnAutoBuild.addEventListener('click', async () => {
      btnAskAi.disabled = true;
      btnAutoBuild.disabled = true;
      btnContinuous.disabled = true;
      btnStopAuto.disabled = false;
      stopRequested = false;
      try {
        if (parts.size > 0) {
          parts.clear();
          history.length = 0;
          buildStep = 0;
          lastFeedback = 'Stage reset for fresh 24-step Auto-Model.';
          app.sync(STAGE_SCENE);
        }
        for (let i = 0; i < MODEL_STEPS.length; i++) {
          if (stopRequested) {
            appendLog(`Auto-Model stopped by user at step ${i}/${MODEL_STEPS.length}.`);
            break;
          }
          btnAutoBuild.textContent = `Step ${i + 1}/${MODEL_STEPS.length}...`;
          try {
            await runModelingStep(MODEL_STEPS[i]);
          } catch (stepErr) {
            appendLog(`Step ${i + 1} retry (${stepErr instanceof Error ? stepErr.message : stepErr})...`);
            await runModelingStep(MODEL_STEPS[i]);
          }
          await new Promise((r) => setTimeout(r, 250));
        }
        if (!stopRequested) {
          appendLog('24-Step Auto-Model complete! Use "Continuous Improve" to keep refining endlessly.');
        }
      } catch (err) {
        appendLog(`Auto-Model stopped: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        btnAskAi.disabled = false;
        btnAutoBuild.disabled = false;
        btnContinuous.disabled = false;
        btnStopAuto.disabled = true;
        btnAutoBuild.textContent = 'Auto-Model (24 Steps)';
      }
    });

    const CONTINUOUS_GOALS = [
      'CONTINUOUS IMPROVE (Anatomy & Proportions): Inspect all existing parts and refine proportions/silhouette using sculpt_deform (mirrorX: true) and transform_region. Do NOT recreate existing base parts.',
      'CONTINUOUS IMPROVE (New Creative Detail): Invent and ADD a brand new sculpted accessory or detail part (e.g., "crest", "antenna", "shoulder_fins", "visor_trim", "halo", "gauntlets", "belt_buckle", "back_thrusters") that fits cleanly onto the current bounds.',
      'CONTINUOUS IMPROVE (Surface & Material Polish): Subdivide any coarse part if faces < 1200, sculpt fine surface contours (mirrorX: true), and refine color/roughness/metalness/emissive contrast.',
      'CONTINUOUS IMPROVE (Critique & Harmony): Check part bounds and last feedback for any misalignment or missed sculpts, fix them, and update the "speech" bubble.',
    ];

    btnContinuous.addEventListener('click', async () => {
      btnAskAi.disabled = true;
      btnAutoBuild.disabled = true;
      btnContinuous.disabled = true;
      btnStopAuto.disabled = false;
      stopRequested = false;
      let cycle = 0;
      try {
        while (!stopRequested) {
          cycle++;
          btnContinuous.textContent = `Improving #${cycle}...`;
          const goal =
            buildStep < MODEL_STEPS.length
              ? MODEL_STEPS[buildStep]
              : CONTINUOUS_GOALS[(cycle - 1) % CONTINUOUS_GOALS.length];
          try {
            await runModelingStep(goal);
          } catch (stepErr) {
            appendLog(`Improve #${cycle} retry (${stepErr instanceof Error ? stepErr.message : stepErr})...`);
            if (!stopRequested) await runModelingStep(goal);
          }
          await new Promise((r) => setTimeout(r, 300));
        }
        appendLog(`Continuous Improve stopped after ${cycle} iterations.`);
      } catch (err) {
        appendLog(`Continuous Improve stopped: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        btnAskAi.disabled = false;
        btnAutoBuild.disabled = false;
        btnContinuous.disabled = false;
        btnStopAuto.disabled = true;
        btnContinuous.textContent = 'Continuous Improve';
      }
    });
  } catch (err) {
    appendLog(`Error: ${err instanceof Error ? err.message : String(err)}`);
  }
}

void init();
