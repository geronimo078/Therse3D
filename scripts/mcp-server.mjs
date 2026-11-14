#!/usr/bin/env node
/**
 * Zero-dependency MCP (Model Context Protocol) stdio <-> WebSocket bridge for therse3d.
 * Usage:
 *   node scripts/mcp-server.mjs [--port 8765]
 * Browser side:
 *   const bridge = new McpBridge(app);
 *   bridge.connect({ url: 'ws://localhost:8765' });
 */
import http from 'node:http';
import crypto from 'node:crypto';
import readline from 'node:readline';

const portArgIdx = process.argv.indexOf('--port');
const PORT = portArgIdx !== -1 ? Number(process.argv[portArgIdx + 1]) || 8765 : 8765;

const TOOLS = [
  {
    name: 'therse_sync_scene',
    description: 'Synchronizes the 3D scene declaratively and returns SyncReport + spatial state.',
    inputSchema: { type: 'object', required: ['entities'], properties: { preset: { type: 'string' }, transition: { type: 'number' }, camera: { type: 'object' }, entities: { type: 'array' }, layouts: { type: 'array' } } },
  },
  {
    name: 'therse_patch_scene',
    description: 'Applies partial upsert/remove updates to existing entities.',
    inputSchema: { type: 'object', properties: { preset: { type: 'string' }, transition: { type: 'number' }, upsert: { type: 'array' }, remove: { type: 'array' } } },
  },
  {
    name: 'therse_undo',
    description: 'Reverts the scene to the previous state.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'therse_inspect',
    description: 'Returns spatial graph, visibility, screen regions, and collision warnings.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'therse_snapshot',
    description: 'Captures an annotated RGB snapshot for VLM visual verification.',
    inputSchema: { type: 'object', properties: { resolution: { type: 'array' } } },
  },
  {
    name: 'therse_camera_focus',
    description: 'Focuses the camera on an entityId or view preset.',
    inputSchema: { type: 'object', properties: { target: { type: 'string' }, view: { type: 'string' }, distance: { type: 'number' } } },
  },
  {
    name: 'therse_pick',
    description: 'Picks the 3D entity at normalized screen coordinates (x, y in [0..1]).',
    inputSchema: { type: 'object', required: ['x', 'y'], properties: { x: { type: 'number' }, y: { type: 'number' } } },
  },
];

let activeSocket = null;
let wsBuffer = Buffer.alloc(0);
let nextReqId = 1;
const pending = new Map();

function sendWsText(socket, text) {
  const payload = Buffer.from(text, 'utf8');
  const len = payload.length;
  let header;
  if (len < 126) {
    header = Buffer.from([0x81, len]);
  } else if (len < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x81;
    header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x81;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  socket.write(Buffer.concat([header, payload]));
}

function processWsFrames() {
  while (wsBuffer.length >= 2) {
    const b0 = wsBuffer[0];
    const b1 = wsBuffer[1];
    const opcode = b0 & 0x0f;
    const masked = (b1 & 0x80) !== 0;
    let payloadLen = b1 & 0x7f;
    let offset = 2;

    if (payloadLen === 126) {
      if (wsBuffer.length < 4) return;
      payloadLen = wsBuffer.readUInt16BE(2);
      offset = 4;
    } else if (payloadLen === 127) {
      if (wsBuffer.length < 10) return;
      payloadLen = Number(wsBuffer.readBigUInt64BE(2));
      offset = 10;
    }

    const maskLen = masked ? 4 : 0;
    if (wsBuffer.length < offset + maskLen + payloadLen) return;

    let payload = wsBuffer.subarray(offset + maskLen, offset + maskLen + payloadLen);
    if (masked) {
      const mask = wsBuffer.subarray(offset, offset + 4);
      payload = Buffer.from(payload);
      for (let i = 0; i < payload.length; i++) {
        payload[i] ^= mask[i % 4];
      }
    }
    wsBuffer = wsBuffer.subarray(offset + maskLen + payloadLen);

    if (opcode === 0x8) {
      activeSocket?.end();
      activeSocket = null;
      return;
    }
    if (opcode === 0x1) {
      try {
        const msg = JSON.parse(payload.toString('utf8'));
        const resolver = pending.get(msg.id);
        if (resolver) {
          pending.delete(msg.id);
          resolver(msg.result);
        }
      } catch {
        // Ignore malformed messages
      }
    }
  }
}

const server = http.createServer((_req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('therse3d MCP WebSocket bridge running\n');
});

server.on('upgrade', (req, socket) => {
  const key = req.headers['sec-websocket-key'];
  if (!key) {
    socket.destroy();
    return;
  }
  const accept = crypto
    .createHash('sha1')
    .update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
    .digest('base64');

  socket.write(
    [
      'HTTP/1.1 101 Switching Protocols',
      'Upgrade: websocket',
      'Connection: Upgrade',
      `Sec-WebSocket-Accept: ${accept}`,
      '\r\n',
    ].join('\r\n'),
  );

  activeSocket = socket;
  wsBuffer = Buffer.alloc(0);
  socket.on('data', (chunk) => {
    wsBuffer = Buffer.concat([wsBuffer, chunk]);
    processWsFrames();
  });
  socket.on('close', () => {
    if (activeSocket === socket) activeSocket = null;
  });
});

server.listen(PORT, '127.0.0.1');

function sendRpc(id, result, error) {
  const msg = error ? { jsonrpc: '2.0', id, error } : { jsonrpc: '2.0', id, result };
  process.stdout.write(`${JSON.stringify(msg)}\n`);
}

function callBrowserTool(tool, args) {
  if (!activeSocket || activeSocket.destroyed) {
    return Promise.reject(new Error(`No therse3d browser tab connected on ws://127.0.0.1:${PORT}`));
  }
  const id = nextReqId++;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error(`Timed out waiting for browser response to "${tool}"`));
    }, 10_000);
    pending.set(id, (res) => {
      clearTimeout(timer);
      resolve(res);
    });
    sendWsText(activeSocket, JSON.stringify({ id, tool, args }));
  });
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on('line', async (line) => {
  if (!line.trim()) return;
  let req;
  try {
    req = JSON.parse(line);
  } catch {
    return;
  }

  const { id, method, params } = req;
  if (method === 'initialize') {
    sendRpc(id, {
      protocolVersion: '2024-11-05',
      capabilities: { tools: {} },
      serverInfo: { name: 'therse3d-mcp', version: '0.1.0' },
    });
  } else if (method === 'tools/list') {
    sendRpc(id, { tools: TOOLS });
  } else if (method === 'tools/call') {
    try {
      const result = await callBrowserTool(params?.name, params?.arguments ?? {});
      sendRpc(id, { content: [{ type: 'text', text: JSON.stringify(result) }] });
    } catch (err) {
      sendRpc(id, {
        isError: true,
        content: [{ type: 'text', text: err instanceof Error ? err.message : String(err) }],
      });
    }
  } else if (id !== undefined) {
    sendRpc(id, {});
  }
});

