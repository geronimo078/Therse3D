import type { Engine } from '../core/Engine';
import { MATERIAL_PRESETS } from '../entities/Materials';

/**
 * Complete JSON Schema for Structured Outputs / Tool Calling with therse3d.
 */
export const THERSE3D_SCENE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['entities'],
  properties: {
    preset: {
      type: 'string',
      enum: ['studio', 'outdoor', 'dark-ui', 'cad-blueprint', 'minimal'],
    },
    transition: { type: 'number', minimum: 0 },
    camera: {
      type: 'object',
      properties: {
        target: {
          oneOf: [
            { type: 'string' },
            { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
          ],
        },
        angle: {
          type: 'string',
          enum: ['front', 'front-elevated', 'top', 'isometric'],
        },
        distance: { type: 'number', minimum: 0.5 },
      },
    },
    entities: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'type'],
        properties: {
          id: { type: 'string' },
          type: {
            type: 'string',
            enum: ['box', 'sphere', 'cylinder', 'plane', 'line', 'model', 'group', 'light', 'cone', 'torus'],
          },
          parent: { type: 'string' },
          src: { type: 'string' },
          url: { type: 'string' },
          morphWeights: {
            oneOf: [
              {
                type: 'object',
                additionalProperties: { type: 'number', minimum: -2, maximum: 2 },
              },
              {
                type: 'array',
                items: { type: 'number', minimum: -2, maximum: 2 },
              },
            ],
          },
          from: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
          to: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
          kind: { type: 'string', enum: ['point', 'directional', 'spot'] },
          intensity: { type: 'number', minimum: 0 },
          range: { type: 'number', minimum: 0.1 },
          castShadow: { type: 'boolean' },
          position: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
          rotation: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 4 },
          scale: {
            oneOf: [
              { type: 'number' },
              { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
            ],
          },
          size: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
          radius: { type: 'number', minimum: 0.001 },
          color: {
            oneOf: [
              { type: 'string' },
              { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 4 },
            ],
          },
          metalness: { type: 'number', minimum: 0, maximum: 1 },
          roughness: { type: 'number', minimum: 0.04, maximum: 1 },
          emissive: { type: 'number', minimum: 0 },
          visible: { type: 'boolean' },
          material: {
            type: 'string',
            enum: Object.keys(MATERIAL_PRESETS),
          },
          texture: {
            oneOf: [{ type: 'string' }, { type: 'integer', minimum: 0 }],
          },
          on: { type: 'string' },
          beside: { type: 'string' },
          side: { type: 'string', enum: ['left', 'right', 'front', 'back'] },
          inside: { type: 'string' },
          avoidOverlap: { type: 'boolean' },
          align: {
            type: 'string',
            enum: [
              'center',
              'front',
              'back',
              'left',
              'right',
              'front-left',
              'front-right',
              'back-left',
              'back-right',
            ],
          },
          margin: { type: 'number' },
          offset: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
          lookAt: {
            oneOf: [
              { type: 'string' },
              { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
            ],
          },
        },
      },
    },
    layouts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['kind', 'entities'],
        properties: {
          kind: { type: 'string', enum: ['grid', 'ring', 'row'] },
          entities: { type: 'array', items: { type: 'string' } },
          center: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
          radius: { type: 'number', minimum: 0.2 },
          spacing: { type: 'number', minimum: 0.1 },
          columns: { type: 'number', minimum: 1 },
          lookAtCenter: { type: 'boolean' },
        },
      },
    },
  },
} as const;

export const SCENE_SYNC_JSON_SCHEMA = THERSE3D_SCENE_SCHEMA;

export const THERSE3D_TOOL_DEFINITIONS = [
  {
    name: 'therse_sync_scene',
    description: 'Synchronizes the 3D scene declaratively and returns SyncReport + spatial state.',
    parameters: THERSE3D_SCENE_SCHEMA,
  },
  {
    name: 'therse_patch_scene',
    description: 'Applies partial upsert/remove updates to existing entities without resending the full scene.',
    parameters: {
      type: 'object',
      properties: {
        preset: { type: 'string' },
        transition: { type: 'number', minimum: 0 },
        upsert: THERSE3D_SCENE_SCHEMA.properties.entities,
        remove: { type: 'array', items: { type: 'string' } },
      },
    },
  },
  {
    name: 'therse_undo',
    description: 'Reverts the scene to the previous state before the last sync or patch.',
    parameters: { type: 'object', properties: {} },
  },
  {
    name: 'therse_inspect',
    description: 'Returns spatial graph, visibility, screen regions, and collision warnings.',
    parameters: { type: 'object', properties: {} },
  },
  {
    name: 'therse_snapshot',
    description: 'Captures an annotated RGB snapshot for multimodal VLM visual verification.',
    parameters: {
      type: 'object',
      properties: {
        resolution: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      },
    },
  },
  {
    name: 'therse_camera_focus',
    description: 'Focuses the camera on a specific entityId or view preset.',
    parameters: {
      type: 'object',
      properties: {
        target: { type: 'string' },
        view: { type: 'string', enum: ['top', 'front', 'front-elevated', 'isometric'] },
        distance: { type: 'number', minimum: 0.5 },
      },
    },
  },
  {
    name: 'therse_pick',
    description: 'Picks the 3D entity at normalized screen coordinates (nx, ny in [0..1]).',
    parameters: {
      type: 'object',
      required: ['x', 'y'],
      properties: {
        x: { type: 'number', minimum: 0, maximum: 1 },
        y: { type: 'number', minimum: 0, maximum: 1 },
      },
    },
  },
] as const;

export class Schema {
  static getPromptContext(app?: Engine): string {
    const materials = Object.keys(MATERIAL_PRESETS).join(', ');
    const lines = [
      'therse3d AI-Native WebGPU 3D Engine',
      'Primitives: box, sphere, cylinder, plane, cone, torus, line, model, group, light',
      `Materials: ${materials}`,
      'Hierarchy & Layout: parent, on (entityId|"floor"), beside (entityId + side: left|right|front|back), inside (entityId), avoidOverlap, align, margin, offset [dx,dy,dz], lookAt, layouts (grid|ring|row)',
      'Expressions & Morphs: morphWeights ({ [blendshapeName]: 0..1 } or number[]) on glTF/model entities with morphTargets, smoothly animated via transition',
      'Presets: studio, outdoor, dark-ui, cad-blueprint, minimal',
      'Tools: therse_sync_scene, therse_patch_scene, therse_undo, therse_inspect, therse_snapshot, therse_camera_focus, therse_pick',
    ];

    if (app) {
      const desc = app.describe();
      const ids = desc.entities.map((e) => e.id).join(', ') || '(empty)';
      lines.push(`Active entities (${desc.entityCount}): ${ids}`);
    }

    return lines.join('\n');
  }
}
