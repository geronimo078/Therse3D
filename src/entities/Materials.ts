/**
 * Semantische Material-Presets & fehlertoleranter Farbparser für Menschen und LLMs.
 */
export interface MaterialDefinition {
  color: [number, number, number, number];
  metalness: number;
  roughness: number;
  emissive: number;
  materialId: number;
}

export type MaterialPresetName =
  | 'oak-wood'
  | 'brushed-aluminum'
  | 'matte-plastic'
  | 'frosted-glass'
  | 'emissive-neon'
  | 'matte-ceramic'
  | 'polished-gold'
  | 'studio-grid'
  | 'carbon-fiber';

export const MATERIAL_PRESETS: Record<MaterialPresetName, MaterialDefinition> = {
  'oak-wood': { color: [0.55, 0.37, 0.22, 1.0], metalness: 0.0, roughness: 0.72, emissive: 0.0, materialId: 1 },
  'brushed-aluminum': { color: [0.78, 0.80, 0.82, 1.0], metalness: 0.92, roughness: 0.28, emissive: 0.0, materialId: 2 },
  'matte-plastic': { color: [0.85, 0.85, 0.88, 1.0], metalness: 0.0, roughness: 0.65, emissive: 0.0, materialId: 3 },
  'frosted-glass': { color: [0.90, 0.95, 0.98, 0.5], metalness: 0.0, roughness: 0.35, emissive: 0.0, materialId: 4 },
  'emissive-neon': { color: [0.10, 0.85, 1.00, 1.0], metalness: 0.0, roughness: 0.20, emissive: 2.0, materialId: 5 },
  'matte-ceramic': { color: [0.92, 0.91, 0.88, 1.0], metalness: 0.0, roughness: 0.85, emissive: 0.0, materialId: 6 },
  'polished-gold': { color: [0.96, 0.78, 0.34, 1.0], metalness: 1.0, roughness: 0.14, emissive: 0.0, materialId: 7 },
  'studio-grid': { color: [0.16, 0.19, 0.24, 1.0], metalness: 0.0, roughness: 0.88, emissive: 0.0, materialId: 8 },
  'carbon-fiber': { color: [0.10, 0.11, 0.13, 1.0], metalness: 0.25, roughness: 0.38, emissive: 0.0, materialId: 9 },
};

const CSS_COLORS: Record<string, [number, number, number]> = {
  white: [1, 1, 1],
  black: [0, 0, 0],
  red: [1, 0, 0],
  green: [0, 0.8, 0],
  blue: [0, 0, 1],
  yellow: [1, 1, 0],
  cyan: [0, 1, 1],
  magenta: [1, 0, 1],
  orange: [1, 0.5, 0],
  gray: [0.5, 0.5, 0.5],
  grey: [0.5, 0.5, 0.5],
};

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

export function parseColor(
  input: string | readonly number[] | undefined,
  warnings?: string[],
): [number, number, number, number] {
  if (input === undefined) {
    return [1, 1, 1, 1];
  }

  if (Array.isArray(input) || ArrayBuffer.isView(input)) {
    const arr = input as readonly number[];
    const r = Number.isFinite(arr[0]) ? Math.min(1, Math.max(0, arr[0]!)) : 1;
    const g = Number.isFinite(arr[1]) ? Math.min(1, Math.max(0, arr[1]!)) : 1;
    const b = Number.isFinite(arr[2]) ? Math.min(1, Math.max(0, arr[2]!)) : 1;
    const a = arr.length >= 4 && Number.isFinite(arr[3]) ? Math.min(1, Math.max(0, arr[3]!)) : 1;
    return [r, g, b, a];
  }

  if (typeof input === 'string') {
    const trimmed = input.trim().toLowerCase();
    const named = CSS_COLORS[trimmed];
    if (named) {
      return [srgbToLinear(named[0]), srgbToLinear(named[1]), srgbToLinear(named[2]), 1];
    }

    if (trimmed.startsWith('#')) {
      const hex = trimmed.slice(1);
      if (/^[0-9a-f]{3}$/.test(hex)) {
        const r = parseInt(hex[0]! + hex[0]!, 16) / 255;
        const g = parseInt(hex[1]! + hex[1]!, 16) / 255;
        const b = parseInt(hex[2]! + hex[2]!, 16) / 255;
        return [srgbToLinear(r), srgbToLinear(g), srgbToLinear(b), 1];
      }
      if (/^[0-9a-f]{6}$/.test(hex) || /^[0-9a-f]{8}$/.test(hex)) {
        const r = parseInt(hex.slice(0, 2), 16) / 255;
        const g = parseInt(hex.slice(2, 4), 16) / 255;
        const b = parseInt(hex.slice(4, 6), 16) / 255;
        const a = hex.length === 8 ? parseInt(hex.slice(6, 8), 16) / 255 : 1;
        return [srgbToLinear(r), srgbToLinear(g), srgbToLinear(b), a];
      }
    }
  }

  warnings?.push(`Invalid color "${String(input)}"; falling back to white.`);
  return [1, 1, 1, 1];
}

export function resolveMaterial(
  presetOrDef: string | Partial<MaterialDefinition> | undefined,
  warnings?: string[],
): MaterialDefinition {
  if (!presetOrDef) {
    return { color: [0.85, 0.85, 0.88, 1.0], metalness: 0.0, roughness: 0.5, emissive: 0.0, materialId: 0 };
  }

  if (typeof presetOrDef === 'string') {
    const preset = MATERIAL_PRESETS[presetOrDef as MaterialPresetName];
    if (preset) {
      return preset;
    }
    warnings?.push(`Unknown material preset "${presetOrDef}"; falling back to matte-plastic.`);
    return MATERIAL_PRESETS['matte-plastic'];
  }

  return {
    color: presetOrDef.color ?? [0.85, 0.85, 0.88, 1.0],
    metalness: Number.isFinite(presetOrDef.metalness) ? Math.min(1, Math.max(0, presetOrDef.metalness!)) : 0.0,
    roughness: Number.isFinite(presetOrDef.roughness) ? Math.min(1, Math.max(0.04, presetOrDef.roughness!)) : 0.5,
    emissive: Number.isFinite(presetOrDef.emissive) ? Math.max(0, presetOrDef.emissive!) : 0.0,
    materialId: presetOrDef.materialId ?? 0,
  };
}
