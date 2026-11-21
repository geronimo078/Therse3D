import { describe, expect, it } from 'vitest';
import { DEFAULT_MAX_FPS, sanitizeMaxFps } from './Engine';

describe('Engine FPS limiting', () => {
  it('defaults to 30 FPS and sanitizes invalid values safely', () => {
    expect(DEFAULT_MAX_FPS).toBe(30);
    expect(sanitizeMaxFps(undefined)).toBe(30);
    expect(sanitizeMaxFps(Number.NaN)).toBe(30);
    expect(sanitizeMaxFps(-10)).toBe(30);
    expect(sanitizeMaxFps(60)).toBe(60);
    expect(sanitizeMaxFps(120)).toBe(120);
    expect(sanitizeMaxFps(0.2)).toBe(1);
    expect(sanitizeMaxFps(0)).toBe(0);
    expect(sanitizeMaxFps(Infinity)).toBe(0);
  });
});
