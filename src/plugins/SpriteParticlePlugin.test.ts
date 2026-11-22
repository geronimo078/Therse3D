import { describe, expect, it, vi } from 'vitest';
import {
  PARTICLE_PRESETS,
  SpriteParticleManager,
  SpriteParticlePlugin,
  generateParticleTexturePixels,
  type ParticleShape,
} from './SpriteParticlePlugin';
import type { Engine, TickCallback } from '../core/Engine';
import { WorldStore } from '../core/WorldStore';
import type { PrimitiveDescriptor } from '../entities/Primitives';

function createMockEngine() {
  const store = new WorldStore(512);
  let nextTextureId = 1;
  const tickCallbacks = new Set<TickCallback>();
  const cleanups: Array<() => void> = [];
  const removedEntities: unknown[] = [];

  const engine = {
    store,
    registerTexture: vi.fn((_source: unknown, _w?: number, _h?: number) => nextTextureId++),
    add: vi.fn((desc: PrimitiveDescriptor) => {
      const id = store.allocate();
      if (desc.position) {
        store.positions[id * 3] = desc.position[0] ?? 0;
        store.positions[id * 3 + 1] = desc.position[1] ?? 0;
        store.positions[id * 3 + 2] = desc.position[2] ?? 0;
      }
      store.visible[id] = desc.visible === false ? 0 : 1;
      return {
        id,
        name: desc.id ?? `sprite-${id}`,
        visible: desc.visible !== false,
        color: [1, 1, 1, 1],
      };
    }),
    remove: vi.fn((entity: { id?: number }) => {
      removedEntities.push(entity);
      if (typeof entity?.id === 'number') {
        store.free(entity.id);
      }
      return true;
    }),
    onTick: vi.fn((cb: TickCallback) => {
      tickCallbacks.add(cb);
      return () => tickCallbacks.delete(cb);
    }),
    use: vi.fn((plugin: (e: Engine) => void | (() => void)) => {
      const cleanup = plugin(engine as unknown as Engine);
      if (typeof cleanup === 'function') cleanups.push(cleanup);
      return engine;
    }),
  };

  return {
    engine: engine as unknown as Engine,
    mocks: {
      engine,
      store,
      tickCallbacks,
      cleanups,
      removedEntities,
    },
  };
}

describe('SpriteParticlePlugin', () => {
  it('generates valid RGBA procedural textures for all particle shapes', () => {
    const shapes: ParticleShape[] = ['soft-circle', 'spark', 'ring', 'streak', 'snowflake'];
    for (const shape of shapes) {
      const pixels = generateParticleTexturePixels(shape, 32);
      expect(pixels.length).toBe(32 * 32 * 4);
      // Center or ring should have visible non-zero alpha
      let maxAlpha = 0;
      for (let i = 3; i < pixels.length; i += 4) {
        if (pixels[i]! > maxAlpha) maxAlpha = pixels[i]!;
      }
      expect(maxAlpha).toBeGreaterThan(180);
    }
  });

  it('spawns, animates, expires, and recycles particles without growing past maxParticles', () => {
    const { engine, mocks } = createMockEngine();
    const particles = SpriteParticlePlugin(engine, {
      maxParticles: 24,
      autoUpdate: false,
      seed: 42,
    });

    expect(particles).toBeInstanceOf(SpriteParticleManager);
    const fire = particles.createEmitter({
      id: 'campfire',
      preset: 'fire',
      position: [1, 0, -2],
      rate: 40,
      lifetime: [0.3, 0.3],
    });

    // Step 0.1s -> spawns 4 particles
    particles.update(0.1);
    expect(particles.getStats().activeParticles).toBe(4);
    expect(particles.getStats().pooledParticles).toBe(4);

    // Step many frames to fill pool and verify recycling
    for (let i = 0; i < 30; i++) {
      particles.update(0.05);
    }
    const stats = particles.getStats();
    expect(stats.activeParticles).toBeLessThanOrEqual(24);
    expect(stats.pooledParticles).toBeLessThanOrEqual(24);

    // Disable emitter and advance past lifetime -> all particles expire cleanly
    fire.setEnabled(false);
    for (let i = 0; i < 10; i++) {
      particles.update(0.05);
    }
    expect(particles.getStats().activeParticles).toBe(0);
    // Pooled entities stay allocated for reuse until dispose()
    expect(particles.getStats().pooledParticles).toBeGreaterThan(0);

    // Cleanup via engine.use() teardown removes all sprite entities
    for (const fn of mocks.cleanups) fn();
    expect(particles.getStats().pooledParticles).toBe(0);
    expect(mocks.removedEntities.length).toBeGreaterThan(0);
  });

  it('supports all built-in presets (fire, smoke, magic, rain, snow, sparks, explosion) and bursts', () => {
    const { engine } = createMockEngine();
    const particles = SpriteParticlePlugin(engine, {
      maxParticles: 160,
      autoUpdate: false,
      seed: 123,
    });

    const presetNames = Object.keys(PARTICLE_PRESETS) as Array<keyof typeof PARTICLE_PRESETS>;
    for (const preset of presetNames) {
      particles.createEmitter({ id: `em-${preset}`, preset, position: [0, 2, 0] });
    }
    expect(particles.listEmitters().length).toBe(presetNames.length);

    // Explosion has initialBurst, continuous presets spawn on update
    const beforeStep = particles.getStats().activeParticles;
    expect(beforeStep).toBeGreaterThan(0);

    particles.update(0.08);
    expect(particles.getStats().activeParticles).toBeGreaterThan(beforeStep);

    // Ad-hoc one-shot burst
    const spawned = particles.burst({ preset: 'sparks', position: [2, 1, 0] }, 10);
    expect(spawned).toBeGreaterThan(0);
    expect(spawned).toBeLessThanOrEqual(10);

    particles.dispose();
  });

  it('moves particles according to gravity, swirl, and emitter updates', () => {
    const { engine, mocks } = createMockEngine();
    const particles = SpriteParticlePlugin(engine, {
      maxParticles: 10,
      autoUpdate: false,
      seed: 7,
    });

    const magic = particles.createEmitter({
      id: 'spell',
      preset: 'custom',
      position: [0, 1, 0],
      rate: 0,
      spawnArea: 'point',
      velocity: [0, 2, 0],
      velocitySpread: [0, 0, 0],
      gravity: [0, -10, 0],
      lifetime: 2.0,
    });

    expect(magic.burst(1)).toBe(1);
    const startY = mocks.store.positions[1]!;
    expect(startY).toBeCloseTo(1.0, 4);

    // After 0.1s: vy = 2 - 1 = 1, y = 1 + 1 * 0.1 = 1.1
    particles.update(0.1);
    expect(mocks.store.positions[1]!).toBeCloseTo(1.1, 4);

    // Emitter position update
    magic.setPosition(3, 4, 5);
    expect(magic.getPosition()).toEqual([3, 4, 5]);

    // Removing emitter immediately releases its active particles
    expect(magic.remove()).toBe(true);
    expect(particles.getStats().activeParticles).toBe(0);

    particles.dispose();
  });

  it('auto-heals invalid inputs and supports engine.use(SpriteParticlePlugin()) form', () => {
    const { engine, mocks } = createMockEngine();
    engine.use(SpriteParticlePlugin({ maxParticles: Number.NaN, textureSize: -50 }));
    expect(mocks.tickCallbacks.size).toBe(1);
    expect(mocks.cleanups.length).toBe(1);

    const mgr = SpriteParticlePlugin(engine, { maxParticles: 16, autoUpdate: false });
    const h = mgr.createEmitter({
      preset: 'unknown-preset' as never,
      position: [Number.NaN, Number.POSITIVE_INFINITY, 1],
      rate: -99,
      lifetime: -5,
    });
    expect(h.getPosition()).toEqual([0, 0, 1]);
    expect(() => mgr.update(Number.NaN)).not.toThrow();
    expect(() => mgr.update(-1)).not.toThrow();
    mgr.dispose();
  });
});
