# Sprite Particle Plugin Demo

Interactive WebGPU showcase for the optional **`SpriteParticlePlugin`** (`src/plugins/SpriteParticlePlugin.ts`).

## Included Effects
- **Fire + Smoke** (`preset: 'fire'`, `preset: 'smoke'`) — Rising flames with turbulence and expanding smoke plumes.
- **Magic Spell** (`preset: 'magic'`) — Swirling arcane sparkles orbiting around a glowing ring.
- **Sparks** (`preset: 'sparks'`) — Gravity-driven glowing spark fountain.
- **Rain** (`preset: 'rain'`) — Elongated falling rain streaks across a 3D box volume.
- **Snow** (`preset: 'snow'`) — Gently fluttering snowflakes with procedural 6-point crystal texture.
- **Explosion Burst** (`particles.burst({ preset: 'explosion', ... })`) — On-demand radial particle burst.

## Run Locally

```bash
npm run dev
```

Open `http://localhost:5173/examples/sprite-particles/` in a WebGPU-enabled browser.
