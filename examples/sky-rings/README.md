# Sky Rings Flight Game (`examples/sky-rings`)

An arcade WebGPU flight game built with **therse3d** where you pilot an airplane through a 12-ring aerial course above an archipelago, featuring volumetric clouds, god rays, and smooth chase-camera physics.

## Files

- `index.html` — Fullscreen WebGPU canvas, score/ring HUD, pause/reset buttons, and flight controls reference.
- `game.js` — Procedural airplane assembly, 12-ring course, keyboard + mouse flight controls, segment-to-sphere ring collision detection, and chase camera.

## Key Engine Features Demonstrated

1. **Imperative Scene & Entity Composition**
   - Uses `Engine.create('#canvas', { preset: 'outdoor' })` with `Light`, `Plane`, `Cylinder`, `Box`, `Sphere`, `Cone`, `Torus`, and `Group` primitives.
   - Builds a multi-part airplane (fuselage, wings, tail, nose cone, and spinning propeller) using hierarchical parenting.
2. **Volumetric Atmosphere & God Rays**
   - Configures `app.setVolumetric(...)` with planar volumetric clouds, atmospheric fog, sun scattering, and directional god rays from the sun light source.
3. **Zero-Allocation Flight & Collision Loop (`app.onTick`)**
   - Updates aircraft yaw, pitch, roll, spinning propeller, and spring-damped chase camera every frame.
   - Uses continuous segment-to-sphere collision detection (`[prevPos, currPos]` against each uncleared ring's center) so fast passes through any ring reliably award `+1` point and advance the course/lap.

## Controls

- **Mouse Steering:** Move the cursor over the canvas to steer pitch and yaw; hold **Left Mouse Button** to boost.
- **Keyboard Steering:** `W` / `S` or `↑` / `↓` for pitch, `A` / `D` or `←` / `→` for yaw/bank, `Shift` or `Space` to boost.
- **HUD:** Use **Pause / Resume** and **Reset** buttons to control the run.

## Running

Start the Vite dev server from the repository root (`npm run dev`) and open `/examples/sky-rings/` in a WebGPU-capable browser.

