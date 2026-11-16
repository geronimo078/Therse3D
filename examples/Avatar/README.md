# Local AI Avatar Studio (`examples/Avatar`)

An interactive **therse3d** demo that connects your local LLM or Vision-Language Model (Ollama, LM Studio, llama.cpp, or any OpenAI-compatible endpoint) directly to a real-time 3D avatar with spatial and visual feedback.

## Quick Start

1. Start the dev server from the repository root:
   ```bash
   npm run dev
   ```
2. Open:
   ```text
   http://localhost:5173/examples/Avatar/index.html
   ```

## Connecting Your Local AI

1. Select a **Local AI Provider** (`Ollama (11434)`, `LM Studio (1234)`, or `Custom OpenAI API`).
2. Enter your **Model Name** (e.g. `qwen2.5:7b`, `qwen2.5-vl:7b`, `gemma3:4b`, `llama3.2`) and optional API key.
3. *(Optional)* Enable **Send VLM Snapshot** if your model supports vision inputs — the demo attaches an annotated `app.snapshot()` image alongside `app.describe()`.
4. Type an instruction (or mood) and click **Ask Local AI**.

> **CORS Note for Ollama:** If the browser blocks requests to `http://localhost:11434`, start Ollama with `OLLAMA_ORIGINS=*` enabled. In LM Studio, enable **CORS** in the Local Server settings.

## Features Demonstrated

- **Closed-Loop AI Control:** Combines `Engine.getPromptContext(app)`, `app.describe()`, and optional `app.snapshot({ layers: ['annotated'] })` so your local AI can see and adjust its 3D avatar via `app.patch()` / `app.sync()` with smoothstep `transition` tweening.
- **Instant Modular Avatar + Quick Expressions:** Works out of the box with a stylized avatar, `SpotLight` studio lighting, and a camera-facing `Text` / `Sprite` speech bubble — including offline expression presets (`Neutral`, `Happy`, `Curious`, `Thinking`).
- **Custom `.glb` Avatar Upload:** Upload any binary glTF 2.0 (`.glb`) head or character model. Embedded `baseColorTexture`, `morphTargets` (blendshapes), and `bones` (skeletal skinning) are automatically detected and exposed to the AI in `app.describe()`.
