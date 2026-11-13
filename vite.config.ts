import { defineConfig } from 'vite';
import { resolve } from 'node:path';

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, 'src/index.ts'),
      name: 'therse3d',
      fileName: 'therse3d',
      formats: ['es', 'cjs'],
    },
    sourcemap: true,
    target: 'es2022',
  },
});

