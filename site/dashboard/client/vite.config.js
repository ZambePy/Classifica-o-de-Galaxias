import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  plugins: [react()],
  resolve: {
    alias: { '@shared': fileURLToPath(new URL('../shared', import.meta.url)) },
  },
  server: {
    port: 5173,
    // Em desenvolvimento, as chamadas /api vão para o servidor Node (porta 3001).
    proxy: { '/api': 'http://localhost:3001' },
  },
  // three.js (≈740 kB) só carrega quando alguém abre um modelo 3D
  build: { outDir: 'dist', emptyOutDir: true, chunkSizeWarningLimit: 800 },
});
