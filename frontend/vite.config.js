// frontend/vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  // Proxy API calls to the FastAPI backend so the browser never has
  // to deal with CORS during development. Any request the React app
  // makes to /api/* is transparently forwarded to localhost:8000.
  // This is only active during `vite dev` — production uses Nginx (Day 7).
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target:       "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },

  // Environment variable prefix: only variables starting with VITE_
  // are exposed to the browser bundle. This prevents accidentally leaking
  // server-side secrets that happen to be in the same .env file.
  envPrefix: "VITE_",
});