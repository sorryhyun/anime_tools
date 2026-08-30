import { defineConfig } from "vite";
import solid from "vite-plugin-solid";
import { viteSingleFile } from "vite-plugin-singlefile";

// Emits ONE self-contained anime_tools/gui/static/index.html: the FastAPI
// server serves only "/", the wheel's package-data is `static/*`, and git
// installs (`uv tool install … git+…`) ship the committed file as-is.
export default defineConfig({
  plugins: [solid(), viteSingleFile()],
  build: {
    outDir: "../anime_tools/gui/static",
    emptyOutDir: false,
    target: "es2022",
    minify: true,
    reportCompressedSize: false,
  },
  server: {
    // `bun run dev` proxies the API to a running `make gui`.
    proxy: { "/api": { target: "http://127.0.0.1:8790", changeOrigin: false } },
  },
});
