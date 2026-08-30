// Bun's dev server treats an .html import as a bundled route (see dev.ts).
declare module "*.html" {
  const html: import("bun").HTMLBundle;
  export default html;
}
