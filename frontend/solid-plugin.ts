// Bun bundler plugin: run Solid's compile-time JSX transform.
//
// Solid's reactivity is *compiled*, not runtime -- `{count()}` has to become a
// fine-grained DOM effect at build time. Bun's own JSX transform can't do that
// (it only knows the classic/automatic runtimes), so every .tsx file goes
// through babel-preset-solid first and reaches Bun's bundler as plain JS.
// This is the whole reason the build isn't just `bun build src/index.tsx`.
import type { BunPlugin } from "bun";
import { transformAsync } from "@babel/core";
// @ts-expect-error - untyped babel presets
import solid from "babel-preset-solid";
// @ts-expect-error - untyped babel presets
import typescript from "@babel/preset-typescript";

export function solidPlugin({ dev = false } = {}): BunPlugin {
  return {
    name: "solid",
    setup(build) {
      // .ts has no JSX -- Bun transpiles those itself, faster than babel.
      build.onLoad({ filter: /\.tsx$/ }, async (args) => {
        const source = await Bun.file(args.path).text();
        const out = await transformAsync(source, {
          filename: args.path,
          babelrc: false,
          configFile: false,
          sourceMaps: dev ? "inline" : false,
          // Presets run last-to-first: preset-typescript strips the types
          // (it infers TSX from the .tsx filename), then preset-solid compiles
          // the JSX. Both must stay on Babel 7 to match @babel/core.
          presets: [
            [solid, { generate: "dom", hydratable: false }],
            [typescript, { onlyRemoveTypeImports: true }],
          ],
        });
        if (!out?.code) throw new Error(`solid transform produced no output for ${args.path}`);
        return { contents: out.code, loader: "js" };
      });
    },
  };
}

export default solidPlugin();
