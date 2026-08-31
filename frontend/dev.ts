#!/usr/bin/env bun
// Dev server with hot reload, proxying /api to a running `make gui`.
//
// Bun bundles the HTML entry itself (see bunfig.toml for the Solid plugin the
// dev server loads), so this file is just the router: everything under /api is
// forwarded to the FastAPI process, everything else is the app shell.
import index from "./index.html";

const PORT = Number(process.env.FRONTEND_PORT ?? 5173);
const API = process.env.ANIME_TOOLS_API ?? "http://127.0.0.1:8790";

// Streaming matters: the log pane is an EventSource, so the proxied body has to
// be piped through, never buffered.
async function proxy(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const target = new URL(url.pathname + url.search, API);
  try {
    const res = await fetch(target, {
      method: req.method,
      headers: req.headers,
      body: req.body,
      redirect: "manual",
      // @ts-expect-error - required by fetch for a streaming request body
      duplex: "half",
    });
    return new Response(res.body, { status: res.status, headers: res.headers });
  } catch {
    return Response.json(
      { detail: `no anime_tools GUI at ${API} -- run \`make gui\`` },
      { status: 502 },
    );
  }
}

const server = Bun.serve({
  port: PORT,
  development: { hmr: true, console: true },
  routes: {
    "/api/*": proxy,
    "/*": index,
  },
});

console.log(`frontend  → ${server.url}   (api → ${API})`);
