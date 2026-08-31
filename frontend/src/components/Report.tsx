import { createMemo, For, Show } from "solid-js";
import { api } from "../api";

type Row = Record<string, unknown>;
const IMG = /\.(png|jpe?g|webp)$/i;
const MAX_ROWS = 300;

/** Renders a stage's report.json: scalar/meta block + first array-of-objects as a table. */
export function Report(props: { path: string; report: unknown }) {
  const view = createMemo(() => {
    const r = props.report;
    if (Array.isArray(r)) return { meta: null, rows: r as Row[] };
    if (r && typeof r === "object") {
      const o = r as Row;
      const rows = (["rows", "images", "groups", "items", "entries"]
        .map((k) => o[k])
        .find(Array.isArray) ??
        Object.values(o).find((v) => Array.isArray(v) && v.length && typeof v[0] === "object")) as
        Row[] | undefined;
      const meta = Object.fromEntries(
        Object.entries(o).filter(([, v]) => !Array.isArray(v) || v.length < 20),
      );
      return { meta, rows: rows ?? [] };
    }
    return { meta: r, rows: [] as Row[] };
  });
  const cols = createMemo(() => {
    const rows = view().rows;
    if (!rows.length || typeof rows[0] !== "object") return [];
    return [...new Set(rows.slice(0, 50).flatMap((r) => Object.keys(r)))].slice(0, 12);
  });

  return (
    <>
      <div class="dim" style="margin-bottom:8px">
        {props.path}
      </div>
      <Show when={view().meta && Object.keys(view().meta as object).length}>
        <pre class="json">{JSON.stringify(view().meta, null, 2)}</pre>
      </Show>
      <Show when={cols().length}>
        <p class="dim">
          {view().rows.length} rows (showing {Math.min(view().rows.length, MAX_ROWS)})
        </p>
        <div style="overflow-x:auto">
          <table>
            <thead>
              <tr>
                <For each={cols()}>{(c) => <th>{c}</th>}</For>
              </tr>
            </thead>
            <tbody>
              <For each={view().rows.slice(0, MAX_ROWS)}>
                {(r) => (
                  <tr>
                    <For each={cols()}>
                      {(c) => (
                        <td>
                          <Cell v={r[c]} />
                        </td>
                      )}
                    </For>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </div>
      </Show>
    </>
  );
}

function Cell(props: { v: unknown }) {
  const v = () => props.v;
  return (
    <Show when={v() != null}>
      <Show
        when={typeof v() === "string" && IMG.test(v() as string)}
        fallback={
          <Show when={typeof v() === "object"} fallback={<>{String(v())}</>}>
            <pre class="json">{JSON.stringify(v(), null, 1)}</pre>
          </Show>
        }
      >
        <a href={api.fileUrl(v() as string)} target="_blank">
          <img
            class="thumb"
            loading="lazy"
            src={api.fileUrl(v() as string)}
            title={v() as string}
          />
        </a>
      </Show>
    </Show>
  );
}
