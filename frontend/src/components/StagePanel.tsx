import { createMemo, For, Show } from "solid-js";
import { StageForm } from "./StageForm";
import { HelpToggle } from "./HelpToggle";
import { StatusLine } from "./StatusLine";
import type { JobStatus, Stage, Values } from "../types";

/** The stage runner's body: the run bar + the schema-driven form for the
    current stage. The dock's button strip (in App) picks a *panel*; when that
    panel holds more than one stage the bar leads with a picker for them, which
    is why several CLIs can share one button.

    The bar is one loop, spelled out rather than hidden behind a mode toggle:

        Run       →  the selected image. Computes the proposals and shows them
                     as a diff on the caption card above; writes nothing.
        Run batch →  the same, over every image the Settings `path_pattern`
                     names.
        Apply     →  writes what you just looked at. It replays the run's
                     report, so it loads no model and can only write text the
                     diff already showed.
        Undo      →  puts those captions back, from the same report. While a
                     job is running this button is Cancel instead — a tagger
                     pass takes minutes and must stay stoppable.

    A stage with no `--apply` (correct, the mask generators, groups) has no dry
    pass to look at: its Run *is* the write, and there is nothing to undo. */
export function StagePanel(props: {
  /** The open stage, already resolved. The panel used to take the whole stage
      list plus an id and re-find it five times. */
  cur?: Stage;
  /** Every stage under the open dock button, including the current one: the
      bar picks between them when the panel holds more than one. */
  siblings?: Stage[];
  onPick: (id: string) => void;
  error?: unknown;
  values: Values;
  setValue: (dest: string, v: unknown) => void;
  reset: () => void;
  busy: boolean;
  /** Another job -- a Settings weights download -- holds the one job slot, so
      starting a stage would only 409. Greys the run buttons; Cancel stays on
      `busy`, since it can only cancel *this* panel's job. */
  locked?: boolean;
  status: JobStatus;
  /** The selected image, or null -- what the per-image Run acts on. */
  rel?: string | null;
  /** Run: `rel` narrows it to one image, null runs the batch. */
  onRun: (rel: string | null) => void;
  /** Apply: no scope of its own -- it writes whatever the last Run proposed. */
  onApply: () => void;
  /** Why Apply is off, or "" when it is on. */
  applyBlocked: string;
  onUndo: () => void;
  /** Why Undo is off, or "" when it is on. */
  undoBlocked: string;
  onCancel: () => void;
  /** Titles of the catalog models this stage needs that are not installed --
      the bar warns before a Run stalls on a surprise first-use fetch. */
  missingModels?: string[];
  onSettings: () => void;
  help: boolean;
  onHelp: () => void;
}) {
  const cur = () => props.cur;
  /** Per-image Run needs a stage that takes a pattern and an image to aim at. */
  const scoped = createMemo(() => !!cur()?.scoped);
  const noImage = createMemo(() => !props.rel);
  const off = createMemo(() => props.busy || !!props.locked || !cur()?.available);
  const aim = () => (props.rel ? `just ${props.rel}` : "select an image in the sidebar first");
  const batchTitle = () => "every image the Settings path_pattern names";

  return (
    <div class="stagepanel">
      <Show when={!props.error} fallback={<div class="err pad">{String(props.error)}</div>}>
        <div class="stagebar">
          <Show when={(props.siblings?.length ?? 0) > 1} fallback={<b>{cur()?.title}</b>}>
            <span class="tabs stagepick">
              <For each={props.siblings}>
                {(s) => (
                  <a
                    classList={{ sel: s.id === props.cur?.id, na: !s.available }}
                    title={s.available ? s.title : s.error}
                    onClick={() => props.onPick(s.id)}
                  >
                    {s.short}
                  </a>
                )}
              </For>
            </span>
          </Show>
          <Show when={props.missingModels?.length}>
            <button
              class="link warn"
              title={
                `Not downloaded yet: ${props.missingModels!.join(", ")}. ` +
                "The first Run fetches them itself — this only moves the wait to a moment you pick."
              }
              onClick={props.onSettings}
            >
              ↓ {props.missingModels!.length} model{props.missingModels!.length === 1 ? "" : "s"}{" "}
              missing
            </button>
          </Show>
          <HelpToggle open={props.help} warn={!!cur()?.notes} onToggle={props.onHelp} />
          <span class="sp" />

          <Show when={scoped()}>
            <button
              classList={{ primary: !cur()?.apply }}
              disabled={off() || noImage()}
              title={aim()}
              onClick={() => props.onRun(props.rel ?? null)}
            >
              Run
            </button>
          </Show>
          <button
            classList={{ primary: !scoped() && !cur()?.apply }}
            disabled={off()}
            title={batchTitle()}
            onClick={() => props.onRun(null)}
          >
            {scoped() ? "Run batch" : "Run"}
          </button>
          <Show when={cur()?.apply}>
            <button
              class="primary"
              disabled={off() || !!props.applyBlocked}
              title={props.applyBlocked || "write what the run proposed"}
              onClick={props.onApply}
            >
              Apply
            </button>
          </Show>

          {/* One button, two jobs: nothing to undo while something is running,
              and a long pass has to stay stoppable. */}
          <Show
            when={props.busy}
            fallback={
              <button
                disabled={!!props.undoBlocked}
                title={props.undoBlocked || "put back the captions the last Apply wrote"}
                onClick={props.onUndo}
              >
                Undo
              </button>
            }
          >
            <button onClick={props.onCancel}>Cancel</button>
          </Show>

          <StatusLine status={props.status} />
        </div>

        <div class="stageform">
          <Show when={cur()} fallback={<div class="dim pad">No stages.</div>}>
            {(s) => (
              <Show
                when={s().available}
                fallback={
                  <div class="doc">
                    {s().title} is unavailable: {s().error}
                    {
                      '\n\nReinstall:  uv tool install --force "anime-tools @ git+https://github.com/sorryhyun/anime_tools"'
                    }
                  </div>
                }
              >
                <StageForm
                  stage={s()}
                  values={props.values}
                  setValue={props.setValue}
                  reset={props.reset}
                  help={props.help}
                  onHelp={props.onHelp}
                />
              </Show>
            )}
          </Show>
        </div>
      </Show>
    </div>
  );
}
