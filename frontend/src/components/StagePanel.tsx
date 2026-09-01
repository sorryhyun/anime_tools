import { createMemo, For, Show } from "solid-js";
import { t } from "../i18n";
import { StageForm } from "./StageForm";
import { HelpToggle } from "./HelpToggle";
import { StatusLine } from "./StatusLine";
import type { JobStatus, Stage, Values } from "../types";

/** The stage runner's body: the run bar + the schema-driven form for the
    current stage. The dock's button strip (in App) picks a *panel*; when that
    panel holds more than one stage the bar leads with a picker for them.

    The bar is three buttons: **Run** the selected image, **Run batch** every
    image the Settings `path_pattern` names, and **Undo** what the last one
    wrote, out of the report it wrote. A Run writes; the text it replaces becomes
    a version badge beside the caption (`revised@2`). */
export function StagePanel(props: {
  cur?: Stage;
  /** Every stage under the open dock button, including the current one. */
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
  onUndo: () => void;
  /** Why Undo is off, or "" when it is on. */
  undoBlocked: string;
  onCancel: () => void;
  /** Titles of the catalog models this stage needs that are not installed --
      the bar warns before a Run stalls on a first-use fetch. */
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
  const aim = () => (props.rel ? t().stage.aim(props.rel) : t().stage.noImage);

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
              title={t().stage.missingModelsHint(props.missingModels!.join(", "))}
              onClick={props.onSettings}
            >
              {t().stage.missingModels(props.missingModels!.length)}
            </button>
          </Show>
          <HelpToggle open={props.help} warn={!!cur()?.notes} onToggle={props.onHelp} />
          <span class="sp" />

          <Show when={scoped()}>
            <button
              class="primary"
              disabled={off() || noImage()}
              title={aim()}
              onClick={() => props.onRun(props.rel ?? null)}
            >
              {t().stage.run}
            </button>
          </Show>
          <button
            classList={{ primary: !scoped() }}
            disabled={off()}
            title={t().stage.batchHint}
            onClick={() => props.onRun(null)}
          >
            {scoped() ? t().stage.runBatch : t().stage.run}
          </button>

          {/* One button, two jobs: nothing to undo while something is running,
              and a long pass has to stay stoppable. */}
          <Show
            when={props.busy}
            fallback={
              <button
                disabled={!!props.undoBlocked}
                title={props.undoBlocked || t().stage.undoHint}
                onClick={props.onUndo}
              >
                {t().stage.undo}
              </button>
            }
          >
            <button onClick={props.onCancel}>{t().stage.cancel}</button>
          </Show>

          <StatusLine status={props.status} />
        </div>

        <div class="stageform">
          <Show when={cur()} fallback={<div class="dim pad">{t().stage.noStages}</div>}>
            {(s) => (
              <Show
                when={s().available}
                fallback={
                  <div class="doc">
                    {t().stage.unavailable(s().title, s().error ?? "")}
                    {`\n\n${t().stage.reinstall}  uv tool install --force "anime-tools @ git+https://github.com/sorryhyun/anime_tools"`}
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
