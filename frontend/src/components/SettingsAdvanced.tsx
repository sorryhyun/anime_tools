import { createEffect, createMemo, For, on, Show } from "solid-js";
import { createStore, reconcile, unwrap } from "solid-js/store";
import { slots, t } from "../i18n";
import type { SettingsOut } from "../config";
import type { HelpArea } from "../layout";
import { stageNotes, stageTitle } from "../stages";
import { MASK_SETTING, REPORT_SETTING } from "../types";
import type { DatasetRoots, Field, Stage } from "../types";
import { FieldRow, grouped } from "./FieldRow";
import { HelpToggle } from "./HelpToggle";
import { SettingRow, SettingsShell } from "./SettingsShell";

/** The Advanced dialog: the Settings-bound stage defaults, and the preflight
    stage's form as its own block. OK writes each block only if it was
    touched. */
export function SettingsAdvanced(props: {
  open: boolean;
  /** The placeholders for a blank `report_root` / `mask_root`. */
  roots?: DatasetRoots;
  /** One argparse Field per Settings-bound stage flag. */
  fields: Field[];
  defaults: Record<string, string>;
  /** The hidden preflight stage, rendered as its own Settings block. */
  preprocess?: Stage;
  preprocessValues: Record<string, unknown>;
  helpOpen: (area: HelpArea) => boolean;
  onHelp: (area: HelpArea) => void;
  onClose: (out: SettingsOut | null) => void;
}) {
  const defEls: Record<string, HTMLInputElement> = {};
  /** The preflight form's edits, keyed by dest. Seeded from the saved values on
      open and diffed on OK, so an untouched block sends nothing. */
  const [pre, setPre] = createStore<Record<string, unknown>>({});
  createEffect(
    on(
      () => props.open,
      (open) => {
        if (!open) return;
        // `unwrap` first: this arrives as a slice of `config.settings`, a
        // store, and a store is a Proxy -- `structuredClone` refuses one with a
        // DataCloneError and takes the whole dialog down with it.
        setPre(reconcile(structuredClone(unwrap(props.preprocessValues))));
      },
    ),
  );
  /** Fields the preflight block shows: the same filter the stage forms use, so
      the roots and `path_pattern` stay bound server-side and out of here. */
  const preFields = createMemo(() =>
    props.preprocess ? grouped(props.preprocess.fields).flatMap(([, fs]) => fs) : [],
  );

  const close = (ok: boolean) => {
    if (!ok) return props.onClose(null);
    const out: SettingsOut = {};
    const defKeys = [...props.fields.map((f) => f.setting!), REPORT_SETTING, MASK_SETTING];
    const defaults = Object.fromEntries(defKeys.map((k) => [k, defEls[k]?.value.trim() ?? ""]));
    if (defKeys.some((k) => defaults[k] !== (props.defaults[k] ?? ""))) out.defaults = defaults;
    if (JSON.stringify(unwrap(pre)) !== JSON.stringify(props.preprocessValues))
      out.preprocess = { ...unwrap(pre) };
    props.onClose(out);
  };

  return (
    <SettingsShell open={props.open} pane="advanced" onClose={close}>
      <h4>
        {t().settings.stageDefaults}
        <HelpToggle open={props.helpOpen("defaults")} onToggle={() => props.onHelp("defaults")} />
      </h4>
      <Show when={props.helpOpen("defaults")}>
        <p class="dim" style="margin:0 0 8px">
          {slots(t().settings.stageDefaultsHelp, (i) => (
            <code>{["--device", "report_root", "captions/autotag", "groups/groups.json"][i]}</code>
          ))}
        </p>
      </Show>
      <div class="kv">
        <For each={props.fields}>
          {(f) => (
            <SettingRow
              label={f.setting!}
              ref={(el) => (defEls[f.setting!] = el)}
              value={props.defaults[f.setting!] ?? ""}
              placeholder={f.default == null ? "(none)" : String(f.default)}
              hint={f.help}
            />
          )}
        </For>
        <SettingRow
          label={REPORT_SETTING}
          ref={(el) => (defEls[REPORT_SETTING] = el)}
          value={props.defaults[REPORT_SETTING] ?? ""}
          placeholder={props.roots?.report_root}
          hint={t().settings.reportRootHint}
        />
        <SettingRow
          label={MASK_SETTING}
          ref={(el) => (defEls[MASK_SETTING] = el)}
          value={props.defaults[MASK_SETTING] ?? ""}
          placeholder={props.roots?.mask_root}
          hint={t().settings.maskRootHint}
        />
      </div>

      <Show when={props.preprocess}>
        {(pre_) => (
          <>
            <h4>
              {stageTitle(pre_())}
              <HelpToggle
                open={props.helpOpen("preprocess")}
                onToggle={() => props.onHelp("preprocess")}
              />
            </h4>
            <Show when={props.helpOpen("preprocess")}>
              <p class="dim" style="margin:0 0 8px">
                {stageNotes(pre_())}{" "}
                {slots(t().settings.preprocessHelp, () => (
                  <code>target_res</code>
                ))}
              </p>
            </Show>
            <div class="twoup">
              <For each={preFields()}>
                {(f) => (
                  <FieldRow
                    field={f}
                    value={pre[f.dest] ?? f.default}
                    dirty={pre[f.dest] !== undefined && String(pre[f.dest]) !== String(f.default)}
                    setValue={(v) => setPre(f.dest, v)}
                  />
                )}
              </For>
            </div>
          </>
        )}
      </Show>
    </SettingsShell>
  );
}
