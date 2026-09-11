import { createSignal, For, Show } from "solid-js";
import { t } from "../i18n";
import { MASK_KINDS, MASK_ROLES, type MaskKind, type MaskRole } from "../types";
import { browsePath, PathPicker } from "./PathPicker";

/** One entry of the mask stage's `--masks`, split out of its `role:kind:value`
    spelling. That spelling is the field's wire format (the form value is the
    list of strings the flag takes), so it is read and written here and nowhere
    else; the value is everything after the second colon, as the server reads
    it, so a Windows path keeps its drive letter. */
interface Entry {
  role: MaskRole;
  kind: MaskKind;
  value: string;
}

const parse = (spec: string): Entry => {
  const [role = "keep", kind = "text", ...rest] = spec.split(":");
  return {
    role: (MASK_ROLES as readonly string[]).includes(role) ? (role as MaskRole) : "keep",
    kind: (MASK_KINDS as readonly string[]).includes(kind) ? (kind as MaskKind) : "text",
    value: rest.join(":"),
  };
};
const spell = (e: Entry) => `${e.role}:${e.kind}:${e.value}`;

/** A segmented control over one of the two closed axes: both choices are on
    screen at once, so a card says what it is without opening a select. */
function Seg<T extends string>(props: {
  options: readonly T[];
  value: T;
  label: (v: T) => string;
  title: string;
  onPick: (v: T) => void;
}) {
  return (
    <span class="seg" title={props.title}>
      <For each={props.options}>
        {(o) => (
          <button
            type="button"
            classList={{ sel: o === props.value }}
            onClick={() => props.onPick(o)}
          >
            {props.label(o)}
          </button>
        )}
      </For>
    </span>
  );
}

/** The mask stage's list of regions: one card per SAM3 prompt, each saying what
    the loss does with it (keep / ignore) and how SAM3 is asked for it (a text
    prompt, or a learned soft prompt file). A card has the field's full width
    rather than a slice of one line, so a text prompt is a box you can write in
    and a soft prompt is a path with the same `…` chooser every path field has.
    A list emptied to nothing reads back as the stage default server-side, since
    a mask run with no region is the one request the stage refuses. */
export function MaskList(props: {
  value: unknown;
  dirty?: boolean;
  setValue: (v: string[]) => void;
}) {
  /** Which card's fallback browser is open -- see `browsePath`. */
  const [picking, setPicking] = createSignal<number | null>(null);
  const entries = (): Entry[] =>
    Array.isArray(props.value) ? props.value.map((s) => parse(String(s))) : [];
  const commit = (next: Entry[]) => props.setValue(next.map(spell));
  const patch = (i: number, p: Partial<Entry>) =>
    commit(entries().map((e, j) => (j === i ? { ...e, ...p } : e)));
  const remove = (i: number) => commit(entries().filter((_, j) => j !== i));
  const add = () => commit([...entries(), { role: "ignore", kind: "text", value: "" }]);
  /* One entry is one argv item and SAM3 reads one prompt, so the box wraps
     rather than taking a newline. */
  const oneLine = (s: string) => s.replace(/\s*\n\s*/g, " ");

  return (
    <div class="masklist" classList={{ dirty: !!props.dirty }}>
      <For each={entries()}>
        {(e, i) => (
          <div class="maskcard" classList={{ ignore: e.role === "ignore" }}>
            <div class="maskhead">
              <Seg
                options={MASK_ROLES}
                value={e.role}
                label={(r) => t().form.maskRoles[r]}
                title={t().form.maskRoleHint}
                onPick={(role) => patch(i(), { role })}
              />
              <Seg
                options={MASK_KINDS}
                value={e.kind}
                label={(k) => t().form.maskKinds[k]}
                title={t().form.maskKindHint}
                onPick={(kind) => patch(i(), { kind })}
              />
              <span class="sp" />
              <button
                type="button"
                class="icon"
                title={t().form.maskRemove}
                onClick={() => remove(i())}
              >
                ×
              </button>
            </div>
            <Show
              when={e.kind === "soft"}
              fallback={
                <textarea
                  rows={2}
                  value={e.value}
                  placeholder={t().form.maskTextPlaceholder}
                  onInput={(ev) => patch(i(), { value: oneLine(ev.currentTarget.value) })}
                />
              }
            >
              <div class="pathrow">
                <input
                  type="text"
                  value={e.value}
                  placeholder={t().form.maskSoftPlaceholder}
                  onInput={(ev) => patch(i(), { value: ev.currentTarget.value })}
                />
                <button
                  type="button"
                  title={t().common.browse}
                  onClick={() =>
                    void browsePath(
                      "file",
                      e.value,
                      (path) => patch(i(), { value: path }),
                      () => setPicking(i()),
                    )
                  }
                >
                  …
                </button>
              </div>
            </Show>
          </div>
        )}
      </For>
      <Show when={!entries().length}>
        <div class="dim">{t().form.maskEmpty}</div>
      </Show>
      <button type="button" class="link" onClick={add}>
        {t().form.maskAdd}
      </button>
      <PathPicker
        open={picking() !== null}
        onClose={(p) => {
          const i = picking();
          if (p !== null && i !== null) patch(i, { value: p });
          setPicking(null);
        }}
      />
    </div>
  );
}
