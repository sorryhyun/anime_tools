import { For, Show } from "solid-js";
import { t } from "../i18n";
import { MASK_KINDS, MASK_ROLES, type MaskKind, type MaskRole } from "../types";

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

/** The mask stage's list of regions: one row per SAM3 prompt, each saying what
    the loss does with it (keep / ignore) and how SAM3 is asked for it (a text
    prompt, or a learned soft prompt file). Rows are added and removed here; a
    list emptied to nothing reads back as the stage default server-side, since
    a mask run with no region is the one request the stage refuses. */
export function MaskList(props: {
  value: unknown;
  dirty?: boolean;
  setValue: (v: string[]) => void;
}) {
  const entries = (): Entry[] =>
    Array.isArray(props.value) ? props.value.map((s) => parse(String(s))) : [];
  const commit = (next: Entry[]) => props.setValue(next.map(spell));
  const patch = (i: number, p: Partial<Entry>) =>
    commit(entries().map((e, j) => (j === i ? { ...e, ...p } : e)));
  const remove = (i: number) => commit(entries().filter((_, j) => j !== i));
  const add = () => commit([...entries(), { role: "ignore", kind: "text", value: "" }]);

  return (
    <div class="masklist" classList={{ dirty: !!props.dirty }}>
      <For each={entries()}>
        {(e, i) => (
          <div class="maskrow">
            <select
              value={e.role}
              title={t().form.maskRoleHint}
              onChange={(ev) => patch(i(), { role: ev.currentTarget.value as MaskRole })}
            >
              <For each={MASK_ROLES}>
                {(r) => <option value={r}>{t().form.maskRoles[r]}</option>}
              </For>
            </select>
            <select
              value={e.kind}
              title={t().form.maskKindHint}
              onChange={(ev) => patch(i(), { kind: ev.currentTarget.value as MaskKind })}
            >
              <For each={MASK_KINDS}>
                {(k) => <option value={k}>{t().form.maskKinds[k]}</option>}
              </For>
            </select>
            <input
              type="text"
              value={e.value}
              placeholder={
                e.kind === "soft" ? t().form.maskSoftPlaceholder : t().form.maskTextPlaceholder
              }
              onInput={(ev) => patch(i(), { value: ev.currentTarget.value })}
            />
            <button
              type="button"
              class="icon"
              title={t().form.maskRemove}
              onClick={() => remove(i())}
            >
              ×
            </button>
          </div>
        )}
      </For>
      <Show when={!entries().length}>
        <div class="dim">{t().form.maskEmpty}</div>
      </Show>
      <button type="button" class="link" onClick={add}>
        {t().form.maskAdd}
      </button>
    </div>
  );
}
