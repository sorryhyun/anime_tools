import { createSignal, type JSX } from "solid-js";
import en, { type Dict } from "./en";
import ko from "./ko";
import ja from "./ja";
import zh from "./zh";

export type { Dict };

const DICTS = { en, ko, ja, zh };
export type Locale = keyof typeof DICTS;
/** The picker's order, and the only place a locale id is spelled. */
export const LOCALES = Object.keys(DICTS) as Locale[];
export const localeName = (l: Locale) => DICTS[l].langName;

const KEY = "lang";
const isLocale = (v: string | null): v is Locale => !!v && v in DICTS;

/** A saved choice wins; otherwise the browser's list is walked, so a
    `ko-KR`/`zh-Hans-CN` visitor lands on their own language and everyone else
    on English. */
function initial(): Locale {
  const saved = localStorage.getItem(KEY);
  if (isLocale(saved)) return saved;
  for (const tag of navigator.languages ?? [navigator.language]) {
    const base = tag.toLowerCase().split("-")[0];
    if (base === "zh") return "zh";
    if (isLocale(base)) return base;
  }
  return "en";
}

const [locale, setLocaleSignal] = createSignal<Locale>(initial());
export { locale };

/** Switch language. Written straight to `localStorage`: this signal lives at
    module scope, where a `createEffect` would never be owned by a root. */
export function setLocale(l: Locale) {
  localStorage.setItem(KEY, l);
  document.documentElement.lang = l;
  setLocaleSignal(l);
}
document.documentElement.lang = locale();

/** The strings, in the current language. Reading it inside JSX is what makes a
    language switch re-render the app — every call site is `t().…`, never
    hoisted into a `const`. */
export const t = (): Dict => DICTS[locale()];

/** Render a message that has markup in the middle of it. Translations stay
 * plain strings with `{0}`/`{1}` slots — a translator never sees a tag, and word
 * order stays theirs — and the component says what each slot is: a `<code>`, a
 * link, a `<b>`.
 */
export function slots(text: string, render: (i: number) => JSX.Element): JSX.Element {
  const out: JSX.Element[] = [];
  const re = /\{(\d+)\}/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    out.push(render(Number(m[1])));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}
