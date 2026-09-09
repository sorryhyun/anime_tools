import { type JSX } from "solid-js";

/** The markdown the guidebook is written in, and nothing more.
 *
 * The books under `anime_tools/gui/guidebooks/` are the only input, so this
 * covers exactly what they use — headings, paragraphs, fenced code, tables,
 * blockquotes, lists, `hr`, and inline code / links / bold / italic — and stops
 * there. A parser generator or a markdown dependency would be a bigger surface
 * than the four files it renders; anything a book grows beyond this list is a
 * reason to extend the table below, not to reach for one.
 *
 * Nothing here builds HTML from the text: every node is a JSX element and the
 * text lands in it as a child, so a `<script>` written into a book is a
 * paragraph saying `<script>`.
 */

/** GitHub's heading anchor, which is what a book's own table of contents links
    to: lowercased, punctuation dropped, spaces hyphenated. Unicode letters are
    kept, so `## 1. 이 도구는` and `#1-이-도구는` still meet. */
export function slug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N} _-]/gu, "")
    .trim()
    .replace(/\s+/g, "-");
}

/** `` `code` ``, `[text](href)`, `**bold**`, `*em*` — one pass, first match
    wins, everything else is text. Link text is run through again, since the
    books write `[`docs/masking.md`](…)`. */
function inline(src: string, href: (raw: string) => string): JSX.Element[] {
  const out: JSX.Element[] = [];
  // The `_` form is guarded against the middle of a word: these books are full
  // of `path_pattern` and `min_pixels` written outside backticks, and CommonMark
  // does not emphasise there either.
  const re =
    /`([^`]+)`|\[([^\]]*)\]\(([^)]+)\)|\*\*([^*]+)\*\*|\*([^*]+)\*|(?<![\p{L}\p{N}])_([^_]+)_(?![\p{L}\p{N}])/gu;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) {
    if (m.index > last) out.push(src.slice(last, m.index));
    if (m[1] !== undefined) out.push(<code>{m[1]}</code>);
    else if (m[3] !== undefined) {
      const url = href(m[3]);
      // An in-page anchor stays one: the dialog catches the click and scrolls
      // its own box. Everything else leaves for GitHub in a new tab, since the
      // panel is the app and a doc must not replace it.
      const away = !url.startsWith("#");
      out.push(
        <a href={url} target={away ? "_blank" : undefined} rel={away ? "noreferrer" : undefined}>
          {inline(m[2] ?? "", href)}
        </a>,
      );
    } else if (m[4] !== undefined) out.push(<b>{m[4]}</b>);
    else out.push(<i>{m[5] ?? m[6]}</i>);
    last = m.index + m[0].length;
  }
  if (last < src.length) out.push(src.slice(last));
  return out;
}

/** A table row's cells. A `\|` inside one is the escape, so an inline `a|b`
    does not open a column. */
function cells(row: string): string[] {
  return row
    .trim()
    .replace(/^\||\|$/g, "")
    .split(/(?<!\\)\|/)
    .map((c) => c.replace(/\\\|/g, "|").trim());
}

const HEADING = /^(#{1,6})\s+(.*)$/;
const BULLET = /^\s*[-*+]\s+/;
const NUMBER = /^\s*(\d+)[.)]\s+/;
const RULE = /^\s*(-{3,}|\*{3,}|_{3,})\s*$/;
const FENCE = /^\s*```/;
const SEPARATOR = /^\s*\|?[\s:|-]+\|[\s:|-]*$/;
/** What ends a paragraph besides a blank line. */
const isBlockStart = (l: string) =>
  HEADING.test(l) ||
  FENCE.test(l) ||
  RULE.test(l) ||
  BULLET.test(l) ||
  NUMBER.test(l) ||
  l.startsWith(">");

export function Markdown(props: { text: string; base: string }): JSX.Element {
  /** Resolve a link the way the file it came from would: relative to the
      book's own directory on GitHub. `new URL` does the `../../../docs/` walk,
      so the books keep writing ordinary relative links. */
  const href = (raw: string) => {
    if (raw.startsWith("#") || /^[a-z]+:/i.test(raw)) return raw;
    try {
      return new URL(raw, props.base).href;
    } catch {
      return raw;
    }
  };

  const render = (text: string): JSX.Element[] => {
    const lines = text.replace(/\r\n/g, "\n").split("\n");
    const out: JSX.Element[] = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) {
        i++;
        continue;
      }
      if (FENCE.test(line)) {
        const body: string[] = [];
        i++;
        while (i < lines.length && !FENCE.test(lines[i])) body.push(lines[i++]);
        i++; // the closing fence
        out.push(
          <pre>
            <code>{body.join("\n")}</code>
          </pre>,
        );
        continue;
      }
      const h = HEADING.exec(line);
      if (h) {
        const level = h[1].length;
        const text = h[2].trim();
        // The id is what a book's table of contents aims at; the tag is picked
        // by level rather than built as a string, so nothing here is markup.
        const id = slug(text);
        const kids = inline(text, href);
        out.push(
          level <= 1 ? (
            <h1 id={id}>{kids}</h1>
          ) : level === 2 ? (
            <h2 id={id}>{kids}</h2>
          ) : level === 3 ? (
            <h3 id={id}>{kids}</h3>
          ) : (
            <h4 id={id}>{kids}</h4>
          ),
        );
        i++;
        continue;
      }
      if (RULE.test(line)) {
        out.push(<hr />);
        i++;
        continue;
      }
      if (line.startsWith(">")) {
        const body: string[] = [];
        while (i < lines.length && (lines[i].startsWith(">") || (body.length && lines[i].trim())))
          body.push(lines[i++].replace(/^>\s?/, ""));
        out.push(<blockquote>{render(body.join("\n"))}</blockquote>);
        continue;
      }
      if (lines[i + 1] !== undefined && line.includes("|") && SEPARATOR.test(lines[i + 1])) {
        const head = cells(line);
        i += 2;
        const rows: string[][] = [];
        while (i < lines.length && lines[i].includes("|")) rows.push(cells(lines[i++]));
        out.push(
          <table>
            <thead>
              <tr>
                {head.map((c) => (
                  <th>{inline(c, href)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr>
                  {r.map((c) => (
                    <td>{inline(c, href)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>,
        );
        continue;
      }
      const ordered = NUMBER.test(line);
      if (ordered || BULLET.test(line)) {
        const marker = ordered ? NUMBER : BULLET;
        const start = ordered ? Number(NUMBER.exec(line)![1]) : 1;
        const items: string[] = [];
        while (i < lines.length && lines[i].trim()) {
          if (marker.test(lines[i])) items.push(lines[i].replace(marker, ""));
          else if (isBlockStart(lines[i]) || items.length === 0) break;
          // A wrapped item continues at column 0 in these books, so a line that
          // starts no block of its own belongs to the item above it.
          else items[items.length - 1] += " " + lines[i].trim();
          i++;
        }
        const li = items.map((it) => <li>{inline(it, href)}</li>);
        out.push(ordered ? <ol start={start}>{li}</ol> : <ul>{li}</ul>);
        continue;
      }
      const para: string[] = [];
      while (i < lines.length && lines[i].trim() && !(para.length && isBlockStart(lines[i])))
        para.push(lines[i++].trim());
      out.push(<p>{inline(para.join(" "), href)}</p>);
    }
    return out;
  };

  return <div class="md">{render(props.text)}</div>;
}
