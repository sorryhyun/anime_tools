import type { JSX } from "solid-js";

/** One row of a parsed caption: the clause key, then its tags.
 *
 * `bag` is the flat tag bag and every other key is a position clause's header,
 * which is what `pos` colours. The caption card renders these for the caption
 * on disk and the diff renders them for what a Run proposes, so the two views
 * of "a caption is a bag plus some clauses" stay the same shape.
 */
export function ClauseRow(props: {
  label: string;
  /** A position clause rather than the bag. */
  pos?: boolean;
  children: JSX.Element;
}) {
  return (
    <div class="clause">
      <span classList={{ ck: true, pos: props.pos }}>{props.label}</span>
      <span class="tags">{props.children}</span>
    </div>
  );
}
