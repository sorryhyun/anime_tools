import type { JSX } from "solid-js";

/** One row of a parsed caption: the clause key, then its tags. `bag` is the
 * flat tag bag and every other key is a position clause's header, which is what
 * `pos` colours. Shared by the caption card and the diff.
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
