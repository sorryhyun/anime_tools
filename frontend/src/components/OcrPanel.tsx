import { For, Show } from "solid-js";
import { t } from "../i18n";
import type { OcrLine } from "../types";

/** What the OCR stage read out of the picture — the `{stem}.ocr.txt` sidecar.

    Deliberately *not* a rung of the caption ladder, and this is the component
    where that shows: there is no badge, no editor and nothing to click into the
    caption. The ladder's rungs are captions — texts the file above them could be
    made to say — and these are the words that are in the image. Putting
    `こんにちは` in the badge row would offer to make the caption say it.

    So it reads as evidence sitting under the captions: what was found and how
    sure the recognizer was. Read-only, because the sidecar is
    generated — editing it would be editing a claim about pixels, and the way to
    change it is to run the stage again. */
export function OcrPanel(props: { lines: OcrLine[] }) {
  return (
    <Show when={props.lines.length}>
      <div class="card ocr">
        <div class="card-h">
          <b>{t().ocr.title}</b>
          <span class="badge">{t().ocr.count(props.lines.length)}</span>
          <span class="dim">{t().ocr.readOnly}</span>
        </div>
        <table class="ocrlines">
          <tbody>
            <For each={props.lines}>
              {(line) => (
                <tr>
                  <td class="dim seq">{line.seq}</td>
                  <td class="text">{line.text}</td>
                  {/* The score is the one number worth reading per line: a
                      vertical Japanese column comes back sideways and scores
                      low, which is what the stage's --min_score is filtering
                      on. Warn-coloured below 0.8 so a marginal line is
                      visible without having to compare digits. */}
                  <td
                    classList={{ dim: true, score: true, low: line.score < 0.8 }}
                    title={t().ocr.scoreHint}
                  >
                    {line.score.toFixed(2)}
                  </td>
                  <td class="dim box mono" title={t().ocr.boxHint}>
                    {line.box[0]},{line.box[1]}
                  </td>
                </tr>
              )}
            </For>
          </tbody>
        </table>
      </div>
    </Show>
  );
}
