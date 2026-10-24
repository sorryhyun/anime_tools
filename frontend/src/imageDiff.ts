/** Longest side a difference is computed at. Past it the image is scaled down
    first: a 4K pair is 33 MB of pixels per side, and the point of the view is
    where two images part, which survives the downscale. */
export const DIFF_MAX_SIDE = 2048;
/** A channel this far apart (out of 255) counts the pixel as changed. JPEG
    noise between two saves of one picture sits well under it. */
export const DIFF_CHANGED = 16;
/** Width of the thumbnails `align` scores its candidates on: enough to tell a
    six-row shift from none, small enough to cost nothing. */
const ALIGN_SIDE = 256;

/** How a pair is laid on one frame. The frame has `aspect`; `fill` stretches
 * both images onto it, `cover` scales each to cover it and crops the overhang
 * evenly — which is what `buckets.py`'s resize does to a training image, so an
 * output made at bucket size lines up with its source under `cover` and drifts
 * by up to the cropped rows under `fill`. `cropped` says which side lost its
 * edges, for the bar to name.
 */
export interface Alignment {
  aspect: number;
  fit: "fill" | "cover";
  cropped: "a" | "b" | null;
}

export interface Pair {
  a: HTMLImageElement;
  b: HTMLImageElement;
  align: Alignment;
}

/** Two images compared pixel by pixel on their alignment's frame. `abs` is the
 * per-channel absolute difference as RGBA (alpha opaque), ready to be scaled
 * by a gain and put on a canvas; `mean` and `changed` are fractions of 1.
 */
export interface PixelDiff {
  width: number;
  height: number;
  abs: Uint8ClampedArray;
  mean: number;
  changed: number;
}

async function load(url: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.src = url;
  await img.decode();
  return img;
}

const ratio = (img: HTMLImageElement) => img.naturalWidth / img.naturalHeight;

/** The image's pixels on a `w`×`h` frame laid out the way an `<img>` with that
    `object-fit` draws it, over white so a transparent PNG compares the way it
    is shown. */
function pixels(
  img: HTMLImageElement,
  w: number,
  h: number,
  fit: Alignment["fit"],
): Uint8ClampedArray {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  if (fit === "fill") ctx.drawImage(img, 0, 0, w, h);
  else {
    const s = Math.max(w / img.naturalWidth, h / img.naturalHeight);
    const dw = img.naturalWidth * s;
    const dh = img.naturalHeight * s;
    ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
  }
  return ctx.getImageData(0, 0, w, h).data;
}

/** The frame a pair of shapes `aspect` is drawn at, its longer side capped. */
function frame(aspect: number, longest: number): [number, number] {
  const w = aspect >= 1 ? longest : longest * aspect;
  return [Math.max(1, Math.round(w)), Math.max(1, Math.round(w / aspect))];
}

/** Which way of laying the pair on one frame matches them best. Same-shaped
    images have only one answer; otherwise stretching onto the left one's shape
    and center-cropping to either one's shape are each scored on a thumbnail
    and the closest wins — a cropped bucket output beside its source scores
    well under the stretch, while a genuinely rescaled image keeps it. */
function align(a: HTMLImageElement, b: HTMLImageElement): Alignment {
  const ra = ratio(a);
  const rb = ratio(b);
  const stretch: Alignment = { aspect: ra, fit: "fill", cropped: null };
  if (Math.abs(ra - rb) < 1e-3) return stretch;
  const candidates: Alignment[] = [
    stretch,
    // Covering a frame of one image's shape crops the other.
    { aspect: ra, fit: "cover", cropped: "b" },
    { aspect: rb, fit: "cover", cropped: "a" },
  ];
  let best = stretch;
  let bestScore = Infinity;
  for (const al of candidates) {
    const [w, h] = frame(al.aspect, ALIGN_SIDE);
    const pa = pixels(a, w, h, al.fit);
    const pb = pixels(b, w, h, al.fit);
    let sum = 0;
    for (let i = 0; i < pa.length; i += 4)
      sum +=
        Math.abs(pa[i] - pb[i]) + Math.abs(pa[i + 1] - pb[i + 1]) + Math.abs(pa[i + 2] - pb[i + 2]);
    if (sum / (w * h) < bestScore) {
      bestScore = sum / (w * h);
      best = al;
    }
  }
  return best;
}

/** Both images decoded and aligned — what the slider and the diff both draw
    from, so the two views always line the pair up the same way. */
export async function loadPair(aUrl: string, bUrl: string): Promise<Pair> {
  const [a, b] = await Promise.all([load(aUrl), load(bUrl)]);
  return { a, b, align: align(a, b) };
}

/** The pair's difference on its alignment's frame, at the resolution of the
    image that frame shows whole (capped at `DIFF_MAX_SIDE`). */
export function pixelDiff(p: Pair): PixelDiff {
  const whole = p.align.cropped === "a" ? p.b : p.a;
  const [width, height] = frame(
    p.align.aspect,
    Math.min(DIFF_MAX_SIDE, Math.max(whole.naturalWidth, whole.naturalHeight)),
  );
  const pa = pixels(p.a, width, height, p.align.fit);
  const pb = pixels(p.b, width, height, p.align.fit);
  const abs = new Uint8ClampedArray(pa.length);
  let sum = 0;
  let changed = 0;
  for (let i = 0; i < pa.length; i += 4) {
    const r = Math.abs(pa[i] - pb[i]);
    const g = Math.abs(pa[i + 1] - pb[i + 1]);
    const bl = Math.abs(pa[i + 2] - pb[i + 2]);
    abs[i] = r;
    abs[i + 1] = g;
    abs[i + 2] = bl;
    abs[i + 3] = 255;
    sum += r + g + bl;
    if (r > DIFF_CHANGED || g > DIFF_CHANGED || bl > DIFF_CHANGED) changed++;
  }
  const n = width * height;
  return { width, height, abs, mean: sum / (n * 3 * 255), changed: changed / n };
}

/** Paint a difference onto `canvas`, each channel multiplied by `gain`: a
    near-twin's differences are a few levels out of 255, black on black at ×1. */
export function drawDiff(canvas: HTMLCanvasElement, d: PixelDiff, gain: number): void {
  canvas.width = d.width;
  canvas.height = d.height;
  const out = new Uint8ClampedArray(d.abs.length);
  for (let i = 0; i < out.length; i += 4) {
    out[i] = d.abs[i] * gain;
    out[i + 1] = d.abs[i + 1] * gain;
    out[i + 2] = d.abs[i + 2] * gain;
    out[i + 3] = 255;
  }
  canvas.getContext("2d")!.putImageData(new ImageData(out, d.width, d.height), 0, 0);
}
