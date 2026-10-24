/** Longest side a difference is computed at. Past it the left image is scaled
    down first: a 4K pair is 33 MB of pixels per side, and the point of the view
    is where two images part, which survives the downscale. */
export const DIFF_MAX_SIDE = 2048;
/** A channel this far apart (out of 255) counts the pixel as changed. JPEG
    noise between two saves of one picture sits well under it. */
export const DIFF_CHANGED = 16;

/** Two images compared pixel by pixel on the left one's grid. `abs` is the
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

/** The image's pixels at `w`×`h`, stretched if its own shape differs, over
    white so a transparent PNG compares the way it is drawn. */
function pixels(img: HTMLImageElement, w: number, h: number): Uint8ClampedArray {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

/** Both images on the first one's grid (capped at `DIFF_MAX_SIDE`), the second
    resampled onto it — the same frame the slider stretches it into, so the two
    views line the pair up the same way. */
export async function pixelDiff(aUrl: string, bUrl: string): Promise<PixelDiff> {
  const [a, b] = await Promise.all([load(aUrl), load(bUrl)]);
  const s = Math.min(1, DIFF_MAX_SIDE / Math.max(a.naturalWidth, a.naturalHeight));
  const width = Math.max(1, Math.round(a.naturalWidth * s));
  const height = Math.max(1, Math.round(a.naturalHeight * s));
  const pa = pixels(a, width, height);
  const pb = pixels(b, width, height);
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
