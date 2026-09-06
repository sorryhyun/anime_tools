import { createSignal } from "solid-js";
import { trackPointer } from "./state";

/** Ctrl/⌘+scroll scales the preview in place, anchored on the pointer so the
    detail aimed at stays under it. */
const ZOOM_MAX = 12;

/** Magnification of whatever a frame is showing, and the pan that keeps the
 * zoomed-in point where it was. `pan` is in screen px and is only ever
 * non-zero while zoomed, so leaving the zoom leaves the picture centred. The
 * three handlers go on the frame; `style` goes on the thing inside it.
 */
export function createZoomPan() {
  const [zoom, setZoom] = createSignal(1);
  const [pan, setPan] = createSignal({ x: 0, y: 0 });
  const reset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };
  const style = () => ({
    transform: `translate(${pan().x}px, ${pan().y}px) scale(${zoom()})`,
  });

  /** Zoom about the pointer. The frame centres the picture, so a point sits at
      `centre + pan + zoom*v`; holding it still across a zoom change is one solve
      for the new pan. */
  function wheel(e: WheelEvent & { currentTarget: HTMLElement }) {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    const z0 = zoom();
    const z = Math.min(ZOOM_MAX, Math.max(1, z0 * Math.exp(-e.deltaY / 400)));
    if (z === z0) return;
    if (z === 1) return reset();
    const r = e.currentTarget.getBoundingClientRect();
    const p = pan();
    const vx = (e.clientX - (r.left + r.width / 2) - p.x) / z0;
    const vy = (e.clientY - (r.top + r.height / 2) - p.y) / z0;
    setZoom(z);
    setPan({
      x: e.clientX - (r.left + r.width / 2) - z * vx,
      y: e.clientY - (r.top + r.height / 2) - z * vy,
    });
  }

  /** Drag to pan, but only once there is something off-frame to reach. */
  function drag(e: PointerEvent) {
    if (zoom() === 1 || e.button !== 0) return;
    const p0 = pan();
    const x0 = e.clientX;
    const y0 = e.clientY;
    trackPointer(e, (ev) => setPan({ x: p0.x + (ev.clientX - x0), y: p0.y + (ev.clientY - y0) }));
  }

  return { zoom, style, wheel, drag, reset };
}
