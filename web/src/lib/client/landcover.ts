/** The paper's U-Net in the browser: satellite tiles -> 5.46 m grid -> ONNX Runtime Web -> cleaned class map.
 *  Mirrors the Python pipeline (monitor_nepal/imagery, model.py, postprocess.py) so results match the app. */
import radiometry from "../../../data/eox_radiometry.json";

export const GSD = 5.464; // metres per model pixel: DeepGlobe 2448 px x 0.5 m resized to 224 px (as in the paper)
export const CLASS_NAMES = ["Urban land", "Agriculture land", "Rangeland", "Forest land", "Water", "Barren land", "Unknown"];
export const CLASS_KEYS = ["urban", "agriculture", "rangeland", "forest", "water", "barren", "unknown"] as const;
export const CLASS_RGB: [number, number, number][] = [
  [20, 151, 176], [237, 161, 0], [196, 80, 181], [0, 131, 0], [42, 120, 214], [181, 113, 43], [137, 135, 129],
];
const R = 6378137;

export type ModelKey = "nepal" | "paper";
export interface Imagery {
  key: string;
  label: string;
  url: string; // {z}/{y}/{x}
  zoom: number;
  year?: number;
}

export function esriLatest(): Imagery {
  return { key: "esri", label: "Esri World Imagery (latest, sub-metre)", zoom: 17,
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" };
}
export function eoxYear(year: number): Imagery {
  const layer = year === 2016 ? "s2cloudless_3857" : `s2cloudless-${year}_3857`;
  return { key: `eox-${year}`, label: `Sentinel-2 cloudless ${year} (10 m)${UNRELIABLE_YEARS.has(year) ? " — less reliable" : ""}`, zoom: 15, year,
    url: `https://tiles.maps.eox.at/wmts/1.0.0/${layer}/default/g/{z}/{y}/{x}.jpg` };
}
export const EOX_YEARS = Object.keys((radiometry as { years: Record<string, unknown> }).years).map(Number).sort();
/** The 2016 and 2018 mosaics were processed differently and the model misreads them (e.g. a fully built-up
 *  Kathmandu core comes out barren in 2016 and half rangeland in 2018); from 2019 results match the sub-metre imagery. */
export const UNRELIABLE_YEARS = new Set([2016, 2018]);

/* ---------------- geometry ---------------- */
function mercator(lon: number, lat: number): [number, number] {
  return [R * (lon * Math.PI) / 180, R * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360))];
}

export interface Grid {
  lon: number;
  lat: number;
  sizeM: number;
  w: number;
  h: number;
}

/** A north-up grid of model pixels centred on a point; sides are multiples of 16 (the U-Net needs that). */
export function gridAround(lon: number, lat: number, sizeM: number): Grid {
  const n = Math.max(64, Math.round(sizeM / GSD / 16) * 16);
  return { lon, lat, sizeM: n * GSD, w: n, h: n };
}

/* ---------------- imagery ---------------- */
async function loadTile(url: string): Promise<ImageBitmap | null> {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(url, { mode: "cors" });
      if (res.status === 404) return null;
      if (res.ok) return await createImageBitmap(await res.blob());
    } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 400 * (attempt + 1)));
  }
  return null;
}

/** RGB of the grid, sampled bilinearly (no antialiasing, like OpenCV INTER_LINEAR in training) from a tile mosaic. */
export async function fetchGrid(grid: Grid, img: Imagery, onProgress?: (f: number) => void): Promise<{ rgb: Uint8ClampedArray; valid: Uint8Array }> {
  const z = img.zoom;
  const world = 256 * 2 ** z;
  const [cx, cy] = mercator(grid.lon, grid.lat);
  const scale = 1 / Math.cos((grid.lat * Math.PI) / 180); // mercator metres per ground metre
  const half = (grid.sizeM / 2) * scale;
  const toPx = (x: number) => ((x + Math.PI * R) / (2 * Math.PI * R)) * world;
  const toPy = (y: number) => ((Math.PI * R - y) / (2 * Math.PI * R)) * world;
  const px0 = toPx(cx - half), px1 = toPx(cx + half), py0 = toPy(cy + half), py1 = toPy(cy - half);
  const tx0 = Math.floor(px0 / 256), tx1 = Math.floor(px1 / 256), ty0 = Math.floor(py0 / 256), ty1 = Math.floor(py1 / 256);
  const cols = tx1 - tx0 + 1, rows = ty1 - ty0 + 1;
  const canvas = new OffscreenCanvas(cols * 256, rows * 256);
  const ctx = canvas.getContext("2d")!;
  const jobs: (() => Promise<void>)[] = [];
  let done = 0;
  for (let ty = ty0; ty <= ty1; ty++) for (let tx = tx0; tx <= tx1; tx++) {
    jobs.push(async () => {
      const bmp = await loadTile(img.url.replace("{z}", String(z)).replace("{y}", String(ty)).replace("{x}", String(tx)));
      if (bmp) ctx.drawImage(bmp, (tx - tx0) * 256, (ty - ty0) * 256);
      onProgress?.(++done / jobs.length);
    });
  }
  const queue = [...jobs];
  await Promise.all(Array.from({ length: 8 }, async () => { while (queue.length) await queue.shift()!(); }));
  const mosaic = ctx.getImageData(0, 0, cols * 256, rows * 256).data;
  const mw = cols * 256, mh = rows * 256;
  const rgb = new Uint8ClampedArray(grid.w * grid.h * 3);
  const valid = new Uint8Array(grid.w * grid.h);
  for (let j = 0; j < grid.h; j++) {
    const fy = py0 + ((j + 0.5) / grid.h) * (py1 - py0) - ty0 * 256 - 0.5;
    const y0 = Math.max(0, Math.min(mh - 1, Math.floor(fy))), y1 = Math.min(mh - 1, y0 + 1), dy = Math.min(1, Math.max(0, fy - y0));
    for (let i = 0; i < grid.w; i++) {
      const fx = px0 + ((i + 0.5) / grid.w) * (px1 - px0) - tx0 * 256 - 0.5;
      const x0 = Math.max(0, Math.min(mw - 1, Math.floor(fx))), x1 = Math.min(mw - 1, x0 + 1), dx = Math.min(1, Math.max(0, fx - x0));
      const o = (j * grid.w + i) * 3;
      const a = (y0 * mw + x0) * 4, b = (y0 * mw + x1) * 4, c = (y1 * mw + x0) * 4, d = (y1 * mw + x1) * 4;
      const alpha = mosaic[a + 3] && mosaic[d + 3];
      valid[j * grid.w + i] = alpha ? 1 : 0;
      for (let k = 0; k < 3; k++) {
        rgb[o + k] = (mosaic[a + k] * (1 - dx) + mosaic[b + k] * dx) * (1 - dy) + (mosaic[c + k] * (1 - dx) + mosaic[d + k] * dx) * dy;
      }
    }
  }
  if (img.year) normalizeEox(rgb, img.year);
  return { rgb, valid };
}

/** Map a Sentinel-2 cloudless year's colours onto the 2018-2025 reference (per-channel gain and offset). */
function normalizeEox(rgb: Uint8ClampedArray, year: number) {
  const r = radiometry as { reference: { mean: number[]; std: number[] }; years: Record<string, { mean: number[]; std: number[] }> };
  const y = r.years[String(year)];
  if (!y) return;
  for (let p = 0; p < rgb.length; p += 3) for (let k = 0; k < 3; k++) {
    rgb[p + k] = ((rgb[p + k] - y.mean[k]) / Math.max(y.std[k], 1e-3)) * r.reference.std[k] + r.reference.mean[k];
  }
}

/** Resize any image to w x h (bilinear, no antialiasing) - used for uploads. */
export function resizeRGBA(src: ImageData, w: number, h: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 3);
  const sw = src.width, sh = src.height, s = src.data;
  for (let j = 0; j < h; j++) {
    const fy = ((j + 0.5) * sh) / h - 0.5, y0 = Math.max(0, Math.floor(fy)), y1 = Math.min(sh - 1, y0 + 1), dy = Math.max(0, fy - y0);
    for (let i = 0; i < w; i++) {
      const fx = ((i + 0.5) * sw) / w - 0.5, x0 = Math.max(0, Math.floor(fx)), x1 = Math.min(sw - 1, x0 + 1), dx = Math.max(0, fx - x0);
      const a = (y0 * sw + x0) * 4, b = (y0 * sw + x1) * 4, c = (y1 * sw + x0) * 4, d = (y1 * sw + x1) * 4;
      for (let k = 0; k < 3; k++) out[(j * w + i) * 3 + k] = (s[a + k] * (1 - dx) + s[b + k] * dx) * (1 - dy) + (s[c + k] * (1 - dx) + s[d + k] * dx) * dy;
    }
  }
  return out;
}

/* ---------------- model ---------------- */
type Ort = typeof import("onnxruntime-web");
const sessions = new Map<ModelKey, Promise<import("onnxruntime-web").InferenceSession>>();
let ortPromise: Promise<Ort> | null = null;

async function ort(): Promise<Ort> {
  ortPromise ??= import("onnxruntime-web").then((m) => {
    m.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
    m.env.wasm.numThreads = 1;
    return m;
  });
  return ortPromise;
}

export function loadModel(key: ModelKey): Promise<import("onnxruntime-web").InferenceSession> {
  if (!sessions.has(key)) {
    sessions.set(key, ort().then((m) => m.InferenceSession.create(`/models/${key === "nepal" ? "unet_nepal" : "unet_deepglobe"}.onnx`,
      { executionProviders: ["wasm"], graphOptimizationLevel: "all" })));
  }
  return sessions.get(key)!;
}

/** Class probabilities (h*w*7, NHWC) for an RGB image whose sides are multiples of 16. */
export async function predict(key: ModelKey, rgb: Uint8ClampedArray, w: number, h: number): Promise<Float32Array> {
  const [m, session] = await Promise.all([ort(), loadModel(key)]);
  const x = new Float32Array(w * h * 3);
  for (let i = 0; i < x.length; i++) x[i] = rgb[i] / 255;
  const out = await session.run({ [session.inputNames[0]]: new m.Tensor("float32", x, [1, h, w, 3]) });
  return out[session.outputNames[0]].data as Float32Array;
}

/* ---------------- post-processing and statistics ---------------- */
export function argmax(probs: Float32Array, n: number): { cls: Uint8Array; conf: Float32Array } {
  const cls = new Uint8Array(n), conf = new Float32Array(n);
  for (let p = 0; p < n; p++) {
    let best = 0, bv = -1;
    for (let k = 0; k < 7; k++) { const v = probs[p * 7 + k]; if (v > bv) { bv = v; best = k; } }
    cls[p] = best; conf[p] = bv;
  }
  return { cls, conf };
}

/** Regions smaller than minPx (8-connected) take the class of the nearest surviving region (as in the app). */
export function clean(cls: Uint8Array, w: number, h: number, minPx = 64): Uint8Array {
  const n = w * h, label = new Int32Array(n).fill(-1), small = new Uint8Array(n);
  const stack: number[] = [];
  let next = 0;
  for (let s = 0; s < n; s++) {
    if (label[s] >= 0) continue;
    const c = cls[s], members: number[] = [];
    label[s] = next; stack.push(s);
    while (stack.length) {
      const p = stack.pop()!; members.push(p);
      const x = p % w, y = (p - x) / w;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx, ny = y + dy;
        if ((dx || dy) && nx >= 0 && ny >= 0 && nx < w && ny < h) {
          const q = ny * w + nx;
          if (label[q] < 0 && cls[q] === c) { label[q] = next; stack.push(q); }
        }
      }
    }
    if (members.length < minPx) for (const p of members) small[p] = 1;
    next++;
  }
  const out = cls.slice();
  const queue: number[] = [];
  for (let p = 0; p < n; p++) if (!small[p]) queue.push(p);
  if (!queue.length) return out;
  const seen = Uint8Array.from(small, (v) => 1 - v);
  for (let head = 0; head < queue.length; head++) { // breadth-first: every removed pixel takes its nearest kept class
    const p = queue[head], x = p % w, y = (p - x) / w;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nx = x + dx, ny = y + dy;
      if (nx >= 0 && ny >= 0 && nx < w && ny < h) {
        const q = ny * w + nx;
        if (!seen[q]) { seen[q] = 1; out[q] = out[p]; queue.push(q); }
      }
    }
  }
  return out;
}

/** Share (%) and area (km²) of each class over valid pixels. */
export function composition(cls: Uint8Array, valid?: Uint8Array) {
  const counts = new Array(7).fill(0);
  let total = 0;
  for (let p = 0; p < cls.length; p++) if (!valid || valid[p]) { counts[cls[p]]++; total++; }
  const px = (GSD * GSD) / 1e6;
  return counts.map((c, k) => ({ k, name: CLASS_NAMES[k], share: total ? (100 * c) / total : 0, km2: c * px }));
}

export function paint(canvas: HTMLCanvasElement, w: number, h: number, draw: (img: ImageData) => void) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(w, h);
  draw(img);
  ctx.putImageData(img, 0, 0);
}

export function paintRGB(canvas: HTMLCanvasElement, rgb: Uint8ClampedArray, w: number, h: number) {
  paint(canvas, w, h, (img) => { for (let p = 0; p < w * h; p++) { img.data.set([rgb[p * 3], rgb[p * 3 + 1], rgb[p * 3 + 2], 255], p * 4); } });
}

export function paintClasses(canvas: HTMLCanvasElement, cls: Uint8Array, w: number, h: number, highlight?: Uint8Array) {
  paint(canvas, w, h, (img) => {
    for (let p = 0; p < w * h; p++) {
      const [r, g, b] = CLASS_RGB[cls[p]];
      const dim = highlight && !highlight[p];
      img.data.set(dim ? [235, 235, 232, 255] : [r, g, b, 255], p * 4);
    }
  });
}
