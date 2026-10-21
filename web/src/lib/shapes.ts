/** Region outlines as SVG paths (equirectangular, 100 units per degree; see scripts/export_web.py). */
import paths0 from "../../data/paths-0.json";
import paths1 from "../../data/paths-1.json";
import paths2 from "../../data/paths-2.json";
import paths3 from "../../data/paths-3.json";

const LON0 = 80.0;
const LAT1 = 30.5;
const COS = Math.cos((28.3 * Math.PI) / 180);

const byLevel: Record<number, Record<string, string>> = { 0: paths0, 1: paths1, 2: paths2, 3: paths3 };

export function pathOf(level: number, code: string): string | undefined {
  return byLevel[level]?.[code];
}
export function allPaths(level: number): Record<string, string> {
  return byLevel[level] ?? {};
}
export const nepalPath = Object.values(paths0)[0] as string;

export function project(lon: number, lat: number): [number, number] {
  return [(lon - LON0) * COS * 100, (LAT1 - lat) * 100];
}

function rings(path: string): [number, number][][] {
  return path.split("M").filter(Boolean).map((ring) =>
    ring.replace("Z", "").split("L").map((pt) => pt.trim().split(" ").map(Number) as [number, number]));
}
/** Is a point inside Nepal's (simplified) outline? */
export function insideNepal(lon: number, lat: number): boolean {
  const [x, y] = project(lon, lat);
  return insidePath(nepalPath, x, y);
}

const ringCache = new Map<string, [number, number][][]>();

function insidePath(path: string, x: number, y: number): boolean {
  let rs = ringCache.get(path);
  if (!rs) {
    rs = rings(path);
    ringCache.set(path, rs);
  }
  let inside = false;
  for (const ring of rs) {
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i];
      const [xj, yj] = ring[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

/** Code of the region at a level that contains a point (simplified outlines, so edges are approximate). */
export function regionAt(level: number, lon: number, lat: number): string | undefined {
  const [x, y] = project(lon, lat);
  for (const [code, path] of Object.entries(allPaths(level))) if (insidePath(path, x, y)) return code;
  return undefined;
}
