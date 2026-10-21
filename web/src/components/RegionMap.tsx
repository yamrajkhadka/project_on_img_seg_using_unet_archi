/** An SVG map of a region inside its surroundings, with optional points (towns, fires, earthquakes). */
import { country } from "@/lib/data";
import { allPaths, nepalPath, pathOf, project } from "@/lib/shapes";

export interface MapPoint {
  lon: number;
  lat: number;
  r?: number;
  color?: string;
  label?: string;
}

export function RegionMap({ highlight, context, contextLevel, frame, points = [], label, height = 360, fills }: {
  highlight?: { level: number; code: string };
  /** codes of neighbouring regions drawn faintly (same level as contextLevel) */
  context?: string[];
  contextLevel?: number;
  /** bbox to frame [x0, y0, x1, y1] in map units; default: Nepal */
  frame?: [number, number, number, number];
  points?: MapPoint[];
  label: string;
  height?: number;
  /** fill colour per context region (choropleth) */
  fills?: Record<string, string>;
}) {
  const [x0, y0, x1, y1] = frame ?? country.bbox;
  const pad = Math.max(x1 - x0, y1 - y0) * 0.06;
  const vb = `${x0 - pad} ${y0 - pad} ${x1 - x0 + 2 * pad} ${y1 - y0 + 2 * pad}`;
  const scale = Math.max(x1 - x0, y1 - y0) / 700;
  const ctx = contextLevel != null ? allPaths(contextLevel) : {};
  const hl = highlight ? pathOf(highlight.level, highlight.code) : undefined;
  return (
    <svg viewBox={vb} className="w-full rounded-xl border border-line bg-card" style={{ maxHeight: height }} role="img" aria-label={label}>
      <path d={nepalPath} fill="var(--bg)" stroke="var(--muted)" strokeWidth={1.2 * scale} />
      {(context ?? []).map((c) => ctx[c] && (
        <path key={c} d={ctx[c]} fill={fills?.[c] ?? "none"} stroke={fills ? "var(--card)" : "var(--line)"} strokeWidth={0.8 * scale} />
      ))}
      {hl && <path d={hl} fill="var(--accent)" fillOpacity={0.18} stroke="var(--accent)" strokeWidth={1.6 * scale} />}
      {points.map((p, i) => {
        const [x, y] = project(p.lon, p.lat);
        return (
          <g key={i}>
            <circle cx={x} cy={y} r={(p.r ?? 3) * scale} fill={p.color ?? "var(--crimson)"} fillOpacity={0.75} stroke="var(--card)" strokeWidth={0.5 * scale} />
            {p.label && <text x={x + 5 * scale} y={y + 3 * scale} fontSize={11 * scale} fill="var(--fg)">{p.label}</text>}
          </g>
        );
      })}
    </svg>
  );
}
