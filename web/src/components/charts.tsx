/** Server-rendered SVG charts: no client JavaScript, readable without colour (labels and tables), light and dark. */
import { CLASS_KEYS, CLASS_NAME, type ClassKey } from "@/lib/data";
import { fmt } from "@/lib/format";

const color = (k: ClassKey | string) => `var(--c-${k})`;

/** One 100 % stacked bar of land-cover shares with a labelled legend. */
export function ShareBar({ share, label }: { share: Partial<Record<ClassKey, number>>; label: string }) {
  const total = CLASS_KEYS.reduce((s, k) => s + (share[k] ?? 0), 0) || 1;
  const widths = CLASS_KEYS.map((k) => ((share[k] ?? 0) / total) * 1000);
  const starts = widths.map((_, i) => widths.slice(0, i).reduce((s, w) => s + w, 0));
  return (
    <figure>
      <svg viewBox="0 0 1000 56" className="h-14 w-full" role="img" aria-label={label} preserveAspectRatio="none">
        {CLASS_KEYS.map((k, i) => (
          <rect key={k} x={starts[i]} y={4} width={Math.max(0, widths[i])} height={48} fill={color(k)} stroke="var(--bg)" strokeWidth={2} />
        ))}
      </svg>
      <figcaption className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm">
        {CLASS_KEYS.map((k) => (
          <span key={k} className="inline-flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-sm" style={{ background: color(k) }} />
            {CLASS_NAME[k]} <b className="tabular-nums">{fmt(share[k], 1)}%</b>
          </span>
        ))}
      </figcaption>
    </figure>
  );
}

/** Small multiples: each class's share year by year (U-Net with 95 % interval, reference map in grey). */
export function ClassTrends({ trend, reference }: {
  trend: Record<string, Partial<Record<ClassKey, [number, number, number]>>>;
  reference?: Record<string, Partial<Record<ClassKey, number>>>;
}) {
  const years = Object.keys(trend).map(Number).sort((a, b) => a - b);
  const refYears = reference ? Object.keys(reference).map(Number).sort((a, b) => a - b) : [];
  const allYears = [...new Set([...years, ...refYears])].sort((a, b) => a - b);
  if (allYears.length < 2) return null;
  const [y0, y1] = [allYears[0], allYears[allYears.length - 1]];
  const W = 300, H = 170, L = 34, R = 8, T = 10, B = 22;
  const sx = (y: number) => L + ((y - y0) / (y1 - y0)) * (W - L - R);
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {CLASS_KEYS.map((k) => {
        const pts = years.map((y) => ({ y, v: trend[String(y)]?.[k] })).filter((p) => p.v) as { y: number; v: [number, number, number] }[];
        const ref = refYears.map((y) => ({ y, v: reference?.[String(y)]?.[k] })).filter((p) => p.v != null) as { y: number; v: number }[];
        const max = Math.min(100, Math.max(1, ...pts.map((p) => p.v[2]), ...ref.map((p) => p.v)) * 1.1);
        const sy = (v: number) => T + (1 - v / max) * (H - T - B);
        const line = pts.map((p, i) => `${i ? "L" : "M"}${sx(p.y)} ${sy(p.v[0])}`).join("");
        const band = pts.length ? `M${pts.map((p) => `${sx(p.y)} ${sy(p.v[2])}`).join("L")}L${[...pts].reverse().map((p) => `${sx(p.y)} ${sy(p.v[1])}`).join("L")}Z` : "";
        const refLine = ref.map((p, i) => `${i ? "L" : "M"}${sx(p.y)} ${sy(p.v)}`).join("");
        const ticks = [0, max / 2, max].map((v) => Math.round(v));
        return (
          <figure key={k}>
            <figcaption className="text-sm font-medium" style={{ color: color(k) }}>{CLASS_NAME[k]}</figcaption>
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${CLASS_NAME[k]} share by year`}>
              {ticks.map((t) => (
                <g key={t}>
                  <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke="var(--line)" />
                  <text x={L - 4} y={sy(t) + 4} textAnchor="end" fontSize="10" fill="var(--muted)">{t}%</text>
                </g>
              ))}
              {[y0, y1].map((y) => <text key={y} x={sx(y)} y={H - 6} textAnchor={y === y0 ? "start" : "end"} fontSize="10" fill="var(--muted)">{y}</text>)}
              {band && <path d={band} fill={color(k)} opacity={0.15} />}
              {refLine && <path d={refLine} fill="none" stroke="var(--c-ref)" strokeWidth={2} />}
              {ref.map((p) => <circle key={`r${p.y}`} cx={sx(p.y)} cy={sy(p.v)} r={2.5} fill="var(--c-ref)" />)}
              {line && <path d={line} fill="none" stroke={color(k)} strokeWidth={2.5} />}
              {pts.map((p) => <circle key={p.y} cx={sx(p.y)} cy={sy(p.v[0])} r={3} fill={color(k)} />)}
            </svg>
          </figure>
        );
      })}
    </div>
  );
}

/** A single series over years (climate, national statistics). */
export function LineChart({ series, label, unit, digits = 1, highlightFrom }: {
  series: [number, number][]; label: string; unit: string; digits?: number; highlightFrom?: number;
}) {
  if (series.length < 2) return null;
  const W = 900, H = 260, L = 56, R = 12, T = 12, B = 26;
  const xs = series.map((p) => p[0]), vs = series.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [v0, v1] = [Math.min(...vs), Math.max(...vs)];
  const pad = (v1 - v0) * 0.08 || 1;
  v0 -= pad; v1 += pad;
  const sx = (x: number) => L + ((x - x0) / (x1 - x0)) * (W - L - R);
  const sy = (v: number) => T + (1 - (v - v0) / (v1 - v0)) * (H - T - B);
  const d = series.map((p, i) => `${i ? "L" : "M"}${sx(p[0])} ${sy(p[1])}`).join("");
  const ticks = [v0 + pad, (v0 + v1) / 2, v1 - pad];
  const xticks = [x0, Math.round((x0 + x1) / 2), x1];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke="var(--line)" />
          <text x={L - 6} y={sy(t) + 4} textAnchor="end" fontSize="12" fill="var(--muted)">{fmt(t, digits)}</text>
        </g>
      ))}
      {xticks.map((x) => <text key={x} x={sx(x)} y={H - 6} textAnchor="middle" fontSize="12" fill="var(--muted)">{x}</text>)}
      {highlightFrom != null && highlightFrom <= x1 && (
        <rect x={sx(Math.max(highlightFrom - 0.5, x0))} y={T} width={sx(x1) - sx(Math.max(highlightFrom - 0.5, x0))} height={H - T - B} fill="var(--accent)" opacity={0.07} />
      )}
      <path d={d} fill="none" stroke="var(--accent)" strokeWidth={2} />
      <text x={L} y={T + 10} fontSize="12" fill="var(--muted)">{unit}</text>
    </svg>
  );
}

/** Vertical bars with optional horizontal reference lines (e.g. flood levels). */
export function BarChart({ data, label, unit, lines = [] }: {
  data: { x: string; v: number }[]; label: string; unit: string; lines?: { v: number; label: string }[];
}) {
  if (!data.length) return null;
  const W = 900, H = 260, L = 64, R = 12, T = 14, B = 26;
  const max = Math.max(...data.map((d) => d.v), ...lines.map((l) => l.v)) * 1.08 || 1;
  const bw = (W - L - R) / data.length;
  const sy = (v: number) => T + (1 - v / max) * (H - T - B);
  const step = Math.ceil(data.length / 12);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label}>
      {[0, max / 2, max].map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke="var(--line)" />
          <text x={L - 6} y={sy(t) + 4} textAnchor="end" fontSize="12" fill="var(--muted)">{fmt(t)}</text>
        </g>
      ))}
      {data.map((d, i) => (
        <g key={d.x}>
          <rect x={L + i * bw + 1} y={sy(d.v)} width={Math.max(1, bw - 2)} height={H - B - sy(d.v)} fill="var(--accent)" rx={1} />
          {i % step === 0 && <text x={L + i * bw + bw / 2} y={H - 6} textAnchor="middle" fontSize="11" fill="var(--muted)">{d.x}</text>}
        </g>
      ))}
      {lines.map((l) => (
        <g key={l.label}>
          <line x1={L} x2={W - R} y1={sy(l.v)} y2={sy(l.v)} stroke="var(--crimson)" strokeDasharray="5 4" />
          <text x={W - R} y={sy(l.v) - 4} textAnchor="end" fontSize="11" fill="var(--crimson)">{l.label}</text>
        </g>
      ))}
      <text x={L} y={T - 2} fontSize="12" fill="var(--muted)">{unit}</text>
    </svg>
  );
}

/** Hourly values with a "now" marker (air quality, temperature). Past hours solid, forecast hours dashed. */
export function HourlyChart({ times, values, label, unit, now, bands }: {
  times: string[]; values: (number | null)[]; label: string; unit: string; now: string;
  bands?: { max: number; color: string }[];
}) {
  const pts = times.map((t, i) => ({ t, v: values[i] })).filter((p): p is { t: string; v: number } => p.v != null);
  if (pts.length < 2) return null;
  const W = 900, H = 240, L = 48, R = 10, T = 12, B = 26;
  const max = Math.max(...pts.map((p) => p.v)) * 1.1 || 1;
  const min = Math.min(0, ...pts.map((p) => p.v));
  const sx = (i: number) => L + (i / (pts.length - 1)) * (W - L - R);
  const sy = (v: number) => T + (1 - (v - min) / (max - min)) * (H - T - B);
  const nowIdx = Math.max(0, pts.findIndex((p) => p.t >= now));
  const path = (a: number, b: number) => pts.slice(a, b + 1).map((p, j) => `${j ? "L" : "M"}${sx(a + j)} ${sy(p.v)}`).join("");
  const daily = pts.every((p) => p.t.endsWith("T00:00")); // one value per day: mark every 7th day
  const days = pts.map((p, i) => ({ i, t: p.t })).filter((p, i) => (daily ? i % 7 === 0 : p.t.endsWith("T00:00")));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label}>
      {bands?.map((b, i) => {
        const lo = i ? bands[i - 1].max : min;
        if (lo >= max) return null;
        return <rect key={b.max} x={L} width={W - L - R} y={sy(Math.min(b.max, max))} height={sy(lo) - sy(Math.min(b.max, max))} fill={b.color} opacity={0.08} />;
      })}
      {[min, (min + max) / 2, max].map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} stroke="var(--line)" />
          <text x={L - 6} y={sy(t) + 4} textAnchor="end" fontSize="12" fill="var(--muted)">{fmt(t)}</text>
        </g>
      ))}
      {days.map((d) => (
        <g key={d.t}>
          <line x1={sx(d.i)} x2={sx(d.i)} y1={T} y2={H - B} stroke="var(--line)" />
          <text x={sx(d.i) + 4} y={H - 8} fontSize="11" fill="var(--muted)">{new Date(d.t).toLocaleDateString("en-GB", { weekday: "short", day: "numeric" })}</text>
        </g>
      ))}
      <path d={path(0, nowIdx)} fill="none" stroke="var(--accent)" strokeWidth={2.2} />
      <path d={path(nowIdx, pts.length - 1)} fill="none" stroke="var(--accent)" strokeWidth={2.2} strokeDasharray="6 5" />
      <line x1={sx(nowIdx)} x2={sx(nowIdx)} y1={T} y2={H - B} stroke="var(--crimson)" strokeWidth={1.5} />
      <text x={sx(nowIdx) + 4} y={T + 10} fontSize="11" fill="var(--crimson)">now</text>
      <text x={L} y={T + 10} fontSize="12" fill="var(--muted)">{unit}</text>
    </svg>
  );
}
