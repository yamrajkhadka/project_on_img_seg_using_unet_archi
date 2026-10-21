/** Districts coloured by land-cover class, in five equal steps of a single hue, with a legend. Several maps on one
 *  page share a single copy of the district outlines (<use> references), which keeps the page small. */
import Link from "next/link";
import { CLASS_NAME, type ClassKey, country, regionsAt } from "@/lib/data";
import { fmt } from "@/lib/format";
import { latestShare } from "@/lib/insights";
import { allPaths, nepalPath } from "@/lib/shapes";

const RAMPS: Partial<Record<ClassKey, string[]>> = {
  forest: ["#e3f1e3", "#b4d9b4", "#78b878", "#3c913c", "#0d630d"],
  urban: ["#e1f0f4", "#b0d9e4", "#6fb8cc", "#2f8fa9", "#0b5f75"],
  agriculture: ["#fbf1d6", "#f3d993", "#e8bd4c", "#c99413", "#8f6500"],
};

export function Choropleths({ keys, idPrefix = "dist" }: { keys: ClassKey[]; idPrefix?: string }) {
  const districts = regionsAt(2);
  const paths = allPaths(2);
  const [x0, y0, x1, y1] = country.bbox;
  const pad = (x1 - x0) * 0.04;
  const vb = `${x0 - pad} ${y0 - pad} ${x1 - x0 + 2 * pad} ${y1 - y0 + 2 * pad}`;
  return (
    <>
      <svg width="0" height="0" className="absolute" aria-hidden>
        <defs>{districts.map((d) => <path key={d.code} id={`${idPrefix}-${d.code}`} d={paths[d.code]} />)}</defs>
      </svg>
      {keys.map((k) => {
        const values = districts.map((d) => ({ d, v: latestShare(d.code)?.[k] })).filter((x): x is { d: typeof x.d; v: number } => x.v != null);
        const step = Math.max(...values.map((x) => x.v)) / 5 || 1;
        const ramp = RAMPS[k] ?? RAMPS.forest!;
        const top = [...values].sort((a, b) => b.v - a.v).slice(0, 5);
        return (
          <figure key={k} className="mb-10">
            <figcaption className="mb-2 font-medium">{CLASS_NAME[k]} share of each district</figcaption>
            <svg viewBox={vb} className="w-full rounded-xl border border-line bg-card" style={{ maxHeight: 360 }} role="img" aria-label={`${CLASS_NAME[k]} share of each district of Nepal`}>
              <path d={nepalPath} fill="var(--bg)" />
              {values.map(({ d, v }) => <use key={d.code} href={`#${idPrefix}-${d.code}`} fill={ramp[Math.min(4, Math.floor(v / step))]} stroke="var(--card)" strokeWidth={0.8} />)}
            </svg>
            <div className="mt-2 flex flex-wrap items-center gap-3 text-sm">
              {ramp.map((c, i) => <span key={c} className="inline-flex items-center gap-1"><span className="inline-block h-3 w-5 rounded-sm" style={{ background: c }} />{fmt(i * step, 0)}–{fmt((i + 1) * step, 0)}%</span>)}
            </div>
            <p className="mt-2 text-sm">Highest: {top.map((x, i) => <span key={x.d.code}>{i ? ", " : ""}<Link href={`/districts/${x.d.slug}`}>{x.d.name}</Link> {fmt(x.v, 0)}%</span>)}.</p>
          </figure>
        );
      })}
    </>
  );
}
