import type { Metadata } from "next";
import Link from "next/link";
import { RegionMap } from "@/components/RegionMap";
import { PageHeader, Section, Source } from "@/components/ui";
import { regionsAt, rivers } from "@/lib/data";
import { dayOfYear, fmt, nptDate, todayNpt } from "@/lib/format";
import { riverFlow } from "@/lib/live";

export const revalidate = 3600;

export const metadata: Metadata = {
  title: "Nepal River Levels Today & Flood Watch: Koshi, Narayani, Karnali, Bagmati",
  description: "Today's flow of Nepal's major rivers (Koshi, Narayani, Karnali, Mahakali, Bagmati, West Rapti) against normal and flood levels, with 7-day forecasts and flood history since 2004.",
  alternates: { canonical: "/rivers" },
};

export default async function Page() {
  const flows = await Promise.all(rivers.rivers.map((r) => riverFlow(r)));
  const today = todayNpt();
  const doy = dayOfYear(today);
  return (
    <>
      <PageHeader eyebrow={`Live · ${nptDate(new Date())}`} title="Rivers and floods in Nepal"
        intro={<p>Nepal&apos;s rivers carry Himalayan snowmelt and monsoon rain to the Ganges. Floods peak from July to September. Today&apos;s modelled flow at the main gauge of six major rivers, against the normal for the date and flood levels.</p>} />
      <RegionMap label="Major river gauges of Nepal" contextLevel={1} context={regionsAt(1).map((p) => p.code)}
        points={rivers.rivers.map((r) => ({ lon: r.lon, lat: r.lat, r: 5, color: "var(--accent)", label: r.name }))} />
      <Section title="Flow today">
        <div className="overflow-x-auto"><table className="data">
          <thead><tr><th>River</th><th>Gauge</th><th className="num">Today (m³/s)</th><th className="num">% of normal</th><th className="num">7-day max</th><th className="num">2-yr flood</th><th className="num">10-yr flood</th></tr></thead>
          <tbody>{rivers.rivers.map((r, i) => {
            const f = flows[i];
            const idx = f ? f.daily.time.indexOf(today) : -1;
            const q = idx >= 0 ? f!.daily.river_discharge[idx] : null;
            const normal = rivers.climatology[r.key]?.find((c) => c[0] === doy)?.[2];
            const max = f && idx >= 0 ? Math.max(...f.daily.river_discharge.slice(idx)) : null;
            const thr = rivers.thresholds[r.key];
            return (
              <tr key={r.key}><td><Link href={`/rivers/${r.key}`}>{r.name}</Link></td><td>{r.gauge}</td><td className="num">{fmt(q)}</td>
                <td className="num">{q != null && normal ? `${fmt((q / normal) * 100)}%` : "—"}</td><td className="num">{fmt(max)}</td>
                <td className="num">{fmt(thr?.q2)}</td><td className="num">{fmt(thr?.q10)}</td></tr>
            );
          })}</tbody>
        </table></div>
      </Section>
      <Source>GloFAS v4 (Copernicus EMS / ECMWF) via Open-Meteo; normal and flood levels from 2004–2023.</Source>
    </>
  );
}
