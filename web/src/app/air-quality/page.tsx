import type { Metadata } from "next";
import Link from "next/link";
import { RegionMap } from "@/components/RegionMap";
import { Badge, PageHeader, Section, Source } from "@/components/ui";
import { placeLabel, regionsAt, townOf } from "@/lib/data";
import { AQI_BANDS, aqiBand, fmt, nptDate } from "@/lib/format";
import { airMany } from "@/lib/live";

export const revalidate = 1800;

export const metadata: Metadata = {
  title: "Nepal Air Quality Today: Live AQI & PM2.5 in All 77 Districts",
  description: "Live air quality index (US AQI) and PM2.5 for Kathmandu, Pokhara, Biratnagar and every district of Nepal, ranked from most to least polluted, with health advice.",
  alternates: { canonical: "/air-quality" },
};

const TONE_COLOR: Record<string, string> = { good: "var(--good)", warn: "var(--warn)", bad: "var(--bad)", muted: "var(--muted)" };

export default async function Page() {
  const districts = regionsAt(2);
  const data = await airMany(districts.map((d) => townOf(d.code)!));
  const rows = districts.map((d, i) => ({ d, t: townOf(d.code)!, c: data?.[i]?.current })).sort((a, b) => (b.c?.us_aqi ?? -1) - (a.c?.us_aqi ?? -1));
  return (
    <>
      <PageHeader eyebrow={`Live · ${nptDate(new Date())}`} title="Air quality in Nepal today"
        intro={<p>US Air Quality Index and fine-particle (PM2.5) pollution at the main town of each of Nepal&apos;s 77 districts, refreshed every 30 minutes. Winter and spring smog from fires, brick kilns, traffic and temperature inversions makes Nepal&apos;s air among the most polluted in South Asia from November to April.</p>} />
      <RegionMap label="Air quality across Nepal" contextLevel={2} context={districts.map((d) => d.code)}
        points={rows.filter((r) => r.c).map((r) => ({ lon: r.t.lon, lat: r.t.lat, r: 5, color: TONE_COLOR[aqiBand(r.c!.us_aqi).tone] }))} />
      <Section title="All districts, most polluted first">
        <div className="overflow-x-auto">
          <table className="data">
            <thead><tr><th>Place</th><th className="num">US AQI</th><th>Category</th><th className="num">PM2.5</th><th className="num">PM10</th></tr></thead>
            <tbody>{rows.map(({ d, c }) => (
              <tr key={d.code}><td><Link href={`/air-quality/${d.slug}`}>{placeLabel(d)}</Link></td><td className="num">{fmt(c?.us_aqi)}</td>
                <td><Badge tone={aqiBand(c?.us_aqi).tone}>{aqiBand(c?.us_aqi).label}</Badge></td><td className="num">{fmt(c?.pm2_5, 0)}</td><td className="num">{fmt(c?.pm10, 0)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </Section>
      <Section title="What the AQI numbers mean">
        <table className="data"><thead><tr><th>US AQI</th><th>Category</th><th>Advice</th></tr></thead>
          <tbody>{AQI_BANDS.map((b, i) => <tr key={b.label}><td>{i ? `${AQI_BANDS[i - 1].max + 1}–${b.max === Infinity ? "500" : b.max}` : `0–${b.max}`}</td><td>{b.label}</td><td>{b.advice}</td></tr>)}</tbody>
        </table>
      </Section>
      <Source>Copernicus Atmosphere Monitoring Service (CAMS) via Open-Meteo; modelled values at each district&apos;s main town.</Source>
    </>
  );
}
