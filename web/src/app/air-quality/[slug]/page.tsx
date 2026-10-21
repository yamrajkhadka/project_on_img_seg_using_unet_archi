import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { areaTitle } from "@/components/AreaPage";
import { HourlyChart } from "@/components/charts";
import { Badge, Breadcrumbs, Card, JsonLd, PageHeader, Section, Source, Stat } from "@/components/ui";
import { placeLabel, regionByCode, regionBySlug, regionsAt, townOf } from "@/lib/data";
import { AQI_BANDS, aqiBand, fmt, nptDate } from "@/lib/format";
import { ordinal } from "@/lib/insights";
import { air, airMany } from "@/lib/live";

export const revalidate = 1800;

export function generateStaticParams() {
  return []; // rendered on first visit, then refreshed every 30 minutes
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const d = regionBySlug(2, (await params).slug);
  if (!d) return {};
  const place = placeLabel(d);
  const title = `Air Quality in ${place} Today: Live AQI & PM2.5`;
  const description = `Live air quality index (US AQI), PM2.5, PM10, ozone and NO₂ in ${place}, Nepal, with the hourly forecast for the next 4 days and health advice.`;
  return { title, description, alternates: { canonical: `/air-quality/${d.slug}` }, openGraph: { title, description } };
}

const BAND_COLORS = ["var(--good)", "#d4a000", "var(--warn)", "var(--bad)", "#8f3f97", "#7e0023"];

export default async function Page({ params }: { params: Promise<{ slug: string }> }) {
  const d = regionBySlug(2, (await params).slug);
  if (!d) notFound();
  const town = townOf(d.code)!;
  const place = placeLabel(d);
  const province = regionByCode(d.parent)!;
  const districts = regionsAt(2);
  const [a, all] = await Promise.all([air(town), airMany(districts.map((x) => townOf(x.code)!))]);
  const c = a?.current;
  const band = aqiBand(c?.us_aqi);
  const ranked = all ? districts.map((x, i) => ({ x, v: all[i]?.current?.us_aqi })).filter((r) => r.v != null).sort((p, q) => (q.v ?? 0) - (p.v ?? 0)) : [];
  const rank = ranked.findIndex((r) => r.x.code === d.code);
  const days = a ? [...new Set(a.hourly.time.map((t) => t.slice(0, 10)))].map((day) => {
    const idx = a.hourly.time.map((t, i) => (t.startsWith(day) ? i : -1)).filter((i) => i >= 0);
    const vals = idx.map((i) => a.hourly.us_aqi[i]).filter((v) => v != null);
    const pm = idx.map((i) => a.hourly.pm2_5[i]).filter((v) => v != null);
    return { day, max: vals.length ? Math.max(...vals) : null, pm: pm.length ? pm.reduce((s, v) => s + v, 0) / pm.length : null };
  }) : [];
  const faq = c ? [
    { q: `Is the air quality in ${place} good today?`, a: `The US AQI in ${place} is ${fmt(c.us_aqi)} (${band.label.toLowerCase()}) as of ${c.time.slice(11)} Nepal time. ${band.advice}` },
    { q: `What is the PM2.5 level in ${place}?`, a: `PM2.5 is ${fmt(c.pm2_5, 0)} µg/m³ and PM10 ${fmt(c.pm10, 0)} µg/m³. The WHO guideline for the 24-hour mean of PM2.5 is 15 µg/m³.` },
    { q: `When is air pollution worst in ${d.name}?`, a: `In Nepal, air quality is usually worst from November to April, when dry weather, stagnant winter air, crop-residue and forest fires and brick kilns add to traffic pollution; the monsoon (June–September) washes the air clean.` },
  ] : [];
  return (
    <>
      <Breadcrumbs items={[{ href: "/air-quality", label: "Air quality" }, { label: place }]} />
      <PageHeader eyebrow={`Live · ${nptDate(new Date())} · ${province.name} Province`} title={`Air quality in ${place} today`}
        intro={<p>Live air quality index and pollutants at {town.town}{town.town !== d.name ? `, the main town of ${d.name} district` : ""}, from the Copernicus (CAMS) air-quality forecast, refreshed every 30 minutes.</p>} />
      {c ? (
        <>
          <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="US AQI now" value={fmt(c.us_aqi)} tone={band.tone} sub={<Badge tone={band.tone}>{band.label}</Badge>} />
            <Stat label="PM2.5" value={`${fmt(c.pm2_5, 0)}`} sub="µg/m³ (WHO 24-h guideline: 15)" />
            <Stat label="PM10" value={`${fmt(c.pm10, 0)}`} sub="µg/m³" />
            <Stat label="Rank among 77 districts" value={rank >= 0 ? ordinal(rank + 1) : "—"} sub="1st = most polluted right now" />
          </div>
          <Card className="mb-10"><b>Health advice:</b> {band.advice}</Card>
          <Section title="Hourly air quality: last 3 days and the next 4" note="US AQI; solid line measured-model history, dashed line forecast. Shaded bands: AQI categories.">
            <HourlyChart times={a!.hourly.time} values={a!.hourly.us_aqi} now={c.time} unit="US AQI" label={`Hourly US AQI in ${place}`}
              bands={AQI_BANDS.map((b, i) => ({ max: b.max === Infinity ? 500 : b.max, color: BAND_COLORS[i] }))} />
          </Section>
          <Section title="Daily summary">
            <table className="data"><thead><tr><th>Day</th><th className="num">Highest AQI</th><th>Category</th><th className="num">Mean PM2.5 (µg/m³)</th></tr></thead>
              <tbody>{days.map((x) => <tr key={x.day}><td>{nptDate(`${x.day}T12:00:00+05:45`, { weekday: "short" })}</td><td className="num">{fmt(x.max)}</td><td>{aqiBand(x.max).label}</td><td className="num">{fmt(x.pm, 0)}</td></tr>)}</tbody>
            </table>
          </Section>
          <Section title="All pollutants now">
            <table className="data"><tbody>
              <tr><td>Fine particles PM2.5</td><td className="num">{fmt(c.pm2_5, 1)} µg/m³</td></tr>
              <tr><td>Coarse particles PM10</td><td className="num">{fmt(c.pm10, 1)} µg/m³</td></tr>
              <tr><td>Nitrogen dioxide NO₂</td><td className="num">{fmt(c.nitrogen_dioxide, 1)} µg/m³</td></tr>
              <tr><td>Ozone O₃</td><td className="num">{fmt(c.ozone, 1)} µg/m³</td></tr>
              <tr><td>Sulphur dioxide SO₂</td><td className="num">{fmt(c.sulphur_dioxide, 1)} µg/m³</td></tr>
              <tr><td>Carbon monoxide CO</td><td className="num">{fmt(c.carbon_monoxide, 0)} µg/m³</td></tr>
            </tbody></table>
          </Section>
        </>
      ) : <p className="mb-10 text-muted">The air-quality service is not responding right now; please check again in a few minutes.</p>}
      {ranked.length > 0 && (
        <Section title="Most and least polluted districts right now">
          <div className="grid gap-6 sm:grid-cols-2">
            {[ranked.slice(0, 5), ranked.slice(-5).reverse()].map((list, j) => (
              <ol key={j} className="space-y-1">{list.map((r) => <li key={r.x.code}><Link href={`/air-quality/${r.x.slug}`}>{placeLabel(r.x)}</Link> — AQI {fmt(r.v)}</li>)}</ol>
            ))}
          </div>
        </Section>
      )}
      {faq.length > 0 && (
        <Section title={`Questions about air quality in ${place}`}>
          <dl className="space-y-4">{faq.map((f) => <div key={f.q}><dt className="font-medium">{f.q}</dt><dd className="mt-1 text-muted">{f.a}</dd></div>)}</dl>
          <JsonLd data={{ "@context": "https://schema.org", "@type": "FAQPage", mainEntity: faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })) }} />
        </Section>
      )}
      <p className="mb-6"><Link href={`/weather/${d.slug}`}>Weather in {place}</Link> · <Link href={`/districts/${d.slug}`}>{areaTitle(d)}: land cover and map</Link></p>
      <Source>Copernicus Atmosphere Monitoring Service (CAMS) via Open-Meteo. Modelled values at the town, not a ground monitor.</Source>
    </>
  );
}
