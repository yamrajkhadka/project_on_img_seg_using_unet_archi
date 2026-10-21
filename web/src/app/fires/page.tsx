import type { Metadata } from "next";
import Link from "next/link";
import { BarChart } from "@/components/charts";
import { RegionMap } from "@/components/RegionMap";
import { JsonLd, PageHeader, Section, Source, Stat, TableTwin } from "@/components/ui";
import { hazards, regionByCode, regionsAt } from "@/lib/data";
import { fmt, nptDate } from "@/lib/format";
import { fires } from "@/lib/live";
import { regionAt } from "@/lib/shapes";

export const revalidate = 1800;

export const metadata: Metadata = {
  title: "Forest Fires in Nepal Today: Live Fire Map by District",
  description: "Active forest fires in Nepal detected by NASA satellites in the last 24 hours and 7 days, mapped and counted by district and province, updated every 30 minutes.",
  alternates: { canonical: "/fires" },
};

export default async function Page() {
  const [day, week] = await Promise.all([fires("24h"), fires("7d")]);
  const byDistrict = new Map<string, number>();
  for (const f of week ?? []) {
    const code = regionAt(2, f.lon, f.lat);
    if (code) byDistrict.set(code, (byDistrict.get(code) ?? 0) + 1);
  }
  const top = [...byDistrict.entries()].sort((a, b) => b[1] - a[1]).map(([code, n]) => ({ d: regionByCode(code)!, n }));
  const viirs = hazards.fires["VIIRS S-NPP"] ?? {};
  const years = Object.keys(viirs).sort();
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const monthly = MONTHS.map((m, i) => ({ x: m, v: Math.round(years.reduce((s, y) => s + viirs[y].months[i], 0) / Math.max(1, years.length)) }));
  const allDistricts = new Map<string, number>();
  for (const y of years) for (const [d, n] of Object.entries(viirs[y].districts)) allDistricts.set(d, (allDistricts.get(d) ?? 0) + n);
  const worst = [...allDistricts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
  const worstYears = [...years].sort((a, b) => viirs[b].total - viirs[a].total).slice(0, 3);
  const faq = [
    { q: "Are there forest fires in Nepal right now?", a: day ? `NASA satellites detected ${day.length} active fire${day.length === 1 ? "" : "s"} in Nepal in the last 24 hours and ${week?.length ?? 0} in the last 7 days${top[0] ? `, most of them in ${top[0].d.name} district (${top[0].n})` : ""}.` : "The NASA fire service is not responding right now." },
    { q: "When is the forest fire season in Nepal?", a: "Most fires burn in the dry season from February to May, peaking in March and April, before the monsoon. Fires are mostly human-caused: clearing land, burning crop residue and grazing land, and accidents." },
    { q: "How are fires detected?", a: "The VIIRS instruments on the Suomi NPP, NOAA-20 and NOAA-21 satellites detect heat from fires about 375 m across, each passing over Nepal around 1:30 pm and 1:30 am. Thick smoke, cloud or small fires can be missed." },
  ];
  return (
    <>
      <PageHeader eyebrow={`Live · ${nptDate(new Date())}`} title="Forest fires in Nepal"
        intro={<p>Satellite fire detections inside Nepal from NASA&apos;s VIIRS sensors, refreshed every 30 minutes, counted by district.</p>} />
      <div className="mb-8 grid gap-4 sm:grid-cols-3">
        <Stat label="Last 24 hours" value={day ? fmt(day.length) : "—"} sub="fire detections" />
        <Stat label="Last 7 days" value={week ? fmt(week.length) : "—"} sub="fire detections" />
        <Stat label="Most affected district (7 days)" value={top[0]?.d.name ?? "—"} sub={top[0] ? `${top[0].n} detections` : undefined} />
      </div>
      <RegionMap label="Active fires in Nepal in the last 7 days" contextLevel={2} context={regionsAt(2).map((d) => d.code)}
        points={(week ?? []).map((f) => ({ lon: f.lon, lat: f.lat, r: 2.2, color: "var(--crimson)" }))} />
      {top.length > 0 && (
        <Section title="Fire detections by district, last 7 days">
          <table className="data"><thead><tr><th>District</th><th>Province</th><th className="num">Detections</th></tr></thead>
            <tbody>{top.map(({ d, n }) => <tr key={d.code}><td><Link href={`/districts/${d.slug}`}>{d.name}</Link></td><td>{d.province}</td><td className="num">{n}</td></tr>)}</tbody>
          </table>
        </Section>
      )}
      {years.length > 0 && (
        <>
          <Section title="Fires every year since 2012" note={`VIIRS S-NPP detections inside Nepal per year (NASA FIRMS archive). The worst years were ${worstYears.join(", ")}.`}>
            <BarChart data={years.map((y) => ({ x: y, v: viirs[y].total }))} unit="detections" label="Fire detections in Nepal per year" />
            <TableTwin caption="Fire detections per year" head={["Year", "Detections"]} rows={[...years].reverse().map((y) => [y, viirs[y].total.toLocaleString("en-US")])} />
          </Section>
          <Section title="The fire season" note={`Average detections per month, ${years[0]}–${years[years.length - 1]}: almost all fires burn from February to May.`}>
            <BarChart data={monthly} unit="detections per month" label="Average fire detections per month" />
          </Section>
          <Section title={`Districts with the most fires, ${years[0]}–${years[years.length - 1]}`}>
            <table className="data"><thead><tr><th>District</th><th className="num">Detections</th><th className="num">Per year</th></tr></thead>
              <tbody>{worst.map(([d, n]) => {
                const r = regionsAt(2).find((x) => x.name === d);
                return <tr key={d}><td>{r ? <Link href={`/districts/${r.slug}`}>{d}</Link> : d}</td><td className="num">{n.toLocaleString("en-US")}</td><td className="num">{Math.round(n / years.length).toLocaleString("en-US")}</td></tr>;
              })}</tbody>
            </table>
          </Section>
        </>
      )}
      <Section title="Questions about fires in Nepal">
        <dl className="space-y-4">{faq.map((f) => <div key={f.q}><dt className="font-medium">{f.q}</dt><dd className="mt-1 text-muted">{f.a}</dd></div>)}</dl>
        <JsonLd data={{ "@context": "https://schema.org", "@type": "FAQPage", mainEntity: faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })) }} />
      </Section>
      <Source>NASA FIRMS, VIIRS 375 m active fire products (Suomi NPP, NOAA-20, NOAA-21).</Source>
    </>
  );
}
