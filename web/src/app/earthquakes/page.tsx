import type { Metadata } from "next";
import { BarChart } from "@/components/charts";
import { RegionMap } from "@/components/RegionMap";
import { JsonLd, PageHeader, Section, Source, Stat } from "@/components/ui";
import { hazards, regionsAt } from "@/lib/data";
import { ago, fmt, nowMs, nptDate, nptTime } from "@/lib/format";
import { quakes } from "@/lib/live";

export const revalidate = 600;

export const metadata: Metadata = {
  title: "Earthquake in Nepal Today: Latest Earthquakes, Live Map & History",
  description: "Latest earthquakes in and around Nepal, updated every 10 minutes from USGS: magnitude, location, depth and time in Nepal Standard Time, with a live map and the strongest earthquakes since 1900.",
  alternates: { canonical: "/earthquakes" },
};

export default async function Page() {
  const [recent, big] = await Promise.all([quakes(30, 2.5), quakes(365 * 126, 6.0, 86400)]);
  const latest = recent?.[0];
  const week = recent?.filter((q) => q.time >= nowMs() - 7 * 86400_000) ?? [];
  const strongest = recent?.length ? recent.reduce((a, b) => (b.mag > a.mag ? b : a)) : undefined;
  const faq = [
    { q: "Was there an earthquake in Nepal today?", a: latest ? `The most recent earthquake of magnitude 2.5 or more in and around Nepal was M${latest.mag.toFixed(1)}, ${latest.place}, on ${nptDate(latest.time)} at ${nptTime(latest.time)} Nepal time (${ago(latest.time)}).` : "No earthquake of magnitude 2.5 or more has been reported in the region in the last 30 days." },
    { q: "Why does Nepal have so many earthquakes?", a: "Nepal lies on the boundary where the Indian plate pushes under the Eurasian plate at about 4–5 cm a year. The strain is released in earthquakes along the Main Himalayan Thrust, such as the M7.8 Gorkha earthquake of 25 April 2015." },
    { q: "What is the biggest earthquake in Nepal's history?", a: "The 1934 Nepal–Bihar earthquake (about M8.0) and the 2015 Gorkha earthquake (M7.8) are the largest of the instrumental era; the earthquakes of 1255 and 1505 are thought to have been comparable or larger." },
  ];
  return (
    <>
      <PageHeader eyebrow="Live · updated every 10 minutes" title="Earthquakes in Nepal"
        intro={<p>Every earthquake of magnitude 2.5 or more in and around Nepal in the last 30 days, from the US Geological Survey, with times in Nepal Standard Time, and the strongest earthquakes in the region since 1900.</p>} />
      <div className="mb-8 grid gap-4 sm:grid-cols-3">
        <Stat label="Latest" value={latest ? `M${latest.mag.toFixed(1)}` : "—"} sub={latest ? `${ago(latest.time)} · ${latest.place}` : "None in 30 days"} />
        <Stat label="Last 7 days (M2.5+)" value={fmt(week.length)} />
        <Stat label="Strongest in 30 days" value={strongest ? `M${strongest.mag.toFixed(1)}` : "—"} sub={strongest ? `${nptDate(strongest.time)} · ${strongest.place}` : undefined} />
      </div>
      <RegionMap label="Earthquakes around Nepal in the last 30 days" contextLevel={1} context={regionsAt(1).map((p) => p.code)}
        points={(recent ?? []).map((q) => ({ lon: q.lon, lat: q.lat, r: 1.5 + q.mag * 1.4, color: "#6250d6" }))} />
      <Section title="Earthquakes in the last 30 days">
        {recent && recent.length ? (
          <div className="overflow-x-auto"><table className="data">
            <thead><tr><th>Time (NPT)</th><th className="num">Magnitude</th><th>Location</th><th className="num">Depth (km)</th></tr></thead>
            <tbody>{recent.map((q) => <tr key={q.id}><td>{nptDate(q.time)} {nptTime(q.time)}</td><td className="num">M{q.mag.toFixed(1)}</td><td><a href={q.url}>{q.place}</a></td><td className="num">{fmt(q.depth)}</td></tr>)}</tbody>
          </table></div>
        ) : <p className="text-muted">{recent ? "No earthquakes of M2.5 or more in the last 30 days." : "The USGS service is not responding right now."}</p>}
      </Section>
      {big && big.length > 0 && (
        <Section title="Strongest earthquakes in the region since 1900 (M6+)">
          <div className="overflow-x-auto"><table className="data">
            <thead><tr><th>Date</th><th className="num">Magnitude</th><th>Location</th><th className="num">Depth (km)</th></tr></thead>
            <tbody>{[...big].sort((a, b) => b.mag - a.mag).slice(0, 40).map((q) => <tr key={q.id}><td>{nptDate(q.time)}</td><td className="num">M{q.mag.toFixed(1)}</td><td><a href={q.url}>{q.place}</a></td><td className="num">{fmt(q.depth)}</td></tr>)}</tbody>
          </table></div>
        </Section>
      )}
      {Object.keys(hazards.quakes).length > 0 && (
        <Section title="Earthquakes per year since 1950" note="Magnitude 4 and above in the search area (USGS). Earlier decades miss many smaller events; the 2015 spike is the Gorkha earthquake and its aftershocks.">
          <BarChart data={Object.entries(hazards.quakes).map(([y, c]) => ({ x: y, v: c.m4 }))} unit="earthquakes M4+" label="Earthquakes of magnitude 4 or more per year around Nepal" />
        </Section>
      )}
      <Section title="Questions about earthquakes in Nepal">
        <dl className="space-y-4">{faq.map((f) => <div key={f.q}><dt className="font-medium">{f.q}</dt><dd className="mt-1 text-muted">{f.a}</dd></div>)}</dl>
        <JsonLd data={{ "@context": "https://schema.org", "@type": "FAQPage", mainEntity: faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })) }} />
      </Section>
      <Source>USGS Earthquake Hazards Program (ComCat), search area 25.4–31.3° N, 79.2–89.2° E. Felt reports and official bulletins: National Earthquake Monitoring and Research Center (seismonepal.gov.np).</Source>
    </>
  );
}
