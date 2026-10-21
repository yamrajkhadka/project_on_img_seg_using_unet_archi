import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { BarChart, HourlyChart } from "@/components/charts";
import { Breadcrumbs, JsonLd, PageHeader, Section, Source, Stat } from "@/components/ui";
import { rivers } from "@/lib/data";
import { dayOfYear, fmt, nptDate, todayNpt } from "@/lib/format";
import { riverFlow } from "@/lib/live";

export const revalidate = 3600;
export const dynamicParams = false;

export function generateStaticParams() {
  return rivers.rivers.map((r) => ({ slug: r.key }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const r = rivers.rivers.find((x) => x.key === slug);
  if (!r) return {};
  const title = `${r.name} River Level Today at ${r.gauge}: Flow, Flood Forecast & History`;
  const description = `Today's flow of the ${r.name} River at ${r.gauge}, Nepal, compared with normal for the date and flood levels, with the 7-day forecast and every year's peak flood since 2004.`;
  return { title, description, alternates: { canonical: `/rivers/${r.key}` }, openGraph: { title, description } };
}

export default async function Page({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const r = rivers.rivers.find((x) => x.key === slug);
  if (!r) notFound();
  const flow = await riverFlow(r);
  const thr = rivers.thresholds[r.key];
  const clim = rivers.climatology[r.key] ?? [];
  const annual = rivers.annual[r.key] ?? [];
  const today = todayNpt();
  const idx = flow ? flow.daily.time.indexOf(today) : -1;
  const q = idx >= 0 ? flow!.daily.river_discharge[idx] : null;
  const normal = clim.find((c) => c[0] === dayOfYear(today))?.[2];
  const fc = flow ? flow.daily.river_discharge.slice(idx >= 0 ? idx : 0) : [];
  const fcMax = fc.length ? Math.max(...fc) : null;
  const status = q == null || !thr ? "No data" : q >= thr.q10 ? "Above the 10-year flood level" : q >= thr.q2 ? "Above the 2-year flood level" : q >= (normal ?? Infinity) * 1.5 ? "High for the season" : "Normal range";
  const biggest = [...annual].sort((a, b) => b[1] - a[1]).slice(0, 3);
  const faq = [
    { q: `What is the ${r.name} River level today?`, a: q != null ? `The ${r.name} at ${r.gauge} is flowing at about ${fmt(q)} m³/s today${normal ? `, ${fmt((q / normal) * 100)}% of the 2004–2023 median for the date` : ""} (${status.toLowerCase()}).` : "Today's value is not available right now." },
    { q: `What is a flood on the ${r.name}?`, a: `Here a flood means a daily flow above the 2-year level (${fmt(thr?.q2)} m³/s, the median annual peak 2004–2023); above ${fmt(thr?.q10)} m³/s is a 10-year flood.` },
    { q: `What were the biggest ${r.name} floods?`, a: biggest.length ? `The largest annual peaks since 2004: ${biggest.map((b) => `${fmt(b[1])} m³/s on ${nptDate(b[3])}`).join("; ")}.` : "No record available." },
  ];
  return (
    <>
      <Breadcrumbs items={[{ href: "/rivers", label: "Rivers" }, { label: r.name }]} />
      <PageHeader eyebrow={`Live · ${nptDate(new Date())} · ${r.province}`} title={`${r.name} River at ${r.gauge}`}
        intro={<p>Daily river discharge from the GloFAS global flood model at {r.gauge}, compared with the 2004–2023 record for the same date, with the 7-day forecast.</p>} />
      <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Flow today" value={q != null ? `${fmt(q)} m³/s` : "—"} sub={status} tone={status.startsWith("Above the 10") ? "bad" : status.startsWith("Above") ? "warn" : undefined} />
        <Stat label="Normal for the date" value={normal ? `${fmt(normal)} m³/s` : "—"} sub="2004–2023 median" />
        <Stat label="Highest in the next 7 days" value={fcMax != null ? `${fmt(fcMax)} m³/s` : "—"} sub="forecast" />
        <Stat label="2-year / 10-year flood level" value={`${fmt(thr?.q2)} / ${fmt(thr?.q10)}`} sub="m³/s" />
      </div>
      {flow && (
        <Section title="Last 30 days and the next 7" note="m³/s; dashed line forecast.">
          <HourlyChart times={flow.daily.time.map((t) => `${t}T00:00`)} values={flow.daily.river_discharge} now={`${today}T00:00`} unit="m³/s" label={`Daily discharge of the ${r.name}`} />
        </Section>
      )}
      <Section title="Peak flow every year" note="Highest daily discharge of each year; dashed lines: 2-year and 10-year flood levels.">
        <BarChart data={annual.map((a) => ({ x: String(a[0]), v: a[1] }))} unit="m³/s" label={`Annual peak flow of the ${r.name}`}
          lines={thr ? [{ v: thr.q2, label: "2-year flood" }, { v: thr.q10, label: "10-year flood" }] : []} />
      </Section>
      <Section title={`Questions about the ${r.name}`}>
        <dl className="space-y-4">{faq.map((f) => <div key={f.q}><dt className="font-medium">{f.q}</dt><dd className="mt-1 text-muted">{f.a}</dd></div>)}</dl>
        <JsonLd data={{ "@context": "https://schema.org", "@type": "FAQPage", mainEntity: faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })) }} />
      </Section>
      <p className="mb-6">Other rivers: {rivers.rivers.filter((x) => x.key !== r.key).map((x, i) => <span key={x.key}>{i ? " · " : ""}<Link href={`/rivers/${x.key}`}>{x.name}</Link></span>)}</p>
      <Source>GloFAS v4 (Copernicus Emergency Management Service / ECMWF) via Open-Meteo. Modelled flows, not gauge readings; official flood warnings: Department of Hydrology and Meteorology (hydrology.gov.np).</Source>
    </>
  );
}
