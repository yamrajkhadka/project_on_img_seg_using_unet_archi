import type { Metadata } from "next";
import Link from "next/link";
import { ClassTrends, ShareBar } from "@/components/charts";
import { PageHeader, Section, Source, TableTwin } from "@/components/ui";
import { CLASS_KEYS, CLASS_NAME, childrenOf, landCoverOf, landcover, regionsAt } from "@/lib/data";
import { fmt, km2 } from "@/lib/format";
import { describeShare, latestShare } from "@/lib/insights";
import { siteConfig } from "@/lib/site";

export const metadata: Metadata = {
  title: "Land Cover of Nepal: Forest, Farmland & Built-up Area Mapped by AI",
  description: "How much of Nepal is forest, farmland, rangeland, built-up land, water and bare land, year by year since 2016, mapped by a U-Net AI model from satellite imagery for every province and district.",
  alternates: { canonical: "/land-cover" },
};

export default function Page() {
  const lc = landCoverOf("NP")!;
  const share = latestShare("NP")!;
  return (
    <>
      <PageHeader eyebrow="Land cover" title="Land cover of Nepal"
        intro={<p>{describeShare("Nepal", share)} These figures come from a U-Net deep-learning model that labels every pixel of satellite imagery, published by <a href={siteConfig.paper.url}>Khadka &amp; Pradhan (2026)</a> and adapted to Nepal.</p>} />
      <Section title="Nepal today" note={<>{lc.imagery}; {lc.n} stratified random sample sites; 95% intervals in the table.</>}>
        <ShareBar share={share} label="Share of Nepal's area by land-cover class" />
        <TableTwin caption="Land cover of Nepal" head={["Class", "Share (%)", "95% interval", "Area"]}
          rows={CLASS_KEYS.map((k) => [CLASS_NAME[k], fmt(lc.latest[k][0], 1), `${fmt(lc.latest[k][1], 1)}–${fmt(lc.latest[k][2], 1)}`, km2(lc.latest[k][3])])} />
      </Section>
      <Section title="Year by year" note="Coloured: the U-Net on each year's Sentinel-2 cloudless mosaic with 95% intervals; grey: the independent 10 m Impact Observatory map. Year-to-year differences of a few points can come from mosaic processing.">
        <ClassTrends trend={lc.trend} reference={lc.reference} />
        <p className="mt-4">Each year in detail: {landcover.years.map((y, i) => <span key={y}>{i ? " · " : ""}<Link href={`/land-cover/${y}`}>{y}</Link></span>)}</p>
      </Section>
      <Section title="By province and district">
        {regionsAt(1).map((p) => (
          <details key={p.code} className="mb-2">
            <summary className="cursor-pointer"><Link href={`/provinces/${p.slug}`}>{p.name} Province</Link> — forest {fmt(latestShare(p.code)?.forest, 1)}%, built-up {fmt(latestShare(p.code)?.urban, 1)}%</summary>
            <ul className="mt-2 grid gap-x-6 sm:grid-cols-2 lg:grid-cols-3">
              {childrenOf(p.code).map((d) => <li key={d.code}><Link href={`/districts/${d.slug}`}>{d.name}</Link> <span className="text-sm text-muted">forest {fmt(latestShare(d.code)?.forest, 0)}%</span></li>)}
            </ul>
          </details>
        ))}
      </Section>
      <Section title="The seven classes">
        <dl className="space-y-2">{landcover.classes.map((c) => <div key={c.key}><dt className="font-medium" style={{ color: `var(--c-${c.key})` }}>{c.name}</dt><dd className="text-muted">{c.definition}</dd></div>)}</dl>
      </Section>
      <Source>Nepal-adapted U-Net (Khadka &amp; Pradhan 2026) on Esri World Imagery and EOX Sentinel-2 cloudless; reference: Impact Observatory/Esri 10 m land cover.</Source>
    </>
  );
}
