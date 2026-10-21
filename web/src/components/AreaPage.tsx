/** The page of a province or district: map, AI land cover with its trend and the reference map, sub-regions, FAQ. */
import Link from "next/link";
import { ClassTrends, ShareBar } from "@/components/charts";
import { RegionMap } from "@/components/RegionMap";
import { Breadcrumbs, Card, JsonLd, PageHeader, Section, Source, Stat, TableTwin } from "@/components/ui";
import {
  ancestors, CLASS_KEYS, CLASS_NAME, childrenOf, LEVEL_NAME, landCoverOf, landcover, localLandCover, placeLabel,
  type Region, regionByCode, regionHref, townOf,
} from "@/lib/data";
import { fmt, km2, pct } from "@/lib/format";
import { changes, describeShare, latestShare, ordinal, rankOf, signed } from "@/lib/insights";
import { siteConfig } from "@/lib/site";

export function areaTitle(r: Region): string {
  return r.level === 1 ? `${r.name} Province` : `${r.name} District`;
}

export function areaDescription(r: Region): string {
  const s = latestShare(r.code);
  const where = r.level === 2 ? `${r.name} district (${r.province} Province)` : `${r.name} Province`;
  return s
    ? `Land cover of ${where}, Nepal: forest ${fmt(s.forest, 1)}%, farmland ${fmt(s.agriculture, 1)}%, built-up ${fmt(s.urban, 1)}%. Map, year-by-year trend since 2016, ${r.level === 1 ? "districts" : "municipalities"}, air quality and weather.`
    : `Map, land cover, ${r.level === 1 ? "districts" : "municipalities"}, air quality and weather of ${where}, Nepal.`;
}

export function AreaPage({ r }: { r: Region }) {
  const lc = landCoverOf(r.code);
  const share = latestShare(r.code);
  const national = latestShare("NP");
  const kids = childrenOf(r.code);
  const up = ancestors(r);
  const title = areaTitle(r);
  const town = r.level === 2 ? townOf(r.code) : undefined;
  const refYears = lc ? Object.keys(lc.reference).sort() : [];
  const ref0 = refYears.length ? lc!.reference[refYears[0]] : undefined;
  const ref1 = refYears.length ? lc!.reference[refYears[refYears.length - 1]] : undefined;
  const forestRank = rankOf(r, "forest");
  const urbanRank = rankOf(r, "urban");
  const peers = r.level === 1 ? "provinces" : "districts";
  const kidRows = kids.map((k) => {
    const loc = r.level === 2 ? localLandCover(k.code).at(-1)?.share : latestShare(k.code);
    return { k, s: loc };
  });

  const faq = share ? [
    { q: `What is the forest cover of ${title}?`, a: `About ${fmt(share.forest, 1)}% of ${title} is forest (95% interval ${fmt(lc!.latest.forest[1], 1)}–${fmt(lc!.latest.forest[2], 1)}%), roughly ${km2(lc!.latest.forest[3])}, according to the Nepal-adapted U-Net applied to ${lc!.n} sample sites.${forestRank ? ` That ranks ${ordinal(forestRank.rank)} of ${forestRank.of} ${peers}.` : ""}` },
    { q: `How much of ${title} is built-up or urban?`, a: `Built-up land covers about ${fmt(share.urban, 1)}% (${km2(lc!.latest.urban[3])}).${urbanRank ? ` ${title} ranks ${ordinal(urbanRank.rank)} of ${urbanRank.of} ${peers} for built-up share.` : ""}` },
    { q: `How big is ${title}?`, a: `${title} covers ${km2(r.area)} and has ${kids.length} ${r.level === 1 ? "districts" : "local units (municipalities and rural municipalities)"}.` },
  ] : [];

  return (
    <>
      <Breadcrumbs items={[...up.map((a) => ({ href: regionHref(a), label: areaTitle(a) })),
        { href: r.level === 1 ? "/provinces" : "/districts", label: r.level === 1 ? "Provinces" : "Districts" }, { label: title }]} />
      <PageHeader eyebrow={`${LEVEL_NAME[r.level]} of Nepal${r.province && r.level === 2 ? ` · ${r.province} Province` : ""}`}
        title={`${title}: land cover, map and ${r.level === 1 ? "districts" : "municipalities"}`}
        intro={<p>{share ? describeShare(title, share, national) : `${title} covers ${km2(r.area)}.`}</p>} />

      <div className="mb-10 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <RegionMap label={`Map of ${title}`} highlight={{ level: r.level, code: r.code }}
          contextLevel={r.level === 1 ? 1 : 2}
          context={r.level === 1 ? undefined : childrenOf(r.parent ?? "").map((d) => d.code)}
          frame={r.level === 2 ? regionByCode(r.parent)?.bbox : undefined}
          points={town ? [{ lon: town.lon, lat: town.lat, r: 4, color: "var(--fg)", label: town.town }] : []} />
        <div className="grid content-start gap-3 sm:grid-cols-2 lg:grid-cols-1">
          <Stat label="Area" value={km2(r.area)} sub={`${kids.length} ${r.level === 1 ? "districts" : "local units"}`} />
          {share && <Stat label="Forest" value={pct(share.forest)} sub={forestRank ? `${ordinal(forestRank.rank)} of ${forestRank.of} ${peers}` : undefined} />}
          {share && <Stat label="Built-up land" value={pct(share.urban)} sub={urbanRank ? `${ordinal(urbanRank.rank)} of ${urbanRank.of} ${peers}` : undefined} />}
          {r.level === 2 && (
            <Card>
              <div className="text-sm text-muted">Live conditions{town ? ` in ${placeLabel(r)}` : ""}</div>
              <div className="mt-1 flex gap-4"><Link href={`/air-quality/${r.slug}`}>Air quality today</Link><Link href={`/weather/${r.slug}`}>Weather forecast</Link></div>
            </Card>
          )}
        </div>
      </div>

      {lc && share && (
        <Section title={`Land cover of ${title}`} note={<>{lc.imagery}. Estimated from {lc.n} randomly placed 1.2 km sample windows classified by the Nepal-adapted U-Net; intervals are 95% confidence. <Link href="/research">How it works</Link>.</>}>
          <ShareBar share={share} label={`Share of ${title} by land-cover class`} />
          <TableTwin caption={`Land cover of ${title}`} head={["Class", "Share (%)", "95% interval", "Area (km²)"]}
            rows={CLASS_KEYS.map((k) => [CLASS_NAME[k], fmt(lc.latest[k][0], 1), `${fmt(lc.latest[k][1], 1)}–${fmt(lc.latest[k][2], 1)}`, fmt(lc.latest[k][3])])} />
        </Section>
      )}

      {lc && Object.keys(lc.trend).length > 1 && (
        <Section title={`${title} year by year`} note={<>Coloured lines: the U-Net on each year&apos;s Sentinel-2 cloudless mosaic, with 95% intervals. Grey: the independent 10 m Impact Observatory map at the same sites. Differences of a few points between years can come from how each year&apos;s mosaic was processed.</>}>
          <ClassTrends trend={lc.trend} reference={lc.reference} />
          {ref0 && ref1 && (
            <p className="mt-4 text-sm">
              Largest changes in the reference map from {refYears[0]} to {refYears[refYears.length - 1]}:{" "}
              {changes(ref0, ref1).slice(0, 3).map((c) => `${CLASS_NAME[c.k].toLowerCase()} ${signed(c.d)} points`).join(", ")}.
            </p>
          )}
        </Section>
      )}

      <Section title={r.level === 1 ? `Districts of ${title}` : `Municipalities and rural municipalities of ${title}`}
        note={r.level === 2 ? `Local-unit land cover is from the 10 m Impact Observatory map (${landcover.reference_years.at(-1)}), clipped to each boundary.` : undefined}>
        <div className="overflow-x-auto">
          <table className="data">
            <thead><tr><th>{r.level === 1 ? "District" : "Local unit"}</th><th className="num">Area (km²)</th><th className="num">Forest %</th><th className="num">Farmland %</th><th className="num">Built-up %</th></tr></thead>
            <tbody>
              {kidRows.map(({ k, s }) => (
                <tr key={k.code}>
                  <td><Link href={regionHref(k)}>{k.name}</Link></td>
                  <td className="num">{fmt(k.area)}</td>
                  <td className="num">{fmt(s?.forest, 1)}</td>
                  <td className="num">{fmt(s?.agriculture, 1)}</td>
                  <td className="num">{fmt(s?.urban, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {faq.length > 0 && (
        <Section title={`Questions about ${title}`}>
          <dl className="space-y-4">
            {faq.map((f) => <div key={f.q}><dt className="font-medium">{f.q}</dt><dd className="mt-1 text-muted">{f.a}</dd></div>)}
          </dl>
          <JsonLd data={{ "@context": "https://schema.org", "@type": "FAQPage",
            mainEntity: faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })) }} />
        </Section>
      )}
      <JsonLd data={{ "@context": "https://schema.org", "@type": "Place", name: title, url: `${siteConfig.url}${regionHref(r)}`,
        containedInPlace: up.length ? { "@type": "AdministrativeArea", name: areaTitle(up[up.length - 1]) } : { "@type": "Country", name: "Nepal" },
        geo: { "@type": "GeoCoordinates", latitude: r.lat, longitude: r.lon } }} />
      <Source>Nepal-adapted U-Net (Khadka &amp; Pradhan 2026) on Esri and EOX Sentinel-2 cloudless imagery; Impact Observatory/Esri 10 m land cover; boundaries: Survey Department of Nepal via OCHA.</Source>
    </>
  );
}
