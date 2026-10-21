import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { areaTitle } from "@/components/AreaPage";
import { ShareBar } from "@/components/charts";
import { RegionMap } from "@/components/RegionMap";
import { Breadcrumbs, Card, JsonLd, PageHeader, Section, Source, Stat, TableTwin } from "@/components/ui";
import {
  CLASS_KEYS, CLASS_NAME, childrenOf, localLandCover, placeLabel, regionByCode, regionBySlug, regionHref, regionsAt, unitKind,
} from "@/lib/data";
import { fmt, km2, pct } from "@/lib/format";
import { changes, compare, describeShare, districtReferenceShare, ordinal, rankOf, signed } from "@/lib/insights";
import { siteConfig } from "@/lib/site";

export const dynamicParams = false;

export function generateStaticParams() {
  return regionsAt(3).map((r) => ({ slug: r.slug }));
}

function describe(slug: string) {
  const r = regionBySlug(3, slug);
  if (!r) return undefined;
  const district = regionByCode(r.parent)!;
  const years = localLandCover(r.code);
  return { r, district, years, first: years[0], last: years.at(-1) };
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const d = describe((await params).slug);
  if (!d) return {};
  const s = d.last?.share;
  const title = `${d.r.name}, ${d.district.name}: Land Cover, Forest & Map`;
  const description = s
    ? `${d.r.name} (${d.district.name} district, ${d.r.province} Province, Nepal) covers ${km2(d.r.area)}: forest ${fmt(s.forest, 1)}%, farmland ${fmt(s.agriculture, 1)}%, built-up ${fmt(s.urban, 1)}% in ${d.last!.year}, and how it changed since ${d.first!.year}.`
    : `${d.r.name} in ${d.district.name} district, ${d.r.province} Province, Nepal: area, map and land cover.`;
  return { title, description, alternates: { canonical: `/municipalities/${d.r.slug}` }, openGraph: { title, description } };
}

export default async function Page({ params }: { params: Promise<{ slug: string }> }) {
  const d = describe((await params).slug);
  if (!d) notFound();
  const { r, district, first, last } = d;
  const province = regionByCode(district.parent)!;
  const kind = unitKind(r);
  const districtShare = districtReferenceShare(district.code); // same 10 m map as the unit itself
  const forestRank = rankOf(r, "forest");
  const urbanRank = rankOf(r, "urban");
  const ch = first && last && first.year !== last.year ? changes(first.share, last.share) : [];
  const siblings = childrenOf(district.code);

  const faq = last ? [
    { q: `What is the forest cover of ${r.name}?`, a: `Forest covers about ${fmt(last.share.forest, 1)}% of ${r.name} (${km2((last.share.forest / 100) * r.area)}) in the ${last.year} 10 m land-cover map${districtShare ? `, ${compare(last.share.forest, districtShare.forest)} the ${district.name} district average of ${fmt(districtShare.forest, 1)}%` : ""}.` },
    { q: `How much of ${r.name} is built-up?`, a: `Built-up land covers ${fmt(last.share.urban, 1)}% (${km2((last.share.urban / 100) * r.area)})${first && first.year !== last.year ? `, ${signed(last.share.urban - first.share.urban)} percentage points since ${first.year}` : ""}.${urbanRank ? ` ${r.name} ranks ${ordinal(urbanRank.rank)} of ${urbanRank.of} local units in Nepal for built-up share.` : ""}` },
    { q: `Where is ${r.name}?`, a: `${r.name} is ${kind === "Local unit" ? "a local unit (municipality or rural municipality)" : `a ${kind.toLowerCase()}`} in ${district.name} district, ${province.name} Province, Nepal, covering ${km2(r.area)}.` },
  ] : [];

  return (
    <>
      <Breadcrumbs items={[{ href: regionHref(province), label: areaTitle(province) }, { href: regionHref(district), label: areaTitle(district) }, { label: r.name }]} />
      <PageHeader eyebrow={`${kind} · ${district.name} district · ${province.name} Province`} title={`${r.name}: land cover and map`}
        intro={<p>{last ? describeShare(r.name, last.share, districtShare, `the ${district.name} district average of`) : `${r.name} covers ${km2(r.area)}.`}</p>} />

      <div className="mb-10 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <RegionMap label={`Map of ${r.name} in ${district.name} district`} highlight={{ level: 3, code: r.code }}
          contextLevel={3} context={siblings.map((s) => s.code)} frame={district.bbox} />
        <div className="grid content-start gap-3 sm:grid-cols-2 lg:grid-cols-1">
          <Stat label="Area" value={km2(r.area)} sub={`${kind}, ${district.name} district`} />
          {last && <Stat label={`Forest (${last.year})`} value={pct(last.share.forest)} sub={forestRank ? `${ordinal(forestRank.rank)} of ${forestRank.of} local units` : undefined} />}
          {last && <Stat label={`Built-up (${last.year})`} value={pct(last.share.urban)} sub={first && first.year !== last.year ? `${signed(last.share.urban - first.share.urban)} points since ${first.year}` : undefined} />}
          <Card>
            <div className="text-sm text-muted">Live conditions nearby</div>
            <div className="mt-1 flex gap-4"><Link href={`/air-quality/${district.slug}`}>Air quality in {placeLabel(district)}</Link></div>
            <div className="mt-1"><Link href={`/weather/${district.slug}`}>Weather forecast</Link></div>
          </Card>
        </div>
      </div>

      {last && (
        <Section title={`Land cover of ${r.name} in ${last.year}`} note={`Impact Observatory / Esri 10 m annual land-cover map, translated to the paper's seven classes and clipped to the boundary of ${r.name}${last.res_m > 10 ? ` (summarised at ${fmt(last.res_m)} m)` : ""}.`}>
          <ShareBar share={last.share} label={`Share of ${r.name} by land-cover class, ${last.year}`} />
          <TableTwin caption={`Land cover of ${r.name}`} head={["Class", `${first?.year ?? ""} (%)`, `${last.year} (%)`, "Change (points)", `Area ${last.year} (km²)`]}
            rows={CLASS_KEYS.map((k) => [CLASS_NAME[k], fmt(first?.share[k], 1), fmt(last.share[k], 1), first ? signed(last.share[k] - first.share[k]) : "—", fmt((last.share[k] / 100) * r.area, 1)])} />
        </Section>
      )}

      {first && last && first.year !== last.year && (
        <Section title={`How ${r.name} changed from ${first.year} to ${last.year}`}>
          <p className="mb-2 text-sm text-muted">{first.year} (compare with {last.year} above):</p>
          <ShareBar share={first.share} label={`Share of ${r.name} by land-cover class, ${first.year}`} />
          <p className="mt-4">
            {ch.slice(0, 3).map((c) => `${CLASS_NAME[c.k]} ${signed(c.d)} percentage points (${signed((c.d / 100) * r.area, 2)} km²)`).join("; ")}.
          </p>
        </Section>
      )}

      <Section title={`Other local units in ${district.name} district`}>
        <ul className="grid gap-x-6 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
          {siblings.filter((s) => s.code !== r.code).map((s) => <li key={s.code}><Link href={regionHref(s)}>{s.name}</Link></li>)}
        </ul>
      </Section>

      {faq.length > 0 && (
        <Section title={`Questions about ${r.name}`}>
          <dl className="space-y-4">{faq.map((f) => <div key={f.q}><dt className="font-medium">{f.q}</dt><dd className="mt-1 text-muted">{f.a}</dd></div>)}</dl>
          <JsonLd data={{ "@context": "https://schema.org", "@type": "FAQPage", mainEntity: faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })) }} />
        </Section>
      )}
      <JsonLd data={{ "@context": "https://schema.org", "@type": "Place", name: r.name, url: `${siteConfig.url}${regionHref(r)}`,
        containedInPlace: { "@type": "AdministrativeArea", name: areaTitle(district) }, geo: { "@type": "GeoCoordinates", latitude: r.lat, longitude: r.lon } }} />
      <Source>Impact Observatory / Esri 10 m land cover (CC BY 4.0); boundaries: Survey Department of Nepal via OCHA.</Source>
    </>
  );
}
