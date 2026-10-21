import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ShareBar } from "@/components/charts";
import { Breadcrumbs, PageHeader, Section, Source, TableTwin } from "@/components/ui";
import { CLASS_KEYS, CLASS_NAME, type ClassKey, landCoverOf, landcover, regionsAt } from "@/lib/data";
import { fmt } from "@/lib/format";
import { signed } from "@/lib/insights";

export const dynamicParams = false;

export function generateStaticParams() {
  return landcover.years.map((y) => ({ year: String(y) }));
}

export async function generateMetadata({ params }: { params: Promise<{ year: string }> }): Promise<Metadata> {
  const { year } = await params;
  const t = landCoverOf("NP")?.trend[year];
  const title = `Nepal Land Cover ${year}: Forest, Farmland & Built-up by District`;
  const description = t ? `Land cover of Nepal in ${year} from the ${year} Sentinel-2 cloudless mosaic: forest ${fmt(t.forest[0], 1)}%, farmland ${fmt(t.agriculture[0], 1)}%, built-up ${fmt(t.urban[0], 1)}%, for every province and district.` : `Land cover of Nepal in ${year}.`;
  return { title, description, alternates: { canonical: `/land-cover/${year}` }, openGraph: { title, description } };
}

export default async function Page({ params }: { params: Promise<{ year: string }> }) {
  const { year } = await params;
  const nat = landCoverOf("NP");
  const t = nat?.trend[year];
  if (!t) notFound();
  const years = landcover.years.map(String);
  const prev = years[years.indexOf(year) - 1];
  const tp = prev ? nat!.trend[prev] : undefined;
  const share = Object.fromEntries(CLASS_KEYS.map((k) => [k, t[k][0]])) as Record<ClassKey, number>;
  const ref = nat!.reference[year];
  const districts = regionsAt(2).map((d) => ({ d, s: landCoverOf(d.code)?.trend[year] })).filter((x) => x.s);
  return (
    <>
      <Breadcrumbs items={[{ href: "/land-cover", label: "Land cover" }, { label: year }]} />
      <PageHeader eyebrow={`Sentinel-2 cloudless ${year}`} title={`Nepal's land cover in ${year}`}
        intro={<p>The Nepal-adapted U-Net applied to the {year} Sentinel-2 cloudless annual mosaic (10 m) at {nat!.n} stratified sample sites: forest {fmt(t.forest[0], 1)}%, farmland {fmt(t.agriculture[0], 1)}%, rangeland {fmt(t.rangeland[0], 1)}%, built-up {fmt(t.urban[0], 1)}%, bare land {fmt(t.barren[0], 1)}% and water {fmt(t.water[0], 1)}%.</p>} />
      {(year === "2016" || year === "2018") && (
        <p className="mb-6 rounded-xl border border-line bg-card p-4 text-sm" style={{ color: "var(--warn)" }}>The {year} mosaic was processed differently from later years and the model reads it less reliably: built-up land is underestimated and bare land and water overestimated. Use 2019 and later for comparisons.</p>
      )}
      <Section title={`Nepal in ${year}`}>
        <ShareBar share={share} label={`Share of Nepal by land-cover class in ${year}`} />
        <TableTwin caption={`Nepal land cover ${year}`} head={["Class", "U-Net (%)", "95% interval", ...(tp ? [`Change since ${prev}`] : []), ...(ref ? ["Reference map (%)"] : [])]}
          rows={CLASS_KEYS.map((k) => [CLASS_NAME[k], fmt(t[k][0], 1), `${fmt(t[k][1], 1)}–${fmt(t[k][2], 1)}`, ...(tp ? [signed(t[k][0] - tp[k][0])] : []), ...(ref ? [fmt(ref[k], 1)] : [])])} />
      </Section>
      <Section title={`Every district in ${year}`} note="District estimates rest on 8 or more sample sites each, so their intervals are wide; compare districts with care.">
        <div className="overflow-x-auto"><table className="data">
          <thead><tr><th>District</th><th>Province</th><th className="num">Forest %</th><th className="num">Farmland %</th><th className="num">Rangeland %</th><th className="num">Built-up %</th></tr></thead>
          <tbody>{districts.map(({ d, s }) => <tr key={d.code}><td><Link href={`/districts/${d.slug}`}>{d.name}</Link></td><td>{d.province}</td>
            <td className="num">{fmt(s!.forest[0], 1)}</td><td className="num">{fmt(s!.agriculture[0], 1)}</td><td className="num">{fmt(s!.rangeland[0], 1)}</td><td className="num">{fmt(s!.urban[0], 1)}</td></tr>)}</tbody>
        </table></div>
      </Section>
      <p className="mb-6">Other years: {years.filter((y) => y !== year).map((y, i) => <span key={y}>{i ? " · " : ""}<Link href={`/land-cover/${y}`}>{y}</Link></span>)}</p>
      <Source>Nepal-adapted U-Net on EOX Sentinel-2 cloudless {year} (contains modified Copernicus Sentinel data); reference: Impact Observatory/Esri 10 m land cover.</Source>
    </>
  );
}
