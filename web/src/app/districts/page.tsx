import type { Metadata } from "next";
import Link from "next/link";
import { Choropleths } from "@/components/Choropleth";
import { PageHeader, Section, Source } from "@/components/ui";
import { childrenOf, regionsAt } from "@/lib/data";
import { fmt } from "@/lib/format";
import { latestShare } from "@/lib/insights";

export const metadata: Metadata = {
  title: "All 77 Districts of Nepal: Land Cover, Forest and Built-up Area",
  description: "Every district of Nepal with its area, forest cover, farmland and built-up land from AI satellite mapping, grouped by province, with links to air quality and weather.",
  alternates: { canonical: "/districts" },
};

export default function Page() {
  const provinces = regionsAt(1);
  return (
    <>
      <PageHeader eyebrow="Districts" title="The 77 districts of Nepal"
        intro={<p>Forest, farmland and built-up shares of every district, estimated by the Nepal-adapted U-Net from satellite imagery. Select a district for its map, trend since 2016, municipalities, air quality and weather.</p>} />
      <Choropleths keys={["forest"]} />
      {provinces.map((p) => (
        <Section key={p.code} title={`${p.name} Province`}>
          <div className="overflow-x-auto">
            <table className="data">
              <thead><tr><th>District</th><th className="num">Area (km²)</th><th className="num">Forest %</th><th className="num">Farmland %</th><th className="num">Built-up %</th><th>Live</th></tr></thead>
              <tbody>
                {childrenOf(p.code).map((d) => {
                  const s = latestShare(d.code);
                  return (
                    <tr key={d.code}>
                      <td><Link href={`/districts/${d.slug}`}>{d.name}</Link></td>
                      <td className="num">{fmt(d.area)}</td>
                      <td className="num">{fmt(s?.forest, 1)}</td>
                      <td className="num">{fmt(s?.agriculture, 1)}</td>
                      <td className="num">{fmt(s?.urban, 1)}</td>
                      <td><Link href={`/air-quality/${d.slug}`}>Air</Link> · <Link href={`/weather/${d.slug}`}>Weather</Link></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Section>
      ))}
      <Source>Nepal-adapted U-Net on the latest high-resolution imagery, 1,038 stratified sample sites.</Source>
    </>
  );
}
