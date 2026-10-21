import type { Metadata } from "next";
import Link from "next/link";
import { RegionMap } from "@/components/RegionMap";
import { PageHeader, Section, Source } from "@/components/ui";
import { childrenOf, regionsAt } from "@/lib/data";
import { fmt } from "@/lib/format";
import { latestShare } from "@/lib/insights";

export const metadata: Metadata = {
  title: "The 7 Provinces of Nepal: Map, Districts and Land Cover",
  description: "Koshi, Madhesh, Bagmati, Gandaki, Lumbini, Karnali and Sudur Paschim: area, districts, forest cover, farmland and built-up land of each province of Nepal.",
  alternates: { canonical: "/provinces" },
};

export default function Page() {
  const provinces = regionsAt(1);
  return (
    <>
      <PageHeader eyebrow="Provinces" title="The 7 provinces of Nepal"
        intro={<p>Nepal has seven provinces, created by the 2015 constitution. Their land cover differs sharply: Madhesh is mostly farmland, Karnali and Sudur Paschim are mountainous and forested, Bagmati holds the Kathmandu Valley.</p>} />
      <RegionMap label="Map of Nepal's 7 provinces" contextLevel={1} context={provinces.map((p) => p.code)} />
      <Section title="Provinces compared">
        <div className="overflow-x-auto">
          <table className="data">
            <thead><tr><th>Province</th><th className="num">Area (km²)</th><th className="num">Districts</th><th className="num">Forest %</th><th className="num">Farmland %</th><th className="num">Built-up %</th></tr></thead>
            <tbody>
              {provinces.map((p) => {
                const s = latestShare(p.code);
                return (
                  <tr key={p.code}>
                    <td><Link href={`/provinces/${p.slug}`}>{p.name}</Link></td>
                    <td className="num">{fmt(p.area)}</td>
                    <td className="num">{childrenOf(p.code).length}</td>
                    <td className="num">{fmt(s?.forest, 1)}</td>
                    <td className="num">{fmt(s?.agriculture, 1)}</td>
                    <td className="num">{fmt(s?.urban, 1)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Section>
      <Source>Nepal-adapted U-Net on the latest high-resolution imagery, 1,038 stratified sample sites.</Source>
    </>
  );
}
