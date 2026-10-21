import type { Metadata } from "next";
import Link from "next/link";
import { PageHeader, Section } from "@/components/ui";
import { childrenOf, regionsAt } from "@/lib/data";

export const metadata: Metadata = {
  title: "All Municipalities and Rural Municipalities of Nepal by District",
  description: "Every metropolitan city, municipality and rural municipality of Nepal, by province and district, with land cover, forest, built-up area and maps.",
  alternates: { canonical: "/municipalities" },
};

export default function Page() {
  const provinces = regionsAt(1);
  const units = regionsAt(3);
  return (
    <>
      <PageHeader eyebrow="Local units" title="Municipalities and rural municipalities of Nepal"
        intro={<p>All {units.length} local units in Nepal&apos;s boundary data, from metropolitan cities to rural municipalities and protected areas. Each page shows its map, land cover and change since 2017.</p>} />
      {provinces.map((p) => (
        <Section key={p.code} title={`${p.name} Province`}>
          {childrenOf(p.code).map((d) => (
            <div key={d.code} className="mb-4">
              <h3 className="font-medium"><Link href={`/districts/${d.slug}`}>{d.name} district</Link></h3>
              <p className="text-sm leading-7">
                {childrenOf(d.code).map((u, i) => <span key={u.code}>{i ? " · " : ""}<Link href={`/municipalities/${u.slug}`}>{u.name}</Link></span>)}
              </p>
            </div>
          ))}
        </Section>
      ))}
    </>
  );
}
