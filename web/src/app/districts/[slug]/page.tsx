import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { AreaPage, areaDescription, areaTitle } from "@/components/AreaPage";
import { regionBySlug, regionsAt } from "@/lib/data";

export const dynamicParams = false;

export function generateStaticParams() {
  return regionsAt(2).map((r) => ({ slug: r.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const r = regionBySlug(2, (await params).slug);
  if (!r) return {};
  const title = `${areaTitle(r)}: Land Cover, Forest, Map & Municipalities`;
  return { title, description: areaDescription(r), alternates: { canonical: `/districts/${r.slug}` }, openGraph: { title, description: areaDescription(r) } };
}

export default async function Page({ params }: { params: Promise<{ slug: string }> }) {
  const r = regionBySlug(2, (await params).slug);
  if (!r) notFound();
  return <AreaPage r={r} />;
}
