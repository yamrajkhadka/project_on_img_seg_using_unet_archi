import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { AreaPage, areaDescription, areaTitle } from "@/components/AreaPage";
import { regionBySlug, regionsAt } from "@/lib/data";

export const dynamicParams = false;

export function generateStaticParams() {
  return regionsAt(1).map((r) => ({ slug: r.slug }));
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const r = regionBySlug(1, (await params).slug);
  if (!r) return {};
  const title = `${areaTitle(r)}: Land Cover, Forest, Map & Districts`;
  return { title, description: areaDescription(r), alternates: { canonical: `/provinces/${r.slug}` }, openGraph: { title, description: areaDescription(r) } };
}

export default async function Page({ params }: { params: Promise<{ slug: string }> }) {
  const r = regionBySlug(1, (await params).slug);
  if (!r) notFound();
  return <AreaPage r={r} />;
}
