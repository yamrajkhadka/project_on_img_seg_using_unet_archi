import type { Metadata } from "next";
import Link from "next/link";
import { LandCoverTool } from "@/components/tools/LandCoverTool";
import { JsonLd, PageHeader, Section } from "@/components/ui";
import { placeOptions } from "@/lib/placeOptions";
import { siteConfig } from "@/lib/site";

export const metadata: Metadata = {
  title: "Land-Cover Change Detection for Nepal: Compare Two Years with AI",
  description: "See where forest, farmland and built-up land changed between two years anywhere in Nepal: a U-Net deep-learning model compares Sentinel-2 cloudless mosaics since 2016 and the latest sub-metre imagery, in your browser.",
  alternates: { canonical: "/change" },
};

export default function Page() {
  return (
    <>
      <PageHeader eyebrow="AI tool" title="Detect land-cover change anywhere in Nepal"
        intro={<p>Choose a place and two dates. The U-Net maps both on the same 5.46 m grid and shows where the land changed, how much of each class was gained or lost, and the largest transitions, such as farmland to built-up land.</p>} />
      <LandCoverTool regions={placeOptions()} mode="change" />
      <Section title="Reading the result">
        <div className="prose-site">
          <p>Compare like with like: two Sentinel-2 cloudless years are the fairest pair, because they share the same 10 m sensor and processing, and each year is colour-normalised before classification. Mixing sub-metre Esri imagery with a 10 m mosaic exaggerates change. National and district trends with confidence intervals are on the <Link href="/land-cover">land-cover pages</Link>.</p>
        </div>
      </Section>
      <JsonLd data={{ "@context": "https://schema.org", "@type": "WebApplication", name: "Nepal land-cover change detection", applicationCategory: "Science",
        operatingSystem: "Any (web browser)", url: `${siteConfig.url}/change`, offers: { "@type": "Offer", price: "0", priceCurrency: "USD" } }} />
    </>
  );
}
