import type { Metadata } from "next";
import Link from "next/link";
import { LandCoverTool } from "@/components/tools/LandCoverTool";
import { JsonLd, PageHeader, Section } from "@/components/ui";
import { placeOptions } from "@/lib/placeOptions";
import { siteConfig } from "@/lib/site";

export const metadata: Metadata = {
  title: "AI Land-Cover Map of Any Place in Nepal (Free, in Your Browser)",
  description: "Map forest, farmland, rangeland, built-up land, water and bare land for any place in Nepal from sub-metre or Sentinel-2 satellite imagery with a U-Net deep-learning model that runs in your browser.",
  alternates: { canonical: "/map" },
};

export default function Page() {
  return (
    <>
      <PageHeader eyebrow="AI tool" title="Map the land cover of any place in Nepal"
        intro={<p>Pick a place, choose the imagery, and the U-Net from <a href={siteConfig.paper.url}>Khadka &amp; Pradhan (2026)</a> labels every 5.46 m pixel as urban, agriculture, rangeland, forest, water or barren land. It runs entirely in your browser, with no sign-up and no upload, and gives you the map and the area of each class.</p>} />
      <LandCoverTool regions={placeOptions()} mode="map" />
      <Section title="How it works">
        <div className="prose-site">
          <p>The imagery is resampled to 5.46 m per pixel, the scale the network was trained at (DeepGlobe&apos;s 2,448-pixel, 0.5 m tiles resized to 224 pixels), with the same bilinear method used in training. Each year&apos;s Sentinel-2 cloudless mosaic is colour-normalised first. Regions smaller than 64 pixels (0.19 ha) take the class of the nearest larger region, the paper&apos;s morphological clean-up.</p>
          <p>The Nepal-adapted model agrees with the independent ESA WorldCover map on about 72% of pixels on districts it never saw in training. Treat single maps as estimates. For whole districts with confidence intervals see the <Link href="/districts">district pages</Link>, and to compare two dates use <Link href="/change">change detection</Link>.</p>
        </div>
      </Section>
      <JsonLd data={{ "@context": "https://schema.org", "@type": "WebApplication", name: "Nepal land-cover mapper", applicationCategory: "Science", operatingSystem: "Any (web browser)",
        url: `${siteConfig.url}/map`, offers: { "@type": "Offer", price: "0", priceCurrency: "USD" } }} />
    </>
  );
}
