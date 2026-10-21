import type { Metadata } from "next";
import Link from "next/link";
import { ModelLab } from "@/components/tools/ModelLab";
import { JsonLd, PageHeader, Section } from "@/components/ui";
import { siteConfig } from "@/lib/site";

export const metadata: Metadata = {
  title: "U-Net Land Cover Segmentation Demo: Classify Your Satellite Image",
  description: "Upload a satellite or aerial image and the U-Net of Khadka & Pradhan (2026) labels every pixel as urban, agriculture, rangeland, forest, water or barren land, in your browser, as in the paper's demo.",
  alternates: { canonical: "/model-lab" },
};

export default function Page() {
  return (
    <>
      <PageHeader eyebrow="Model lab · the paper's demo" title="Classify every pixel of your satellite image"
        intro={<p>The demo described in Section 3.6 of <a href={siteConfig.paper.url}>Khadka &amp; Pradhan (2026)</a>: upload an image and the U-Net segments it into the seven DeepGlobe land-cover classes. Choose the original DeepGlobe model or the Nepal-adapted one, feed the image exactly as the paper did or at its true scale, and switch post-processing on or off.</p>} />
      <ModelLab />
      <Section title="Tips">
        <div className="prose-site">
          <p>The original model was trained on 50 cm DeepGlobe imagery, so it works best on sub-metre images such as Google Earth or drone mosaics taken straight down. For Sentinel-2 or Landsat images use the Nepal-adapted model at true scale (10 m or 30 m per pixel). No image to hand? <Link href="/map">Map any place in Nepal</Link> directly from satellite imagery.</p>
        </div>
      </Section>
      <JsonLd data={{ "@context": "https://schema.org", "@type": "WebApplication", name: "U-Net land cover segmentation demo", applicationCategory: "Science",
        operatingSystem: "Any (web browser)", url: `${siteConfig.url}/model-lab`, offers: { "@type": "Offer", price: "0", priceCurrency: "USD" } }} />
    </>
  );
}
