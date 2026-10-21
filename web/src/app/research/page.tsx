import type { Metadata } from "next";
import { JsonLd, PageHeader, Section, Source } from "@/components/ui";
import { model } from "@/lib/data";
import { fmt } from "@/lib/format";
import { siteConfig } from "@/lib/site";

export const metadata: Metadata = {
  title: "The Research: U-Net Land-Cover Segmentation for Nepal",
  description: "The deep-learning model behind Nepal Monitor: a U-Net with a focal Tversky loss and morphological post-processing (Khadka & Pradhan, 2026), adapted to Nepal and tested on held-out districts.",
  alternates: { canonical: "/research" },
};

export default function Page() {
  const rows = model.results.filter((r) => r.reference.startsWith("ESA"));
  const sources = [...new Set(rows.map((r) => r.source))];
  return (
    <>
      <PageHeader eyebrow="Research" title="The model behind the maps"
        intro={<p>Every land-cover figure on this site comes from a U-Net convolutional neural network that labels each pixel of a satellite image as urban, agriculture, rangeland, forest, water, barren or unknown, designed and published by Yamraj Khadka and Rajat Pradhan.</p>} />
      <Section title="The paper">
        <p className="prose-site">{siteConfig.paper.citation} <a href={siteConfig.paper.url}>doi:{siteConfig.paper.doi}</a></p>
        <p className="prose-site">The paper trains a U-Net from scratch on the DeepGlobe Land Cover dataset with a composite loss (weighted cross-entropy plus focal Tversky) to cope with rare classes, and cleans its output with morphological post-processing. Its closing recommendation, fine-tuning for Nepal, is what this site does.</p>
        <p className="mt-2"><a href={siteConfig.appUrl}>Interactive app (the paper&apos;s demo, upgraded)</a> · <a href={siteConfig.codeUrl}>Code and data</a></p>
      </Section>
      <Section title="Adapting it to Nepal">
        <p className="prose-site">The network was fine-tuned on Nepal: {model.summary}. Validation mean IoU on districts never seen in training rose from {fmt(model.baseline_miou, 3)} with the paper&apos;s weights to {fmt(model.val_miou, 3)}.</p>
      </Section>
      <Section title="Independent test" note={model.evaluation}>
        <div className="overflow-x-auto"><table className="data">
          <thead><tr><th>Imagery</th><th className="num">Original model</th><th className="num">Nepal-adapted</th></tr></thead>
          <tbody>{sources.map((s) => {
            const o = rows.find((r) => r.source === s && r.model === "paper"), n = rows.find((r) => r.source === s && r.model === "nepal");
            return <tr key={s}><td>{s}</td><td className="num">{o ? `${fmt(o.overall * 100)}% · mIoU ${fmt(o.miou, 2)}` : "—"}</td><td className="num">{n ? `${fmt(n.overall * 100)}% · mIoU ${fmt(n.miou, 2)}` : "—"}</td></tr>;
          })}</tbody>
        </table></div>
        {model.inter_reference && <p className="mt-3 text-sm text-muted">Pixel agreement with ESA WorldCover 2021. For scale, two independent reference maps agree with each other on {fmt(model.inter_reference.overall * 100)}% of pixels.</p>}
      </Section>
      <Section title="How the national figures are made">
        <p className="prose-site">Mapping all of Nepal at 5.5 m every year would take days of computing, so the site uses a statistical survey: 1,038 random 1.2 km windows, at least eight in every district, are classified each year, and the shares are estimated with 95% confidence intervals (stratified by district). Each year&apos;s Sentinel-2 cloudless mosaic is colour-normalised before classification so processing differences are not mistaken for change, and every estimate is shown next to the independent Impact Observatory 10 m map.</p>
      </Section>
      <JsonLd data={{ "@context": "https://schema.org", "@type": "ScholarlyArticle", headline: "Land Cover Segmentation from Satellite Imagery Using U-Net with Custom Loss and Morphological Postprocessing",
        author: [{ "@type": "Person", name: "Yamraj Khadka" }, { "@type": "Person", name: "Rajat Pradhan" }], datePublished: "2026",
        isPartOf: { "@type": "Periodical", name: "Journal of Engineering Issues and Solutions" }, sameAs: siteConfig.paper.url }} />
      <Source>Khadka &amp; Pradhan (2026); evaluation against ESA WorldCover 2021 and Impact Observatory 2021 on 11 held-out districts.</Source>
    </>
  );
}
