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
      <Section title="The network">
        <div className="prose-site">
          <p><b>Architecture.</b> A U-Net: encoder with 32-64-128-256 filters, a 512-filter bottleneck with dropout 0.5, a decoder with transposed convolutions and skip connections, each block Conv 3×3 + ReLU → BatchNorm → Conv 3×3 + ReLU → BatchNorm, and a softmax over 7 classes; 7,772,071 parameters.</p>
          <p><b>Training in the paper.</b> 803 DeepGlobe tiles (2,448 px at 0.5 m) resized to 224 px, so the network sees 5.46 m per pixel, with flips, brightness/contrast, shift-scale-rotate and elastic augmentation; loss = 0.4 × weighted cross-entropy + 0.6 × focal Tversky (α 0.7, β 0.3, γ 0.75); Adam with early stopping.</p>
        </div>
        <table className="data mt-4 max-w-md"><thead><tr><th>Class (paper, DeepGlobe)</th><th className="num">IoU</th></tr></thead>
          <tbody>{[["Urban land", 0.7187], ["Agriculture land", 0.717], ["Rangeland", 0.3525], ["Forest land", 0.7216], ["Water", 0.6582], ["Barren land", 0.5681], ["Unknown", 0.9826], ["Mean", 0.6741]].map(([c, v]) => <tr key={c}><td>{c}</td><td className="num">{fmt(v as number, 3)}</td></tr>)}</tbody>
        </table>
      </Section>
      <Section title="Four findings from scaling the model up">
        <ol className="prose-site list-decimal space-y-2 pl-5">
          <li><b>The network depends on how images are shrunk.</b> Training used OpenCV&apos;s bilinear resize without antialiasing, which keeps a sparse sample of the 0.5 m texture. On 40 DeepGlobe tiles forest IoU is 0.58 with that resize and 0.00 with an antialiased one; this site resamples the same way.</li>
          <li><b>A post-processing quirk.</b> The published clean-up never re-assigns the specks it removes, so they become class 0, urban land. Here, removed specks take the class of the nearest surviving region.</li>
          <li><b>Annual mosaics need colour normalisation.</b> The yearly Sentinel-2 cloudless mosaics differ in processing; without correction the network reports processing changes as land-cover change. Each year is mapped onto the 2018–2025 average. Even so, the 2016 and 2018 mosaics are read less reliably than 2019 onwards.</li>
          <li><b>The domain gap the paper predicted is large.</b> On Nepal&apos;s held-out districts the original weights reach a mean IoU of {fmt(model.baseline_miou, 3)}; fine-tuning raised it to {fmt(model.val_miou, 3)}.</li>
        </ol>
      </Section>
      <Section title="Data sources">
        <div className="overflow-x-auto"><table className="data">
          <thead><tr><th>Source</th><th>Used for</th><th>Licence</th></tr></thead>
          <tbody>{[
            ["Esri World Imagery and Wayback", "Sub-metre imagery for the U-Net and maps", "Esri terms of use"],
            ["Sentinel-2 cloudless by EOX (2016, 2018 onwards)", "Annual 10 m mosaics for the year-by-year land cover", "CC BY 4.0 (2016) / CC BY-NC-SA 4.0"],
            ["Impact Observatory / Esri 10 m land cover", "Training labels, reference map, local-unit land cover", "CC BY 4.0"],
            ["ESA WorldCover 2021", "Independent test of the model", "CC BY 4.0"],
            ["NASA FIRMS (VIIRS, MODIS)", "Active fires, live and since 2001", "NASA open data"],
            ["NASA GIBS (MODIS Terra)", "Daily satellite images", "NASA open data"],
            ["USGS ComCat", "Earthquakes, live and since 1950", "Public domain"],
            ["Open-Meteo (ECMWF, DWD, NOAA; CAMS; ERA5)", "Weather, air quality, weather history", "CC BY 4.0"],
            ["GloFAS v4 (Copernicus EMS) via Open-Meteo", "River flows, forecasts and flood levels", "CC BY 4.0"],
            ["World Bank WDI and Climate Knowledge Portal (CRU TS, ERA5)", "National statistics and climate since 1901", "CC BY 4.0"],
            ["OCHA / Survey Department of Nepal", "Province, district and local-unit boundaries", "CC BY-IGO"],
          ].map(([s, u, l]) => <tr key={s}><td>{s}</td><td>{u}</td><td>{l}</td></tr>)}</tbody>
        </table></div>
      </Section>
      <Section title="Limitations">
        <ul className="prose-site list-disc space-y-2 pl-5">
          <li>Every figure is a model estimate. Against ESA WorldCover the Nepal-adapted model agrees on about 70–75% of pixels, close to how well the two reference maps agree with each other.</li>
          <li>Rangeland is the hardest class (the paper&apos;s IoU 0.35): grassland, shrubs and fallow fields look alike from above.</li>
          <li>District figures rest on 8 or more sample sites each; read their 95% intervals before comparing districts.</li>
          <li>Weather, air quality and river flows are model values at a point (a district&apos;s main town or a river gauge), not ground measurements. Use official sources for warnings.</li>
          <li>Satellites miss fires under thick cloud or smoke and very small fires; earthquake catalogues are incomplete for small events before recent decades.</li>
        </ul>
      </Section>
      <JsonLd data={{ "@context": "https://schema.org", "@type": "ScholarlyArticle", headline: "Land Cover Segmentation from Satellite Imagery Using U-Net with Custom Loss and Morphological Postprocessing",
        author: [{ "@type": "Person", name: "Yamraj Khadka" }, { "@type": "Person", name: "Rajat Pradhan" }], datePublished: "2026",
        isPartOf: { "@type": "Periodical", name: "Journal of Engineering Issues and Solutions" }, sameAs: siteConfig.paper.url }} />
      <Source>Khadka &amp; Pradhan (2026); evaluation against ESA WorldCover 2021 and Impact Observatory 2021 on 11 held-out districts.</Source>
    </>
  );
}
