import type { Metadata } from "next";
import Link from "next/link";
import { PageHeader, Section } from "@/components/ui";
import { siteConfig } from "@/lib/site";

export const metadata: Metadata = {
  title: "About Nepal Monitor",
  description: "Nepal Monitor is an independent, research-based site tracking Nepal's air, weather, hazards and land cover, built by Yamraj Khadka on peer-reviewed land-cover research.",
  alternates: { canonical: "/about" },
};

export default function Page() {
  return (
    <>
      <PageHeader title="About Nepal Monitor" />
      <div className="prose-site">
        <p>Nepal Monitor brings together, for every province, district and municipality of Nepal, the live conditions people check every day (air quality, weather, earthquakes, fires and rivers) and the slow changes that matter over years: forests, farmland and built-up land.</p>
        <p>It is built and maintained by <a href={siteConfig.authorUrl}>{siteConfig.author}</a>. The land-cover maps come from the U-Net model published in <a href={siteConfig.paper.url}>Khadka &amp; Pradhan (2026)</a>, adapted to Nepal; see <Link href="/research">the research</Link> for how it works and how accurate it is.</p>
        <p>Live data comes from public scientific services: NASA FIRMS (fires), the US Geological Survey (earthquakes), Copernicus/ECMWF via Open-Meteo (weather, air quality, river flow), and the World Bank (national statistics). Figures are model estimates, not official measurements; for warnings always follow Nepal&apos;s Department of Hydrology and Meteorology and the National Disaster Risk Reduction and Management Authority.</p>
      </div>
      <Section title="Data updates">
        <p className="prose-site">Live pages refresh every 10–60 minutes. Once a month the site adds every newly published year of satellite imagery, river records, climate and major events automatically, so its timelines keep growing.</p>
      </Section>
    </>
  );
}
