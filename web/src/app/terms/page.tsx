import type { Metadata } from "next";
import { PageHeader } from "@/components/ui";

export const metadata: Metadata = { title: "Terms of Use", description: "Terms for using Nepal Monitor and its data.", alternates: { canonical: "/terms" } };

export default function Page() {
  return (
    <>
      <PageHeader title="Terms of use" intro={<p>Last updated 21 October 2026.</p>} />
      <div className="prose-site">
        <p><b>Information only.</b> Nepal Monitor shows model estimates and data from public scientific sources for information and research. It is not an official warning service. For weather, flood and earthquake warnings and emergencies, follow the Department of Hydrology and Meteorology, the National Earthquake Monitoring and Research Center and the National Disaster Risk Reduction and Management Authority of Nepal.</p>
        <p><b>No warranty.</b> Data may be delayed, incomplete or wrong. We provide the site as it is, without warranties, and are not liable for decisions made using it.</p>
        <p><b>Reuse.</b> You may quote and link to the site&apos;s figures with attribution to &quot;Nepal Monitor (nepalmonitor.yamrajkhadka.com.np)&quot; and to the original data sources named on each page, under their licences (for example CC BY 4.0 for Impact Observatory land cover and CC BY-NC-SA 4.0 for Sentinel-2 cloudless by EOX). Research use of the model should cite Khadka &amp; Pradhan (2026).</p>
      </div>
    </>
  );
}
