import type { Metadata } from "next";
import { AlertList } from "@/components/Alerts";
import { PageHeader, Section, Source } from "@/components/ui";
import { computeAlerts } from "@/lib/alerts";
import { nptDate } from "@/lib/format";

export const revalidate = 900;

export const metadata: Metadata = {
  title: "Nepal Alerts Today: Air Pollution, Fires, Earthquakes, Heavy Rain & Floods",
  description: "Live alerts for Nepal: unhealthy air, unusual forest-fire activity, strong earthquakes, heavy-rain forecasts and rivers above flood levels, checked every 15 minutes.",
  alternates: { canonical: "/alerts" },
};

export default async function Page() {
  const alerts = await computeAlerts();
  return (
    <>
      <PageHeader eyebrow={`Live · ${nptDate(new Date())}`} title="Alerts for Nepal"
        intro={<p>Checked every 15 minutes from live data. These are automated signals, not official warnings: always follow the Department of Hydrology and Meteorology and the National Disaster Risk Reduction and Management Authority.</p>} />
      <AlertList alerts={alerts} />
      <Section title="What triggers an alert">
        <table className="data"><tbody>
          <tr><td>Air quality</td><td>US AQI above 150 (unhealthy) at a district&apos;s main town; severe above 200.</td></tr>
          <tr><td>Fires</td><td>20 or more detections in 24 hours and more than twice the previous six days&apos; daily average.</td></tr>
          <tr><td>Earthquake</td><td>Magnitude 4.5 or more within the region in the last 7 days; severe from 5.5.</td></tr>
          <tr><td>Heavy rain</td><td>64.5 mm or more forecast in a day at a district town in the next 3 days (Nepal&apos;s &quot;heavy rain&quot; threshold); severe from 115.6 mm.</td></tr>
          <tr><td>River flood watch</td><td>Forecast flow above the 2-year flood level at a major gauge in the next 7 days; severe above the 10-year level.</td></tr>
        </tbody></table>
      </Section>
      <Source>NASA FIRMS, USGS, Open-Meteo (CAMS, ECMWF, GloFAS).</Source>
    </>
  );
}
