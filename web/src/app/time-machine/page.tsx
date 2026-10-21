import type { Metadata } from "next";
import Link from "next/link";
import { type TMPlace, TimeMachine } from "@/components/tools/TimeMachine";
import { PageHeader, Section } from "@/components/ui";
import { placeLabel, regionsAt, rivers, townOf } from "@/lib/data";
import { nowMs } from "@/lib/format";

export const revalidate = 86400; // "yesterday" moves on every day

export const metadata: Metadata = {
  title: "Nepal Time Machine: Weather, Earthquakes & Rivers on Any Date",
  description: "Look up any day or month since 1940 in any district of Nepal: temperature and rain, air quality, earthquakes, river flows and floods, and NASA's satellite image of the country that day.",
  alternates: { canonical: "/time-machine" },
};

export default function Page() {
  const kathmandu = regionsAt(2).find((d) => d.slug === "kathmandu")!;
  const places: TMPlace[] = [kathmandu, ...regionsAt(2).filter((d) => d.code !== kathmandu.code)].map((d) => {
    const t = townOf(d.code)!;
    return { label: placeLabel(d), lat: t.lat, lon: t.lon };
  });
  const tmRivers = rivers.rivers.map((r) => ({
    ...r, median: (rivers.climatology[r.key] ?? []).map((c) => c[2]), q2: rivers.thresholds[r.key]?.q2 ?? Infinity, q10: rivers.thresholds[r.key]?.q10 ?? Infinity,
  }));
  return (
    <>
      <PageHeader eyebrow="Time machine" title="Nepal on any date"
        intro={<p>Choose a district and a day or a month, back to 1940, to see the weather, air quality, earthquakes and river flows, and NASA&apos;s view of Nepal from space that day. For years, decades and centuries see <Link href="/land-cover">land cover year by year</Link>, <Link href="/climate">climate since 1901</Link> and <Link href="/history">Nepal since 1768</Link>.</p>} />
      <TimeMachine places={places} rivers={tmRivers} yesterday={new Date(nowMs() - 86400_000 + 5.75 * 3600_000).toISOString().slice(0, 10)} />
      <Section title="What each record covers">
        <table className="data"><tbody>
          <tr><td>Weather</td><td>1940 to yesterday (ERA5 reanalysis; recent weeks from forecast models)</td></tr>
          <tr><td>Air quality</td><td>Recent years only (CAMS); older dates show &quot;not available&quot;</td></tr>
          <tr><td>River flows</td><td>1984 to yesterday (GloFAS v4)</td></tr>
          <tr><td>Earthquakes</td><td>Magnitude 2.5 and above in the USGS catalogue (complete for the region only in recent decades)</td></tr>
          <tr><td>Satellite image</td><td>Daily since March 2000 (NASA MODIS Terra)</td></tr>
        </tbody></table>
      </Section>
    </>
  );
}
