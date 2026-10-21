import type { Metadata } from "next";
import { LineChart } from "@/components/charts";
import { PageHeader, Section, Source, TableTwin } from "@/components/ui";
import { climate } from "@/lib/data";
import { fmt } from "@/lib/format";
import { signed } from "@/lib/insights";

export const metadata: Metadata = {
  title: "Climate of Nepal Since 1901: Temperature & Rainfall Trends",
  description: "Nepal's national average temperature and annual rainfall every year since 1901 (CRU TS) and 1950 (ERA5), the warming trend per decade, the warmest and wettest years, updated every year.",
  alternates: { canonical: "/climate" },
};

function slope(series: [number, number][], from: number): number {
  const pts = series.filter((p) => p[0] >= from);
  const n = pts.length, mx = pts.reduce((s, p) => s + p[0], 0) / n, my = pts.reduce((s, p) => s + p[1], 0) / n;
  return pts.reduce((s, p) => s + (p[0] - mx) * (p[1] - my), 0) / pts.reduce((s, p) => s + (p[0] - mx) ** 2, 0);
}
const mean = (series: [number, number][], a: number, b: number) => {
  const v = series.filter((p) => p[0] >= a && p[0] <= b).map((p) => p[1]);
  return v.reduce((s, x) => s + x, 0) / v.length;
};

export default function Page() {
  const t = climate.cru_tas, p = climate.cru_pr;
  const last = t.series[t.series.length - 1];
  const normalT = mean(t.series, 1991, 2020);
  const warmest = [...t.series].sort((a, b) => b[1] - a[1]).slice(0, 5);
  const wettest = [...p.series].sort((a, b) => b[1] - a[1]).slice(0, 5);
  const driest = [...p.series].sort((a, b) => a[1] - b[1]).slice(0, 5);
  const perDecade = slope(t.series, 1971) * 10;
  return (
    <>
      <PageHeader eyebrow="Climate" title="Nepal's climate since 1901"
        intro={<p>Nepal&apos;s average temperature was {fmt(last[1], 2)}°C in {last[0]}, {signed(last[1] - normalT, 2)}°C compared with the 1991–2020 normal. Since 1971 the country has warmed by about {fmt(perDecade, 2)}°C per decade. The warmest years on record are {warmest.map((w) => w[0]).join(", ")}.</p>} />
      <Section title="Average temperature" note={`National mean of the CRU TS 4.08 0.5° grid (World Bank Climate Knowledge Portal); years after ${t.base_end} (shaded) from ERA5, calibrated to it over the ten years they share.`}>
        <LineChart series={t.series} label="Nepal annual mean temperature since 1901" unit="°C" digits={1} highlightFrom={t.base_end + 1} />
        <TableTwin caption="Nepal annual mean temperature" head={["Year", "°C"]} rows={[...t.series].reverse().map((s) => [s[0], fmt(s[1], 2)])} />
      </Section>
      <Section title="Annual rainfall" note={`National mean precipitation, mm per year; years after ${p.base_end} from ERA5, calibrated.`}>
        <LineChart series={p.series} label="Nepal annual precipitation since 1901" unit="mm/year" digits={0} highlightFrom={p.base_end + 1} />
        <p className="mt-4 text-sm">Wettest years: {wettest.map((w) => `${w[0]} (${fmt(w[1])} mm)`).join(", ")}. Driest: {driest.map((w) => `${w[0]} (${fmt(w[1])} mm)`).join(", ")}.</p>
        <TableTwin caption="Nepal annual precipitation" head={["Year", "mm"]} rows={[...p.series].reverse().map((s) => [s[0], fmt(s[1])])} />
      </Section>
      <Section title="The same from ERA5 reanalysis (since 1950)" note="ERA5 is a weather-model reconstruction; it runs cooler and wetter than CRU over Nepal's mountains but shows the same changes.">
        <LineChart series={climate.era5_tas.series} label="Nepal mean temperature, ERA5" unit="°C" highlightFrom={climate.era5_tas.base_end + 1} />
      </Section>
      <Source>CRU TS 4.08 and ERA5 national averages from the World Bank Climate Knowledge Portal; recent years: ERA5 via Open-Meteo at 24 points across Nepal, calibrated.</Source>
    </>
  );
}
