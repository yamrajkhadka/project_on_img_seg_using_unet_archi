import type { Metadata } from "next";
import Link from "next/link";
import { PageHeader, Section, Source } from "@/components/ui";
import { placeLabel, regionsAt, townOf } from "@/lib/data";
import { fmt, nptDate, WEATHER_CODES } from "@/lib/format";
import { weatherMany } from "@/lib/live";

export const revalidate = 1800;

export const metadata: Metadata = {
  title: "Nepal Weather Today: Temperature & Rain in All 77 Districts",
  description: "Current temperature, weather, humidity, wind and rain in Kathmandu, Pokhara, Biratnagar and the main town of every district of Nepal, with 7-day forecasts.",
  alternates: { canonical: "/weather" },
};

export default async function Page() {
  const provinces = regionsAt(1);
  const districts = regionsAt(2);
  const data = await weatherMany(districts.map((d) => townOf(d.code)!));
  const byCode = new Map(districts.map((d, i) => [d.code, data?.[i]?.current]));
  return (
    <>
      <PageHeader eyebrow={`Live · ${nptDate(new Date())}`} title="Weather in Nepal today"
        intro={<p>Current conditions at the main town of every district, from the Terai at 70 m to mountain districts above 3,000 m. Select a district for the hourly and 7-day forecast.</p>} />
      {provinces.map((p) => (
        <Section key={p.code} title={`${p.name} Province`}>
          <div className="overflow-x-auto">
            <table className="data">
              <thead><tr><th>Place</th><th className="num">°C</th><th>Weather</th><th className="num">Humidity</th><th className="num">Wind (km/h)</th><th className="num">Rain (mm)</th></tr></thead>
              <tbody>{districts.filter((d) => d.parent === p.code).map((d) => {
                const c = byCode.get(d.code);
                return (
                  <tr key={d.code}><td><Link href={`/weather/${d.slug}`}>{placeLabel(d)}</Link></td><td className="num">{fmt(c?.temperature_2m, 1)}</td>
                    <td>{c ? WEATHER_CODES[c.weather_code] ?? "" : "—"}</td><td className="num">{c ? `${c.relative_humidity_2m}%` : "—"}</td>
                    <td className="num">{fmt(c?.wind_speed_10m)}</td><td className="num">{fmt(c?.precipitation, 1)}</td></tr>
                );
              })}</tbody>
            </table>
          </div>
        </Section>
      ))}
      <Source>Open-Meteo (ECMWF, DWD, NOAA). For official warnings see the Department of Hydrology and Meteorology (mfd.gov.np).</Source>
    </>
  );
}
