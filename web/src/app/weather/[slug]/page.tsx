import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { areaTitle } from "@/components/AreaPage";
import { BarChart, HourlyChart } from "@/components/charts";
import { Breadcrumbs, JsonLd, PageHeader, Section, Source, Stat } from "@/components/ui";
import { placeLabel, regionByCode, regionBySlug, regionsAt, townOf } from "@/lib/data";
import { fmt, nptDate, WEATHER_CODES } from "@/lib/format";
import { weather } from "@/lib/live";

export const revalidate = 1800;

export const dynamicParams = false;

export function generateStaticParams() {
  return regionsAt(2).map((r) => ({ slug: r.slug })); // every district, rebuilt on the site's schedule
}

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const d = regionBySlug(2, (await params).slug);
  if (!d) return {};
  const place = placeLabel(d);
  const title = `Weather in ${place} Today & 7-Day Forecast`;
  const description = `Current temperature, rain, humidity and wind in ${place}, Nepal, with the hourly and 7-day weather forecast, sunrise, sunset and UV index.`;
  return { title, description, alternates: { canonical: `/weather/${d.slug}` }, openGraph: { title, description } };
}

const compass = (deg: number) => ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.round(deg / 45) % 8];

export default async function Page({ params }: { params: Promise<{ slug: string }> }) {
  const d = regionBySlug(2, (await params).slug);
  if (!d) notFound();
  const town = townOf(d.code)!;
  const place = placeLabel(d);
  const province = regionByCode(d.parent)!;
  const w = await weather(town);
  const c = w?.current;
  const nowIdx = w ? Math.max(0, w.hourly.time.findIndex((t) => t >= c!.time.slice(0, 13) + ":00")) : 0;
  const next48 = w ? { t: w.hourly.time.slice(Math.max(0, nowIdx - 12), nowIdx + 48), v: w.hourly.temperature_2m.slice(Math.max(0, nowIdx - 12), nowIdx + 48) } : null;
  // daily arrays run from 30 days ago to 6 days ahead
  const t0 = w ? Math.max(0, w.daily.time.indexOf(c!.time.slice(0, 10))) : 0;
  const tomorrow = t0 + 1;
  const past = w ? w.daily.time.slice(0, t0).map((day, i) => ({ day, max: w.daily.temperature_2m_max[i], min: w.daily.temperature_2m_min[i], rain: w.daily.precipitation_sum[i] })) : [];
  const pastRain = past.reduce((s, p) => s + (p.rain ?? 0), 0);
  const faq = w && c ? [
    { q: `What is the temperature in ${place} now?`, a: `It is ${fmt(c.temperature_2m, 1)}°C in ${town.town} (feels like ${fmt(c.apparent_temperature, 1)}°C), ${(WEATHER_CODES[c.weather_code] ?? "").toLowerCase()}, with ${c.relative_humidity_2m}% humidity, at ${c.time.slice(11)} Nepal time.` },
    { q: `Will it rain in ${place} tomorrow?`, a: `The chance of rain tomorrow is ${fmt(w.daily.precipitation_probability_max[tomorrow])}% with about ${fmt(w.daily.precipitation_sum[tomorrow], 1)} mm expected; temperatures from ${fmt(w.daily.temperature_2m_min[tomorrow])}°C to ${fmt(w.daily.temperature_2m_max[tomorrow])}°C.` },
    { q: `What is the elevation of ${town.town}?`, a: `The forecast point is at about ${fmt(w.elevation)} m above sea level.` },
  ] : [];
  return (
    <>
      <Breadcrumbs items={[{ href: "/weather", label: "Weather" }, { label: place }]} />
      <PageHeader eyebrow={`Live · ${nptDate(new Date())} · ${province.name} Province`} title={`Weather in ${place}`}
        intro={<p>Current weather and the 7-day forecast for {town.town}{town.town !== d.name ? `, the main town of ${d.name} district` : ""}, from the ECMWF and national weather models (Open-Meteo), refreshed every 3 hours.</p>} />
      {w && c ? (
        <>
          <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Temperature" value={`${fmt(c.temperature_2m, 1)}°C`} sub={`Feels like ${fmt(c.apparent_temperature, 1)}°C · ${WEATHER_CODES[c.weather_code] ?? ""}`} />
            <Stat label="Humidity" value={`${c.relative_humidity_2m}%`} sub={`Rain now ${fmt(c.precipitation, 1)} mm`} />
            <Stat label="Wind" value={`${fmt(c.wind_speed_10m)} km/h`} sub={`from the ${compass(c.wind_direction_10m)}`} />
            <Stat label="Today" value={`${fmt(w.daily.temperature_2m_min[t0])}° / ${fmt(w.daily.temperature_2m_max[t0])}°`} sub={`Sunrise ${w.daily.sunrise[t0].slice(11)} · sunset ${w.daily.sunset[t0].slice(11)}`} />
          </div>
          {next48 && (
            <Section title="Temperature, next 48 hours" note="°C; solid line past hours, dashed line forecast.">
              <HourlyChart times={next48.t} values={next48.v} now={c.time} unit="°C" label={`Hourly temperature in ${place}`} />
            </Section>
          )}
          <Section title="7-day forecast">
            <div className="overflow-x-auto">
              <table className="data">
                <thead><tr><th>Day</th><th>Weather</th><th className="num">Min °C</th><th className="num">Max °C</th><th className="num">Rain (mm)</th><th className="num">Chance of rain</th><th className="num">UV index</th></tr></thead>
                <tbody>{w.daily.time.map((day, i) => i < t0 ? null : (
                  <tr key={day}><td>{nptDate(`${day}T12:00:00+05:45`, { weekday: "short", year: undefined })}</td><td>{WEATHER_CODES[w.daily.weather_code[i]] ?? ""}</td>
                    <td className="num">{fmt(w.daily.temperature_2m_min[i])}</td><td className="num">{fmt(w.daily.temperature_2m_max[i])}</td>
                    <td className="num">{fmt(w.daily.precipitation_sum[i], 1)}</td><td className="num">{fmt(w.daily.precipitation_probability_max[i])}%</td><td className="num">{fmt(w.daily.uv_index_max[i], 1)}</td></tr>
                ))}</tbody>
              </table>
            </div>
          </Section>
          {past.length > 0 && (
            <Section title="The last 30 days" note={`Daily rain (bars); ${fmt(pastRain)} mm in total. Highest temperature ${fmt(Math.max(...past.map((p) => p.max)), 1)}°C, lowest ${fmt(Math.min(...past.map((p) => p.min)), 1)}°C.`}>
              <BarChart data={past.map((p) => ({ x: p.day.slice(5), v: p.rain ?? 0 }))} unit="mm" label={`Daily rain in ${place} over the last 30 days`} />
            </Section>
          )}
        </>
      ) : <p className="mb-10 text-muted">The weather service is not responding right now; please check again in a few minutes.</p>}
      {faq.length > 0 && (
        <Section title={`Questions about the weather in ${place}`}>
          <dl className="space-y-4">{faq.map((f) => <div key={f.q}><dt className="font-medium">{f.q}</dt><dd className="mt-1 text-muted">{f.a}</dd></div>)}</dl>
          <JsonLd data={{ "@context": "https://schema.org", "@type": "FAQPage", mainEntity: faq.map((f) => ({ "@type": "Question", name: f.q, acceptedAnswer: { "@type": "Answer", text: f.a } })) }} />
        </Section>
      )}
      <p className="mb-6"><Link href={`/air-quality/${d.slug}`}>Air quality in {place}</Link> · <Link href={`/districts/${d.slug}`}>{areaTitle(d)}: land cover and map</Link></p>
      <Source>Open-Meteo (ECMWF IFS, DWD ICON, NOAA GFS). Model forecast for the town, not an official warning; for warnings see the Department of Hydrology and Meteorology (mfd.gov.np).</Source>
    </>
  );
}
