"use client";
/** Any date in Nepal's recent past: weather, air quality, earthquakes, river flows and the NASA satellite image of
 *  that day (or a whole month), fetched in the browser from the public archives. */
import { useState } from "react";

export interface TMPlace { label: string; lat: number; lon: number }
export interface TMRiver { key: string; name: string; gauge: string; lat: number; lon: number; median: number[]; q2: number; q10: number }

const fmt = (v: number | null | undefined, d = 0) => (v == null || Number.isNaN(v) ? "—" : v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
const iso = (d: Date) => d.toISOString().slice(0, 10);
const daysAgo = (day: string, today: string) => (Date.parse(today) - Date.parse(day)) / 86400_000;

async function json<T>(url: string): Promise<T | null> {
  try { const r = await fetch(url); return r.ok ? ((await r.json()) as T) : null; } catch { return null; }
}

interface Daily { time: string[]; temperature_2m_max: number[]; temperature_2m_min: number[]; precipitation_sum: number[] }
interface Hourly { time: string[]; temperature_2m: number[]; precipitation: number[] }
interface QuakeFeature { id: string; properties: { mag: number; place: string; time: number; url: string }; geometry: { coordinates: number[] } }

export function TimeMachine({ places, rivers, yesterday }: { places: TMPlace[]; rivers: TMRiver[]; yesterday: string }) {
  const [place, setPlaceRaw] = useState(places[0]);
  const [scale, setScaleRaw] = useState<"day" | "month">("day");
  const [day, setDayRaw] = useState(yesterday);
  const [busy, setBusy] = useState(false);
  const [data, setData] = useState<{ daily?: Daily | null; hourly?: Hourly | null; aqi?: { time: string[]; us_aqi: number[] } | null; quakes?: QuakeFeature[] | null; flows?: (number | null)[][]; start: string; end: string } | null>(null);

  const start = scale === "day" ? day : `${day.slice(0, 7)}-01`;
  const endD = scale === "day" ? day : iso(new Date(Date.UTC(+day.slice(0, 4), +day.slice(5, 7), 0)));
  const end = endD > yesterday ? yesterday : endD;

  async function load() {
    setBusy(true);
    const recent = daysAgo(start, yesterday) < 80;
    const wxBase = recent ? "https://api.open-meteo.com/v1/forecast" : "https://archive-api.open-meteo.com/v1/archive";
    const loc = `latitude=${place.lat}&longitude=${place.lon}&timezone=Asia%2FKathmandu&start_date=${start}&end_date=${end}`;
    const [wx, aq, qk, ...flows] = await Promise.all([
      json<{ daily: Daily; hourly: Hourly }>(`${wxBase}?${loc}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&hourly=temperature_2m,precipitation`),
      json<{ hourly: { time: string[]; us_aqi: number[] } }>(`https://air-quality-api.open-meteo.com/v1/air-quality?${loc}&hourly=us_aqi`),
      json<{ features: QuakeFeature[] }>(`https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=${start}&endtime=${iso(new Date(Date.parse(end) + 86400_000))}&minmagnitude=2.5&minlatitude=25.4&maxlatitude=31.3&minlongitude=79.2&maxlongitude=89.2&orderby=time`),
      ...rivers.map((r) => json<{ daily: { time: string[]; river_discharge: number[] } }>(`https://flood-api.open-meteo.com/v1/flood?latitude=${r.lat}&longitude=${r.lon}&daily=river_discharge&start_date=${start}&end_date=${end}`)),
    ]);
    setData({ daily: wx?.daily ?? null, hourly: wx?.hourly ?? null, aqi: aq?.hourly ?? null, quakes: qk?.features ?? null,
      flows: flows.map((f) => f?.daily.river_discharge ?? []), start, end });
    setBusy(false);
  }

  // a new choice clears the previous answer
  const setPlace = (p: TMPlace) => { setPlaceRaw(p); setData(null); };
  const setScale = (s: "day" | "month") => { setScaleRaw(s); setData(null); };
  const setDay = (d: string) => { setDayRaw(d); setData(null); };

  const doy = (d: string) => Math.min(365, Math.floor((Date.parse(d) - Date.UTC(+d.slice(0, 4), 0, 0)) / 86400_000));
  const aqiVals = data?.aqi?.us_aqi.filter((v) => v != null) ?? [];
  const gibsDay = start;
  const nasaOk = start >= "2000-03-01";
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end gap-4 rounded-xl border border-line bg-card p-4">
        <label className="text-sm">Place
          <select className="mt-1 block rounded-lg border border-line bg-card p-2" value={place.label} onChange={(e) => setPlace(places.find((p) => p.label === e.target.value)!)}>
            {places.map((p) => <option key={p.label}>{p.label}</option>)}
          </select>
        </label>
        <fieldset className="text-sm"><legend>Time scale</legend>
          <div className="mt-1 flex gap-2">
            {(["day", "month"] as const).map((s) => <button key={s} type="button" onClick={() => setScale(s)} className={`rounded-full border px-3 py-1 ${scale === s ? "border-accent text-accent" : "border-line text-muted"}`}>{s === "day" ? "A day" : "A month"}</button>)}
          </div>
        </fieldset>
        <label className="text-sm">{scale === "day" ? "Date" : "Month"}
          {scale === "day"
            ? <input type="date" min="1940-01-01" max={yesterday} value={day} onChange={(e) => e.target.value && setDay(e.target.value)} className="mt-1 block rounded-lg border border-line bg-card p-2" />
            : <input type="month" min="1940-01" max={yesterday.slice(0, 7)} value={day.slice(0, 7)} onChange={(e) => e.target.value && setDay(`${e.target.value}-01`)} className="mt-1 block rounded-lg border border-line bg-card p-2" />}
        </label>
        <button type="button" onClick={load} disabled={busy} className="rounded-lg bg-[var(--accent)] px-4 py-2 font-medium text-white disabled:opacity-60">{busy ? "Loading…" : "Show"}</button>
      </div>

      {data && (
        <div className="space-y-8">
          <section>
            <h2 className="mb-2 text-xl font-semibold">Weather in {place.label}</h2>
            {data.daily && data.daily.time.length ? (
              scale === "day" ? (
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="rounded-xl border border-line bg-card p-4"><div className="text-sm text-muted">High / low</div><div className="text-2xl font-semibold">{fmt(data.daily.temperature_2m_max[0], 1)}° / {fmt(data.daily.temperature_2m_min[0], 1)}°C</div></div>
                  <div className="rounded-xl border border-line bg-card p-4"><div className="text-sm text-muted">Rain</div><div className="text-2xl font-semibold">{fmt(data.daily.precipitation_sum[0], 1)} mm</div></div>
                  <div className="rounded-xl border border-line bg-card p-4"><div className="text-sm text-muted">Air quality (US AQI, daily range)</div><div className="text-2xl font-semibold">{aqiVals.length ? `${fmt(Math.min(...aqiVals))}–${fmt(Math.max(...aqiVals))}` : "not available"}</div></div>
                </div>
              ) : (
                <div className="overflow-x-auto"><table className="data">
                  <thead><tr><th>Day</th><th className="num">High °C</th><th className="num">Low °C</th><th className="num">Rain (mm)</th></tr></thead>
                  <tbody>{data.daily.time.map((t, i) => <tr key={t}><td>{t}</td><td className="num">{fmt(data.daily!.temperature_2m_max[i], 1)}</td><td className="num">{fmt(data.daily!.temperature_2m_min[i], 1)}</td><td className="num">{fmt(data.daily!.precipitation_sum[i], 1)}</td></tr>)}</tbody>
                  <tfoot><tr><th>Month</th><th className="num">{fmt(Math.max(...data.daily.temperature_2m_max), 1)}</th><th className="num">{fmt(Math.min(...data.daily.temperature_2m_min), 1)}</th><th className="num">{fmt(data.daily.precipitation_sum.reduce((s, v) => s + (v ?? 0), 0), 0)} total</th></tr></tfoot>
                </table></div>
              )
            ) : <p className="text-muted">No weather record for this date.</p>}
            <p className="mt-1 text-xs text-muted">ERA5 reanalysis and weather models via Open-Meteo, at {place.label}.</p>
          </section>

          <section>
            <h2 className="mb-2 text-xl font-semibold">Rivers</h2>
            <div className="overflow-x-auto"><table className="data">
              <thead><tr><th>River</th><th className="num">{scale === "day" ? "Flow (m³/s)" : "Peak (m³/s)"}</th><th className="num">% of normal</th><th>Flood level</th></tr></thead>
              <tbody>{rivers.map((r, i) => {
                const f = (data.flows?.[i] ?? []).filter((v): v is number => v != null);
                const q = f.length ? (scale === "day" ? f[0] : Math.max(...f)) : null;
                const med = r.median[doy(data.start) - 1];
                return <tr key={r.key}><td>{r.name} at {r.gauge}</td><td className="num">{fmt(q)}</td><td className="num">{q != null && med ? `${fmt((q / med) * 100)}%` : "—"}</td>
                  <td>{q == null ? "—" : q >= r.q10 ? "above 10-year" : q >= r.q2 ? "above 2-year" : "normal range"}</td></tr>;
              })}</tbody>
            </table></div>
            <p className="mt-1 text-xs text-muted">GloFAS v4 via Open-Meteo (from 1984); normal = 2004–2023 median for the date.</p>
          </section>

          <section>
            <h2 className="mb-2 text-xl font-semibold">Earthquakes (M2.5+)</h2>
            {data.quakes && data.quakes.length ? (
              <ul className="space-y-1 text-sm">{data.quakes.slice(0, 30).map((q) => <li key={q.id}><b>M{q.properties.mag.toFixed(1)}</b> · <a href={q.properties.url}>{q.properties.place}</a> · {new Date(q.properties.time).toLocaleString("en-GB", { timeZone: "Asia/Kathmandu", dateStyle: "medium", timeStyle: "short" })} NPT</li>)}</ul>
            ) : <p className="text-muted">{data.quakes ? "None recorded in the region." : "The USGS service did not respond."}</p>}
          </section>

          {nasaOk && (
            <section>
              <h2 className="mb-2 text-xl font-semibold">Nepal from space on {gibsDay}</h2>
              <div className="grid max-w-xl grid-cols-2 overflow-hidden rounded-xl">
                {[[46, 26], [47, 26], [46, 27], [47, 27]].map(([x, y]) => (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img key={`${x}-${y}`} alt="" className="block w-full" src={`https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/${gibsDay}/GoogleMapsCompatible_Level9/6/${y}/${x}.jpg`} />
                ))}
              </div>
              <p className="mt-1 text-xs text-muted">NASA MODIS Terra true colour (GIBS), about 10:30 local time; Nepal is the strip between the Indian plains and the Tibetan plateau.</p>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
