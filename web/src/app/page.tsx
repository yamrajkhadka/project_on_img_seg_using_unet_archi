import Link from "next/link";
import { AlertList } from "@/components/Alerts";
import { ShareBar } from "@/components/charts";
import { RegionMap } from "@/components/RegionMap";
import { Badge, Card, LinkGrid, PageHeader, Section, Source, Stat } from "@/components/ui";
import { CLASS_KEYS, landCoverOf, placeLabel, regionsAt, rivers, townOf } from "@/lib/data";
import { ago, aqiBand, dayOfYear, fmt, nowMs, nptDate, nptTime, todayNpt, WEATHER_CODES } from "@/lib/format";
import { computeAlerts } from "@/lib/alerts";
import { airMany, fires, quakes, riverFlow, weather } from "@/lib/live";
import { siteConfig } from "@/lib/site";

export const revalidate = 600;

export default async function Home() {
  const districts = regionsAt(2);
  const pts = districts.map((d) => townOf(d.code)!);
  const ktm = districts.find((d) => d.slug === "kathmandu")!;
  const [aq, fire24, quake30, ktmWx, flows, alerts] = await Promise.all([
    airMany(pts, 1800), fires("24h"), quakes(30), weather(townOf(ktm.code)!, 1800),
    Promise.all(rivers.rivers.map((r) => riverFlow(r))), computeAlerts(),
  ]);
  const aqi = districts.map((d, i) => ({ d, aqi: aq?.[i]?.current?.us_aqi ?? null, pm: aq?.[i]?.current?.pm2_5 ?? null }))
    .filter((x) => x.aqi != null).sort((a, b) => (b.aqi ?? 0) - (a.aqi ?? 0));
  const ktmAqi = aqi.find((x) => x.d.code === ktm.code);
  const week = nowMs() - 7 * 86400_000;
  const quakesWeek = quake30?.filter((q) => q.time >= week) ?? null;
  const nat = landCoverOf("NP");
  const share = nat ? Object.fromEntries(CLASS_KEYS.map((k) => [k, nat.latest[k][0]])) : {};
  const today = todayNpt();
  const doy = dayOfYear(today);

  return (
    <>
      <PageHeader eyebrow={`Live · ${nptDate(new Date())}`} title="Nepal right now"
        intro={<p>Air quality, weather, earthquakes, forest fires and river flows across Nepal, updated every few minutes, and an AI map of the country&apos;s land cover. Pick a <Link href="/districts">district</Link> or a <Link href="/municipalities">municipality</Link> for local detail.</p>} />

      {alerts.length > 0 && <div className="mb-8"><AlertList alerts={alerts} /></div>}

      <div className="mb-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Air quality, Kathmandu (US AQI)" value={fmt(ktmAqi?.aqi)} tone={aqiBand(ktmAqi?.aqi).tone}
          sub={<>{aqiBand(ktmAqi?.aqi).label} · PM2.5 {fmt(ktmAqi?.pm, 0)} µg/m³</>} />
        <Stat label="Kathmandu now" value={ktmWx ? `${fmt(ktmWx.current.temperature_2m)}°C` : "—"}
          sub={ktmWx ? `${WEATHER_CODES[ktmWx.current.weather_code] ?? ""} · humidity ${ktmWx.current.relative_humidity_2m}%` : "Weather service unavailable"} />
        <Stat label="Active fires, last 24 h" value={fire24 ? fmt(fire24.length) : "—"} sub={<Link href="/fires">Fire map and history</Link>} />
        <Stat label="Earthquakes M2.5+, last 7 days" value={quakesWeek ? fmt(quakesWeek.length) : "—"} sub={<Link href="/earthquakes">Latest earthquakes</Link>} />
      </div>

      <Section title="Air quality now: most polluted districts" note={<>US AQI from the CAMS forecast at each district&apos;s main town. <Link href="/air-quality">All 77 districts</Link>.</>}>
        {aqi.length ? (
          <div className="overflow-x-auto">
            <table className="data">
              <thead><tr><th>District</th><th className="num">US AQI</th><th>Category</th><th className="num">PM2.5 (µg/m³)</th></tr></thead>
              <tbody>
                {aqi.slice(0, 10).map(({ d, aqi: a, pm }) => (
                  <tr key={d.code}>
                    <td><Link href={`/air-quality/${d.slug}`}>{placeLabel(d)}</Link></td>
                    <td className="num">{fmt(a)}</td>
                    <td><Badge tone={aqiBand(a).tone}>{aqiBand(a).label}</Badge></td>
                    <td className="num">{fmt(pm, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <p className="text-muted">The air-quality service is not responding right now.</p>}
      </Section>

      <Section title="Fires and earthquakes" note={<>Red: satellite fire detections in the last 24 hours (NASA FIRMS VIIRS). Purple: earthquakes M2.5+ in the last 30 days (USGS).</>}>
        <RegionMap label="Map of Nepal with active fires and recent earthquakes" contextLevel={1} context={regionsAt(1).map((p) => p.code)}
          points={[
            ...(fire24 ?? []).map((f) => ({ lon: f.lon, lat: f.lat, r: 2.5, color: "var(--crimson)" })),
            ...(quake30 ?? []).map((q) => ({ lon: q.lon, lat: q.lat, r: 2 + q.mag * 1.2, color: "#6250d6" })),
          ]} />
        {quake30 && quake30.length > 0 && (
          <ul className="mt-4 space-y-1 text-sm">
            {quake30.slice(0, 5).map((q) => (
              <li key={q.id}><b>M{q.mag.toFixed(1)}</b> · {q.place} · {nptDate(q.time)} {nptTime(q.time)} NPT · {fmt(q.depth)} km deep · {ago(q.time)}</li>
            ))}
          </ul>
        )}
      </Section>

      <Section title="Rivers today" note={<>Daily discharge from the GloFAS flood model at each river&apos;s main gauge, compared with the 2004–2023 median for the date. <Link href="/rivers">Flood levels and forecasts</Link>.</>}>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {rivers.rivers.map((r, i) => {
            const f = flows[i];
            const idx = f ? f.daily.time.indexOf(today) : -1;
            const q = idx >= 0 ? f!.daily.river_discharge[idx] : null;
            const normal = rivers.climatology[r.key]?.find((c) => c[0] === doy)?.[2];
            const thr = rivers.thresholds[r.key];
            const tone = q != null && thr && q >= thr.q10 ? "bad" : q != null && thr && q >= thr.q2 ? "warn" : undefined;
            return (
              <Card key={r.key}>
                <div className="text-sm text-muted"><Link href={`/rivers/${r.key}`}>{r.name}</Link> at {r.gauge}</div>
                <div className="mt-1 text-2xl font-semibold tabular-nums" style={tone ? { color: `var(--${tone})` } : undefined}>{q != null ? `${fmt(q)} m³/s` : "—"}</div>
                <div className="text-sm text-muted">{q != null && normal ? `${fmt((q / normal) * 100)}% of normal for the date` : "No data"}</div>
              </Card>
            );
          })}
        </div>
      </Section>

      <Section title="Nepal from space" note={<>NASA MODIS Terra true-colour image of {nptDate(new Date(nowMs() - 86400_000))} (about 10:30 local time). Clouds, snow on the Himalaya and smoke or haze over the plains are visible. <Link href="/time-machine">Any date since 2000</Link>.</>}>
        <div className="grid max-w-2xl grid-cols-2 overflow-hidden rounded-xl border border-line">
          {[[46, 26], [47, 26], [46, 27], [47, 27]].map(([x, y]) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img key={`${x}-${y}`} alt={x === 46 && y === 26 ? "Satellite image of Nepal and the Himalaya" : ""} loading="lazy" className="block w-full"
              src={`https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/${new Date(nowMs() - 86400_000).toISOString().slice(0, 10)}/GoogleMapsCompatible_Level9/6/${y}/${x}.jpg`} />
          ))}
        </div>
      </Section>

      {nat && (
        <Section title="Nepal's land cover, mapped by AI" note={<>{nat.imagery}; {nat.n} stratified sample sites classified by the Nepal-adapted U-Net. <Link href="/land-cover">Year by year and by region</Link>.</>}>
          <ShareBar share={share} label="Share of Nepal's area by land-cover class" />
        </Section>
      )}

      <Section title="Explore Nepal">
        <LinkGrid links={[
          { href: "/map", label: "AI land-cover map of any place", sub: "runs in your browser" },
          { href: "/change", label: "Detect land-cover change", sub: "compare two years" },
          { href: "/model-lab", label: "Classify your own satellite image", sub: "the paper's demo" },
          { href: "/time-machine", label: "Nepal on any date", sub: "weather, quakes, rivers since 1940" },
          { href: "/alerts", label: "Alerts", sub: "air, fires, quakes, rain, floods" },
          { href: "/air-quality", label: "Air quality in every district", sub: "AQI, PM2.5, forecast" },
          { href: "/weather", label: "Weather in every district", sub: "now and 7 days" },
          { href: "/earthquakes", label: "Earthquakes in Nepal", sub: "live and history" },
          { href: "/fires", label: "Forest fires", sub: "satellite detections" },
          { href: "/rivers", label: "River levels and floods", sub: "6 major rivers" },
          { href: "/land-cover", label: "Land cover of Nepal", sub: "forest, farmland, built-up" },
          { href: "/provinces", label: "7 provinces" },
          { href: "/districts", label: "77 districts" },
          { href: "/municipalities", label: "Every municipality" },
          { href: "/climate", label: "Climate since 1901" },
          { href: "/history", label: "Nepal since 1768" },
          { href: "/research", label: "The research and the model" },
        ]} />
      </Section>
      <Source>NASA FIRMS, USGS, Open-Meteo (CAMS, ECMWF, GloFAS), Esri, EOX, Impact Observatory. Interactive maps and analysis: <a href={siteConfig.appUrl}>Monitor Nepal app</a>.</Source>
    </>
  );
}
