/** Live alerts, with the same rules as the Streamlit app (monitor_nepal/live/alerts.py). */
import { placeLabel, regionsAt, rivers, townOf } from "./data";
import { fmt, nowMs } from "./format";
import { airMany, fires, quakes, rainMany, riverFlow } from "./live";
import { regionAt } from "./shapes";

export interface Alert {
  severity: "critical" | "warning";
  topic: string;
  title: string;
  detail: string;
  href: string;
}

const HEAVY_RAIN_MM = 64.5;
const VERY_HEAVY_RAIN_MM = 115.6;

export async function computeAlerts(): Promise<Alert[]> {
  const districts = regionsAt(2);
  const towns = districts.map((d) => townOf(d.code)!);
  const [aq, week, q30, rain, flows] = await Promise.all([
    airMany(towns), fires("7d"), quakes(30), rainMany(towns), Promise.all(rivers.rivers.map((r) => riverFlow(r))),
  ]);
  const out: Alert[] = [];
  const now = nowMs();

  if (aq) {
    const bad = districts.map((d, i) => ({ d, v: aq[i]?.current?.us_aqi })).filter((x) => (x.v ?? 0) > 150).sort((a, b) => (b.v ?? 0) - (a.v ?? 0));
    if (bad.length) out.push({ severity: (bad[0].v ?? 0) > 200 ? "critical" : "warning", topic: "Air quality",
      title: `Unhealthy air in ${bad.length} district${bad.length > 1 ? "s" : ""}`,
      detail: bad.slice(0, 5).map((x) => `${placeLabel(x.d)} ${fmt(x.v)}`).join(", ") + " (US AQI)", href: "/air-quality" });
  }
  if (week) {
    const last = week.filter((f) => Date.parse(f.time) >= now - 86400_000);
    const perDay = (week.length - last.length) / 6;
    if (last.length >= 20 && last.length > 2 * Math.max(perDay, 1)) {
      const counts = new Map<string, number>();
      for (const f of last) { const c = regionAt(1, f.lon, f.lat); if (c) counts.set(c, (counts.get(c) ?? 0) + 1); }
      const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
      const prov = top ? regionsAt(1).find((p) => p.code === top[0])?.name : undefined;
      out.push({ severity: "warning", topic: "Fires", title: `Fire activity ${fmt(last.length / Math.max(perDay, 1), 1)}× the past week's average`,
        detail: `${last.length} detections in 24 hours${prov ? `, most in ${prov} Province` : ""}.`, href: "/fires" });
    }
  }
  if (q30) {
    const strong = q30.filter((q) => q.time >= now - 7 * 86400_000 && q.mag >= 4.5).sort((a, b) => b.mag - a.mag);
    if (strong.length) out.push({ severity: strong[0].mag >= 5.5 ? "critical" : "warning", topic: "Earthquake",
      title: `M${strong[0].mag.toFixed(1)} earthquake`, detail: `${strong[0].place}, ${new Date(strong[0].time).toLocaleDateString("en-GB", { timeZone: "Asia/Kathmandu", day: "numeric", month: "short" })}.`, href: "/earthquakes" });
  }
  if (rain) {
    const heavy = districts.map((d, i) => ({ d, mm: Math.max(...(rain[i]?.daily.precipitation_sum ?? [0]).map((v) => v ?? 0)) }))
      .filter((x) => x.mm >= HEAVY_RAIN_MM).sort((a, b) => b.mm - a.mm);
    if (heavy.length) out.push({ severity: heavy[0].mm >= VERY_HEAVY_RAIN_MM ? "critical" : "warning", topic: "Heavy rain",
      title: `${heavy[0].mm >= VERY_HEAVY_RAIN_MM ? "Very heavy" : "Heavy"} rain forecast in ${heavy.length} district${heavy.length > 1 ? "s" : ""}`,
      detail: heavy.slice(0, 5).map((x) => `${x.d.name} ${fmt(x.mm)} mm`).join(", ") + " in a day, next 3 days.", href: `/weather/${heavy[0].d.slug}` });
  }
  rivers.rivers.forEach((r, i) => {
    const f = flows[i], thr = rivers.thresholds[r.key];
    if (!f || !thr) return;
    const today = new Date(now + 5.75 * 3600_000).toISOString().slice(0, 10);
    const idx = f.daily.time.indexOf(today);
    const ahead = idx >= 0 ? f.daily.river_discharge.slice(idx) : [];
    const peak = ahead.length ? Math.max(...ahead) : 0;
    if (peak >= thr.q2) out.push({ severity: peak >= thr.q10 ? "critical" : "warning", topic: "River flood watch",
      title: `${r.name} ${peak >= thr.q10 ? "above the 10-year" : "above the 2-year"} flood level`,
      detail: `Up to ${fmt(peak)} m³/s at ${r.gauge} in the next 7 days (2-year level ${fmt(thr.q2)} m³/s).`, href: `/rivers/${r.key}` });
  });
  return out.sort((a, b) => (a.severity === b.severity ? 0 : a.severity === "critical" ? -1 : 1));
}
