/** Live data, fetched on the server and cached for the page's revalidation period. Every fetcher returns null
 *  (never throws) when a source is down, so a page always renders. */
import { insideNepal } from "./shapes";

const FORECAST = "https://api.open-meteo.com/v1/forecast";
const AIR = "https://air-quality-api.open-meteo.com/v1/air-quality";
const FLOOD = "https://flood-api.open-meteo.com/v1/flood";
const QUAKES = "https://earthquake.usgs.gov/fdsnws/event/1/query";
const FIRES = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/{folder}/csv/{prefix}_South_Asia_{window}.csv";
export const NEPAL_BOX = { west: 80.0, south: 26.3, east: 88.3, north: 30.5 };

async function getJson<T>(url: string, params: Record<string, string | number>, revalidate: number): Promise<T | null> {
  const q = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]));
  try {
    const res = await fetch(`${url}?${q}`, { next: { revalidate }, signal: AbortSignal.timeout(15000) });
    return res.ok ? ((await res.json()) as T) : null;
  } catch {
    return null;
  }
}

export interface Point {
  lat: number;
  lon: number;
}

/* ---------------- weather ---------------- */
export interface Weather {
  current: { time: string; temperature_2m: number; apparent_temperature: number; relative_humidity_2m: number;
    precipitation: number; weather_code: number; wind_speed_10m: number; wind_direction_10m: number; is_day: number };
  hourly: { time: string[]; temperature_2m: number[]; precipitation_probability: number[]; precipitation: number[] };
  daily: { time: string[]; weather_code: number[]; temperature_2m_max: number[]; temperature_2m_min: number[];
    precipitation_sum: number[]; precipitation_probability_max: number[]; sunrise: string[]; sunset: string[];
    uv_index_max: number[] };
  elevation: number;
}

export function weather(p: Point, revalidate = 1800) {
  return getJson<Weather>(FORECAST, {
    latitude: p.lat, longitude: p.lon, timezone: "Asia/Kathmandu", forecast_days: 7,
    current: "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,wind_direction_10m,is_day",
    hourly: "temperature_2m,precipitation_probability,precipitation",
    daily: "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,sunrise,sunset,uv_index_max",
  }, revalidate);
}

/** Current conditions for many places in one request. */
export async function weatherMany(points: Point[], revalidate = 1800) {
  const data = await getJson<Weather | Weather[]>(FORECAST, {
    latitude: points.map((p) => p.lat).join(","), longitude: points.map((p) => p.lon).join(","),
    timezone: "Asia/Kathmandu", current: "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
  }, revalidate);
  if (!data) return null;
  return Array.isArray(data) ? data : [data];
}

/* ---------------- air quality ---------------- */
export interface Air {
  current: { time: string; us_aqi: number; pm2_5: number; pm10: number; nitrogen_dioxide: number; ozone: number;
    sulphur_dioxide: number; carbon_monoxide: number };
  hourly: { time: string[]; us_aqi: number[]; pm2_5: number[] };
}

export function air(p: Point, revalidate = 1800) {
  return getJson<Air>(AIR, {
    latitude: p.lat, longitude: p.lon, timezone: "Asia/Kathmandu", past_days: 3, forecast_days: 4,
    current: "us_aqi,pm2_5,pm10,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide",
    hourly: "us_aqi,pm2_5",
  }, revalidate);
}

export async function airMany(points: Point[], revalidate = 1800) {
  const data = await getJson<Air | Air[]>(AIR, {
    latitude: points.map((p) => p.lat).join(","), longitude: points.map((p) => p.lon).join(","),
    timezone: "Asia/Kathmandu", current: "us_aqi,pm2_5,pm10",
  }, revalidate);
  if (!data) return null;
  return Array.isArray(data) ? data : [data];
}

/* ---------------- rivers ---------------- */
export interface Flow {
  daily: { time: string[]; river_discharge: number[] };
}

export function riverFlow(p: Point, revalidate = 3600) {
  return getJson<Flow>(FLOOD, { latitude: p.lat, longitude: p.lon, daily: "river_discharge", past_days: 30, forecast_days: 7 },
    revalidate);
}

/* ---------------- earthquakes ---------------- */
export interface Quake {
  id: string;
  mag: number;
  place: string;
  time: number;
  depth: number;
  lat: number;
  lon: number;
  url: string;
  felt: number | null;
}

/** Earthquakes in and around Nepal (search box 25.4–31.3 N, 79.2–89.2 E). */
export async function quakes(days = 30, minMag = 2.5, revalidate = 600): Promise<Quake[] | null> {
  const start = new Date(Date.now() - days * 86400_000).toISOString().slice(0, 10);
  const data = await getJson<{ features: { id: string; properties: { mag: number; place: string; time: number; url: string; felt: number | null }; geometry: { coordinates: [number, number, number] } }[] }>(
    QUAKES, { format: "geojson", starttime: start, minmagnitude: minMag, minlatitude: 25.4, maxlatitude: 31.3,
      minlongitude: 79.2, maxlongitude: 89.2, orderby: "time" }, revalidate);
  if (!data) return null;
  return data.features.map((f) => ({
    id: f.id, mag: f.properties.mag, place: f.properties.place, time: f.properties.time, url: f.properties.url,
    felt: f.properties.felt, lon: f.geometry.coordinates[0], lat: f.geometry.coordinates[1], depth: f.geometry.coordinates[2],
  }));
}

/* ---------------- fires ---------------- */
export interface Fire {
  lat: number;
  lon: number;
  time: string;
  frp: number;
  confidence: string;
  sensor: string;
}

const FIRE_SENSORS = [
  { name: "VIIRS S-NPP", folder: "suomi-npp-viirs-c2", prefix: "SUOMI_VIIRS_C2" },
  { name: "VIIRS NOAA-20", folder: "noaa-20-viirs-c2", prefix: "J1_VIIRS_C2" },
  { name: "VIIRS NOAA-21", folder: "noaa-21-viirs-c2", prefix: "J2_VIIRS_C2" },
];

/** NASA FIRMS active fires inside Nepal, last 24 h or 7 days (the three VIIRS satellites). */
export async function fires(window: "24h" | "7d" = "24h", revalidate = 1800): Promise<Fire[] | null> {
  const all = await Promise.all(FIRE_SENSORS.map(async (s) => {
    const url = FIRES.replace("{folder}", s.folder).replace("{prefix}", s.prefix).replace("{window}", window);
    try {
      const res = await fetch(url, { next: { revalidate }, signal: AbortSignal.timeout(20000) });
      if (!res.ok) return null;
      const lines = (await res.text()).trim().split("\n");
      const head = lines[0].split(",");
      const col = (n: string) => head.indexOf(n);
      const [iLat, iLon, iDate, iTime, iFrp, iConf] = ["latitude", "longitude", "acq_date", "acq_time", "frp", "confidence"].map(col);
      const out: Fire[] = [];
      for (const line of lines.slice(1)) {
        const c = line.split(",");
        const lat = +c[iLat], lon = +c[iLon];
        if (lat < NEPAL_BOX.south || lat > NEPAL_BOX.north || lon < NEPAL_BOX.west || lon > NEPAL_BOX.east) continue;
        if (!insideNepal(lon, lat)) continue;
        const t = c[iTime].padStart(4, "0");
        out.push({ lat, lon, frp: +c[iFrp], confidence: c[iConf], sensor: s.name,
          time: `${c[iDate]}T${t.slice(0, 2)}:${t.slice(2)}:00Z` });
      }
      return out;
    } catch {
      return null;
    }
  }));
  if (all.every((x) => x === null)) return null;
  return all.flatMap((x) => x ?? []);
}
