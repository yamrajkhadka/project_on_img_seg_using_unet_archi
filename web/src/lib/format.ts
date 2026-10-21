/** Number, date and category formatting shared by every page. */
export const TZ = "Asia/Kathmandu";

export function fmt(v: number | null | undefined, digits = 0): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function pct(v: number | null | undefined, digits = 1): string {
  return v == null ? "—" : `${fmt(v, digits)}%`;
}

export function km2(v: number): string {
  return `${fmt(v, v >= 100 ? 0 : 1)} km²`;
}

export function nptTime(iso: string | number | Date, opts: Intl.DateTimeFormatOptions = {}): string {
  return new Date(iso).toLocaleString("en-GB", { timeZone: TZ, hour: "2-digit", minute: "2-digit", ...opts });
}

export function nptDate(iso: string | number | Date, opts: Intl.DateTimeFormatOptions = {}): string {
  return new Date(iso).toLocaleDateString("en-GB", { timeZone: TZ, day: "numeric", month: "short", year: "numeric", ...opts });
}

export function ago(iso: string | number | Date): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  const d = Math.round(s / 86400);
  return d === 1 ? "1 day ago" : `${d} days ago`;
}

/** US EPA AQI categories. */
export const AQI_BANDS = [
  { max: 50, label: "Good", tone: "good", advice: "Air quality is satisfactory; enjoy outdoor activities." },
  { max: 100, label: "Moderate", tone: "warn", advice: "Unusually sensitive people should consider limiting long outdoor exertion." },
  { max: 150, label: "Unhealthy for sensitive groups", tone: "warn", advice: "Children, older adults and people with heart or lung disease should reduce long or heavy outdoor exertion." },
  { max: 200, label: "Unhealthy", tone: "bad", advice: "Everyone should reduce long or heavy outdoor exertion; sensitive groups should avoid it. Consider an N95 mask outdoors." },
  { max: 300, label: "Very unhealthy", tone: "bad", advice: "Everyone should avoid long or heavy outdoor exertion; sensitive groups should stay indoors." },
  { max: Infinity, label: "Hazardous", tone: "bad", advice: "Health warning of emergency conditions: everyone should stay indoors and keep activity low." },
] as const;

export function aqiBand(aqi: number | null | undefined) {
  if (aqi == null) return { max: 0, label: "No data", tone: "muted", advice: "" } as const;
  return AQI_BANDS.find((b) => aqi <= b.max) ?? AQI_BANDS[AQI_BANDS.length - 1];
}

export const WEATHER_CODES: Record<number, string> = {
  0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Rime fog",
  51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle",
  61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
  71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Light showers", 81: "Showers",
  82: "Violent showers", 85: "Snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
  96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
};


/** The current time in milliseconds (kept out of components so rendering stays pure). */
export function nowMs(): number {
  return Date.now();
}

/** Today's date in Nepal as YYYY-MM-DD. */
export function todayNpt(): string {
  return new Date(Date.now() + 5.75 * 3600_000).toISOString().slice(0, 10);
}

/** Day of the year (1–365) of a YYYY-MM-DD date. */
export function dayOfYear(day: string): number {
  return Math.min(365, Math.floor((Date.parse(day) - Date.UTC(+day.slice(0, 4), 0, 0)) / 86400_000));
}
