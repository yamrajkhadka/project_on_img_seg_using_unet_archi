"use client";
/** Choose where to run the model: a well-known place, any province/district/local unit, or coordinates. */
import { useMemo, useState } from "react";

export interface PlaceOption {
  label: string;
  lon: number;
  lat: number;
  km?: number;
}
export interface Picked {
  label: string;
  lon: number;
  lat: number;
  km: number;
}

export const FAMOUS: PlaceOption[] = [
  { label: "Kathmandu core (Durbar Square → Thamel)", lat: 27.709, lon: 85.311, km: 2 },
  { label: "Bhaktapur, Kathmandu Valley east", lat: 27.6722, lon: 85.428, km: 2.5 },
  { label: "Pokhara & Phewa Lake", lat: 28.21, lon: 83.964, km: 3 },
  { label: "Nepalgunj (the paper's Fig. 8)", lat: 28.05, lon: 81.6167, km: 2.5 },
  { label: "Dharan (IoE Purwanchal Campus)", lat: 26.8125, lon: 87.283, km: 2.5 },
  { label: "Biratnagar", lat: 26.455, lon: 87.275, km: 2.5 },
  { label: "Chitwan — Sauraha & Rapti River", lat: 27.578, lon: 84.495, km: 3 },
  { label: "Koshi River barrage", lat: 26.526, lon: 86.935, km: 3 },
  { label: "Lumbini", lat: 27.4833, lon: 83.276, km: 2.5 },
  { label: "Janakpur", lat: 26.7288, lon: 85.9263, km: 2.5 },
  { label: "Melamchi Bazaar (2021 flood)", lat: 27.833, lon: 85.577, km: 2 },
  { label: "Rara Lake", lat: 29.526, lon: 82.087, km: 3 },
  { label: "Lo Manthang, Upper Mustang", lat: 29.183, lon: 83.958, km: 3 },
  { label: "Khumbu Glacier & Everest Base Camp", lat: 27.987, lon: 86.835, km: 3 },
];

export function PlacePicker({ regions, value, onChange, maxKm = 3 }: {
  regions: PlaceOption[]; value: Picked; onChange: (p: Picked) => void; maxKm?: number;
}) {
  const [mode, setMode] = useState<"famous" | "region" | "coords">("famous");
  const [query, setQuery] = useState("");
  const [lat, setLat] = useState(String(value.lat));
  const [lon, setLon] = useState(String(value.lon));
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q.length < 2 ? [] : regions.filter((r) => r.label.toLowerCase().includes(q)).slice(0, 12);
  }, [query, regions]);
  const tab = (m: typeof mode, label: string) => (
    <button type="button" onClick={() => setMode(m)} className={`rounded-full border px-3 py-1 text-sm ${mode === m ? "border-accent text-accent" : "border-line text-muted"}`}>{label}</button>
  );
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">{tab("famous", "Well-known places")}{tab("region", "Any district or municipality")}{tab("coords", "Coordinates")}</div>
      {mode === "famous" && (
        <select className="w-full rounded-lg border border-line bg-card p-2" value={value.label}
          onChange={(e) => { const p = FAMOUS.find((f) => f.label === e.target.value)!; onChange({ label: p.label, lon: p.lon, lat: p.lat, km: Math.min(maxKm, p.km ?? 2) }); }}>
          {!FAMOUS.some((f) => f.label === value.label) && <option>{value.label}</option>}
          {FAMOUS.map((f) => <option key={f.label}>{f.label}</option>)}
        </select>
      )}
      {mode === "region" && (
        <div>
          <input className="w-full rounded-lg border border-line bg-card p-2" placeholder="Type a name, e.g. Pokhara, Ilam, Bharatpur…" value={query} onChange={(e) => setQuery(e.target.value)} />
          {matches.length > 0 && (
            <ul className="mt-1 max-h-56 overflow-auto rounded-lg border border-line bg-card">
              {matches.map((m) => (
                <li key={m.label}><button type="button" className="w-full px-3 py-1.5 text-left hover:bg-[var(--bg)]"
                  onClick={() => { onChange({ label: m.label, lon: m.lon, lat: m.lat, km: value.km }); setQuery(""); }}>{m.label}</button></li>
              ))}
            </ul>
          )}
          <p className="mt-1 text-xs text-muted">Centres the analysis on the area&apos;s centre point.</p>
        </div>
      )}
      {mode === "coords" && (
        <div className="flex flex-wrap items-end gap-2">
          <label className="text-sm">Latitude<input className="mt-1 block w-32 rounded-lg border border-line bg-card p-2" value={lat} onChange={(e) => setLat(e.target.value)} /></label>
          <label className="text-sm">Longitude<input className="mt-1 block w-32 rounded-lg border border-line bg-card p-2" value={lon} onChange={(e) => setLon(e.target.value)} /></label>
          <button type="button" className="rounded-lg border border-line px-3 py-2 text-sm"
            onClick={() => { const a = Number(lat), o = Number(lon); if (a > 26 && a < 31 && o > 79.8 && o < 88.5) onChange({ label: `${a.toFixed(4)}, ${o.toFixed(4)}`, lat: a, lon: o, km: value.km }); }}>Use</button>
        </div>
      )}
      <label className="block text-sm">Area: <b>{value.km.toFixed(1)} km × {value.km.toFixed(1)} km</b>
        <input type="range" min={0.6} max={maxKm} step={0.1} value={value.km} className="mt-1 block w-full"
          onChange={(e) => onChange({ ...value, km: Number(e.target.value) })} />
      </label>
      <p className="text-sm text-muted">Selected: {value.label} ({value.lat.toFixed(4)}, {value.lon.toFixed(4)})</p>
    </div>
  );
}
