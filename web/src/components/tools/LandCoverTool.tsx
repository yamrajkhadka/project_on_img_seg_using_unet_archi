"use client";
/** Run the U-Net on any place in Nepal in the browser: one map ("map") or two dates compared ("change"). */
import { useEffect, useRef, useState } from "react";
import {
  argmax, clean, CLASS_NAMES, CLASS_RGB, composition, EOX_YEARS, eoxYear, esriLatest, fetchGrid, GSD, gridAround, UNRELIABLE_YEARS,
  type Imagery, type ModelKey, paintClasses, paintRGB, predict,
} from "@/lib/client/landcover";
import { FAMOUS, type Picked, PlacePicker, type PlaceOption } from "./PlacePicker";

const IMAGERY: Imagery[] = [esriLatest(), ...[...EOX_YEARS].reverse().map(eoxYear)];
const byKey = (k: string) => IMAGERY.find((i) => i.key === k)!;
const fmt = (v: number, d = 1) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

interface Result { label: string; cls: Uint8Array; valid: Uint8Array; rgb: Uint8ClampedArray; w: number; h: number; conf: number }

async function classify(place: Picked, img: Imagery, model: ModelKey, doClean: boolean, status: (s: string) => void): Promise<Result> {
  const grid = gridAround(place.lon, place.lat, place.km * 1000);
  status(`Downloading ${img.label}…`);
  const { rgb, valid } = await fetchGrid(grid, img, (f) => status(`Downloading ${img.label}… ${Math.round(f * 100)}%`));
  if (valid.reduce((s, v) => s + v, 0) < valid.length * 0.2) throw new Error(`${img.label} has no imagery here.`);
  status("Running the U-Net in your browser (the first run also downloads the 31 MB model)…");
  const probs = await predict(model, rgb, grid.w, grid.h);
  const { cls: raw, conf } = argmax(probs, grid.w * grid.h);
  let cls = doClean ? clean(raw, grid.w, grid.h, 64) : raw;
  cls = cls.map((c, p) => (valid[p] ? c : 6));
  let sum = 0, n = 0;
  for (let p = 0; p < conf.length; p++) if (valid[p]) { sum += conf[p]; n++; }
  return { label: img.label, cls, valid, rgb, w: grid.w, h: grid.h, conf: n ? sum / n : 0 };
}

function Legend({ shares }: { shares?: { k: number; share: number; km2: number }[] }) {
  return (
    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm">
      {CLASS_NAMES.slice(0, 6).map((name, k) => (
        <span key={name} className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm" style={{ background: `rgb(${CLASS_RGB[k].join(",")})` }} />
          {name}{shares && <b className="tabular-nums">{fmt(shares[k].share)}%</b>}
        </span>
      ))}
    </div>
  );
}

export function LandCoverTool({ regions, mode }: { regions: PlaceOption[]; mode: "map" | "change" }) {
  const first = FAMOUS[0];
  const [place, setPlace] = useState<Picked>({ label: first.label, lon: first.lon, lat: first.lat, km: first.km ?? 2 });
  const firstGood = EOX_YEARS.find((y) => !UNRELIABLE_YEARS.has(y)) ?? EOX_YEARS[0];
  const [imgA, setImgA] = useState(mode === "change" ? `eox-${firstGood}` : "esri");
  const [imgB, setImgB] = useState(`eox-${EOX_YEARS[EOX_YEARS.length - 1]}`);
  const [model, setModel] = useState<ModelKey>("nepal");
  const [doClean, setDoClean] = useState(true);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const canvases = useRef<(HTMLCanvasElement | null)[]>([]);

  async function run() {
    setBusy(true); setError(""); setResults([]);
    try {
      const out = [await classify(place, byKey(imgA), model, doClean, setStatus)];
      if (mode === "change") out.push(await classify(place, byKey(imgB), model, doClean, setStatus));
      setResults(out);
      setStatus("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
      setStatus("");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => { // paint once the canvases for the new results exist
    const c = canvases.current;
    if (mode === "map" && results.length === 1 && c[0] && c[1]) {
      paintRGB(c[0], results[0].rgb, results[0].w, results[0].h);
      paintClasses(c[1], results[0].cls, results[0].w, results[0].h);
    } else if (mode === "change" && results.length === 2 && c[0] && c[1] && c[2]) {
      const [a, b] = results;
      const changed = a.cls.map((v, p) => (a.valid[p] && b.valid[p] && v !== b.cls[p] ? 1 : 0));
      paintClasses(c[0], a.cls, a.w, a.h);
      paintClasses(c[1], b.cls, b.w, b.h);
      paintClasses(c[2], b.cls, b.w, b.h, changed);
    }
  }, [results, mode]);

  const comp = results.map((r) => composition(r.cls, r.valid));
  const lowRes = (k: string) => k.startsWith("eox");
  const select = (value: string, set: (v: string) => void) => (
    <select className="rounded-lg border border-line bg-card p-2" value={value} onChange={(e) => set(e.target.value)}>
      {IMAGERY.map((i) => <option key={i.key} value={i.key}>{i.label}</option>)}
    </select>
  );
  const download = () => {
    const c = canvases.current[mode === "map" ? 1 : 2];
    if (!c) return;
    const a = document.createElement("a");
    a.href = c.toDataURL("image/png");
    a.download = `nepal-monitor-${mode}-${place.label.replace(/[^a-z0-9]+/gi, "-").toLowerCase()}.png`;
    a.click();
  };

  let transitions: { from: number; to: number; km2: number }[] = [];
  if (mode === "change" && results.length === 2) {
    const [a, b] = results, m = new Map<string, number>();
    for (let p = 0; p < a.cls.length; p++) if (a.valid[p] && b.valid[p] && a.cls[p] !== b.cls[p]) {
      const key = `${a.cls[p]}-${b.cls[p]}`; m.set(key, (m.get(key) ?? 0) + 1);
    }
    transitions = [...m.entries()].map(([k, n]) => ({ from: +k.split("-")[0], to: +k.split("-")[1], km2: (n * GSD * GSD) / 1e6 }))
      .sort((x, y) => y.km2 - x.km2).slice(0, 10);
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6 rounded-xl border border-line bg-card p-4 lg:grid-cols-2">
        <PlacePicker regions={regions} value={place} onChange={setPlace} />
        <div className="space-y-3">
          <label className="block text-sm">{mode === "change" ? "Earlier imagery" : "Imagery"}<div className="mt-1">{select(imgA, setImgA)}</div></label>
          {mode === "change" && <label className="block text-sm">Later imagery<div className="mt-1">{select(imgB, setImgB)}</div></label>}
          <label className="block text-sm">Model
            <select className="mt-1 block rounded-lg border border-line bg-card p-2" value={model} onChange={(e) => setModel(e.target.value as ModelKey)}>
              <option value="nepal">Nepal-adapted U-Net (recommended)</option>
              <option value="paper">Original U-Net from the paper (sub-metre imagery only)</option>
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={doClean} onChange={(e) => setDoClean(e.target.checked)} /> Remove specks smaller than 64 pixels (0.19 ha)</label>
          {[imgA, mode === "change" ? imgB : ""].some((k) => UNRELIABLE_YEARS.has(Number(k.split("-")[1]))) && (
            <p className="text-sm" style={{ color: "var(--warn)" }}>The 2016 and 2018 mosaics were processed differently and the model misreads them (built-up areas can come out as barren land or water). For change, compare 2019 or later.</p>
          )}
          {model === "paper" && (lowRes(imgA) || (mode === "change" && lowRes(imgB))) && (
            <p className="text-sm" style={{ color: "var(--warn)" }}>The original model was trained on sub-metre imagery and is unreliable on 10 m Sentinel-2; use the Nepal-adapted model.</p>
          )}
          <button type="button" disabled={busy} onClick={run} className="rounded-lg bg-[var(--accent)] px-4 py-2 font-medium text-white disabled:opacity-60">
            {busy ? "Working…" : mode === "change" ? "Detect change" : "Map land cover"}
          </button>
          {status && <p className="text-sm text-muted" aria-live="polite">{status}</p>}
          {error && <p className="text-sm" style={{ color: "var(--bad)" }}>{error}</p>}
        </div>
      </div>

      {results.length > 0 && mode === "map" && (
        <div>
          <div className="grid gap-4 md:grid-cols-2">
            <figure><canvas ref={(el) => { canvases.current[0] = el; }} className="w-full rounded-lg" style={{ imageRendering: "pixelated" }} /><figcaption className="mt-1 text-sm text-muted">{results[0].label} at 5.46 m</figcaption></figure>
            <figure><canvas ref={(el) => { canvases.current[1] = el; }} className="w-full rounded-lg" style={{ imageRendering: "pixelated" }} /><figcaption className="mt-1 text-sm text-muted">U-Net land cover · mean confidence {fmt(results[0].conf * 100, 0)}%</figcaption></figure>
          </div>
          <Legend shares={comp[0]} />
          <table className="data mt-4"><thead><tr><th>Class</th><th className="num">Share</th><th className="num">Area (km²)</th></tr></thead>
            <tbody>{comp[0].slice(0, 6).map((c) => <tr key={c.k}><td>{c.name}</td><td className="num">{fmt(c.share)}%</td><td className="num">{fmt(c.km2, 2)}</td></tr>)}</tbody></table>
          <button type="button" onClick={download} className="mt-3 rounded-lg border border-line px-3 py-1.5 text-sm">Download the map (PNG)</button>
        </div>
      )}

      {results.length === 2 && mode === "change" && (
        <div>
          <div className="grid gap-4 md:grid-cols-3">
            {[results[0].label, results[1].label, "Where it changed (later class shown)"].map((cap, i) => (
              <figure key={cap}><canvas ref={(el) => { canvases.current[i] = el; }} className="w-full rounded-lg" style={{ imageRendering: "pixelated" }} /><figcaption className="mt-1 text-sm text-muted">{cap}</figcaption></figure>
            ))}
          </div>
          <Legend />
          <table className="data mt-4"><thead><tr><th>Class</th><th className="num">Earlier</th><th className="num">Later</th><th className="num">Net change (km²)</th></tr></thead>
            <tbody>{comp[0].slice(0, 6).map((c, k) => <tr key={c.k}><td>{c.name}</td><td className="num">{fmt(c.share)}%</td><td className="num">{fmt(comp[1][k].share)}%</td><td className="num">{fmt(comp[1][k].km2 - c.km2, 2)}</td></tr>)}</tbody></table>
          {transitions.length > 0 && (
            <>
              <h3 className="mt-6 font-medium">Largest transitions</h3>
              <table className="data mt-2"><thead><tr><th>From</th><th>To</th><th className="num">Area (km²)</th></tr></thead>
                <tbody>{transitions.map((t) => <tr key={`${t.from}-${t.to}`}><td>{CLASS_NAMES[t.from]}</td><td>{CLASS_NAMES[t.to]}</td><td className="num">{fmt(t.km2, 3)}</td></tr>)}</tbody></table>
            </>
          )}
          <p className="mt-3 text-sm text-muted">Model maps of two dates differ partly because of the model and the imagery, not only real change: small scattered differences are usually noise, large coherent patches are more likely real.</p>
          <button type="button" onClick={download} className="mt-3 rounded-lg border border-line px-3 py-1.5 text-sm">Download the change map (PNG)</button>
        </div>
      )}
    </div>
  );
}
