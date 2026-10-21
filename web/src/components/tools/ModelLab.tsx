"use client";
/** The paper's demo in the browser: upload a satellite image, the U-Net labels every pixel. Nothing is uploaded to
 *  a server; the image and the model stay on the visitor's device. */
import { useEffect, useRef, useState } from "react";
import { argmax, clean, CLASS_NAMES, CLASS_RGB, composition, GSD, type ModelKey, paintClasses, paintRGB, predict, resizeRGBA } from "@/lib/client/landcover";

const fmt = (v: number, d = 1) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

export function ModelLab() {
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<"paper" | "native">("paper");
  const [gsd, setGsd] = useState(0.5);
  const [model, setModel] = useState<ModelKey>("paper");
  const [doClean, setDoClean] = useState(true);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<{ rgb: Uint8ClampedArray; cls: Uint8Array; w: number; h: number; conf: number; native: boolean } | null>(null);
  const inCanvas = useRef<HTMLCanvasElement>(null);
  const outCanvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (result && inCanvas.current && outCanvas.current) {
      paintRGB(inCanvas.current, result.rgb, result.w, result.h);
      paintClasses(outCanvas.current, result.cls, result.w, result.h);
    }
  }, [result]);

  async function run() {
    if (!file) return;
    setBusy(true); setResult(null);
    try {
      setStatus("Reading the image…");
      const bmp = await createImageBitmap(file);
      const c = new OffscreenCanvas(bmp.width, bmp.height);
      const ctx = c.getContext("2d")!;
      ctx.drawImage(bmp, 0, 0);
      const src = ctx.getImageData(0, 0, bmp.width, bmp.height);
      let w = 224, h = 224;
      if (mode === "native") {
        const f = gsd / GSD;
        w = Math.max(64, Math.min(1024, Math.round((bmp.width * f) / 16) * 16));
        h = Math.max(64, Math.min(1024, Math.round((bmp.height * f) / 16) * 16));
      }
      const rgb = resizeRGBA(src, w, h);
      setStatus("Running the U-Net in your browser (the first run also downloads the 31 MB model)…");
      const probs = await predict(model, rgb, w, h);
      const { cls: raw, conf } = argmax(probs, w * h);
      const cls = doClean ? clean(raw, w, h, 64) : raw;
      setResult({ rgb, cls, w, h, conf: conf.reduce((s, v) => s + v, 0) / conf.length, native: mode === "native" });
      setStatus("");
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "Could not process this image.");
    } finally {
      setBusy(false);
    }
  }

  const comp = result ? composition(result.cls) : null;
  return (
    <div className="space-y-6">
      <div className="grid gap-6 rounded-xl border border-line bg-card p-4 lg:grid-cols-2">
        <div className="space-y-3">
          <label className="block text-sm">Satellite or aerial image (JPG or PNG, taken from directly above)
            <input type="file" accept="image/*" className="mt-1 block w-full text-sm" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <p className="text-xs text-muted">The image never leaves your device: the model runs in your browser.</p>
          <fieldset className="space-y-1 text-sm">
            <legend className="mb-1">How the image is fed to the network</legend>
            <label className="flex items-center gap-2"><input type="radio" checked={mode === "paper"} onChange={() => setMode("paper")} /> As in the paper: resize the whole image to 224 × 224</label>
            <label className="flex items-center gap-2"><input type="radio" checked={mode === "native"} onChange={() => setMode("native")} /> True scale: resample to 5.46 m per pixel</label>
            {mode === "native" && (
              <label className="ml-6 block">Image resolution (metres per pixel)
                <input type="number" min={0.1} max={30} step={0.1} value={gsd} onChange={(e) => setGsd(Number(e.target.value) || 0.5)} className="ml-2 w-24 rounded border border-line bg-card p-1" />
              </label>
            )}
          </fieldset>
        </div>
        <div className="space-y-3">
          <label className="block text-sm">Model
            <select className="mt-1 block rounded-lg border border-line bg-card p-2" value={model} onChange={(e) => setModel(e.target.value as ModelKey)}>
              <option value="paper">Original U-Net from the paper (DeepGlobe)</option>
              <option value="nepal">Nepal-adapted U-Net</option>
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={doClean} onChange={(e) => setDoClean(e.target.checked)} /> Post-processing: remove regions smaller than 64 pixels</label>
          <button type="button" disabled={!file || busy} onClick={run} className="rounded-lg bg-[var(--accent)] px-4 py-2 font-medium text-white disabled:opacity-60">{busy ? "Working…" : "Classify every pixel"}</button>
          {status && <p className="text-sm text-muted" aria-live="polite">{status}</p>}
        </div>
      </div>
      {result && comp && (
        <div>
          <div className="grid gap-4 md:grid-cols-2">
            <figure><canvas ref={inCanvas} className="w-full rounded-lg" /><figcaption className="mt-1 text-sm text-muted">Network input · {result.w} × {result.h} px</figcaption></figure>
            <figure><canvas ref={outCanvas} className="w-full rounded-lg" style={{ imageRendering: "pixelated" }} /><figcaption className="mt-1 text-sm text-muted">Predicted land cover · mean confidence {fmt(result.conf * 100, 0)}%</figcaption></figure>
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm">
            {comp.map((c) => (
              <span key={c.k} className="inline-flex items-center gap-1.5">
                <span className="inline-block h-3 w-3 rounded-sm" style={{ background: `rgb(${CLASS_RGB[c.k].join(",")})` }} />{CLASS_NAMES[c.k]} <b>{fmt(c.share)}%</b>
                {result.native && <span className="text-muted">({fmt(c.km2, 3)} km²)</span>}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
