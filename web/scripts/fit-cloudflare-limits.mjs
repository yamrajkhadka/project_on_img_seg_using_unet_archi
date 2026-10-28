// Cloudflare serves files of up to 25 MiB. After `next build`:
// - each U-Net model in out/models (about 30 MiB) is replaced by two halves (<name>.onnx.part0, .part1), which
//   src/lib/client/landcover.ts fetches and joins; the models in public/models stay whole;
// - the copy of onnxruntime's .wasm (about 27 MiB) that the bundler emits is dropped: landcover.ts loads the
//   runtime from jsDelivr (env.wasm.wasmPaths), so the copy is never requested.
// Fails if any file is still over the limit.
import { readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";

const LIMIT = 25 * 1024 * 1024;
const PARTS = 2; // keep in step with MODEL_PARTS in src/lib/client/landcover.ts

for (const name of (await readdir("out/models")).filter((f) => f.endsWith(".onnx"))) {
  const file = path.join("out/models", name);
  const data = await readFile(file);
  const size = Math.ceil(data.length / PARTS);
  if (size >= LIMIT) throw new Error(`${name}: ${PARTS} parts of ${size} bytes would exceed the 25 MiB limit`);
  for (let i = 0; i < PARTS; i++) await writeFile(`${file}.part${i}`, data.subarray(i * size, (i + 1) * size));
  await rm(file);
  console.log(`Split ${name} (${(data.length / 1048576).toFixed(1)} MiB) into ${PARTS} parts`);
}

for (const name of (await readdir("out/_next/static/media")).filter((f) => /^ort-wasm.*\.wasm$/.test(f))) {
  await rm(path.join("out/_next/static/media", name));
  console.log(`Dropped ${name} (onnxruntime loads from jsDelivr)`);
}

const tooBig = [];
async function walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await walk(full);
    else if ((await stat(full)).size > LIMIT) tooBig.push(full);
  }
}
await walk("out");
if (tooBig.length) throw new Error(`Over Cloudflare's 25 MiB limit: ${tooBig.join(", ")}`);
