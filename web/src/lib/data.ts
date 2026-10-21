/** Typed access to the data exported by scripts/export_web.py (web/data/*.json). Server-side only. */
import climateJson from "../../data/climate.json";
import eventsJson from "../../data/events.json";
import hazardsJson from "../../data/hazards.json";
import landcoverJson from "../../data/landcover.json";
import localJson from "../../data/local_landcover.json";
import modelJson from "../../data/model.json";
import nationalJson from "../../data/national.json";
import regionsJson from "../../data/regions.json";
import riversJson from "../../data/rivers.json";
import townsJson from "../../data/towns.json";

export type Level = 1 | 2 | 3;
export type ClassKey = "urban" | "agriculture" | "rangeland" | "forest" | "water" | "barren";
export const CLASS_KEYS: ClassKey[] = ["urban", "agriculture", "rangeland", "forest", "water", "barren"];

export interface Region {
  code: string;
  level: Level;
  name: string;
  slug: string;
  parent: string | null;
  province: string | null;
  area: number;
  lon: number;
  lat: number;
  bbox: [number, number, number, number];
}

/** share %, low, high (95% interval), area km² */
export type Estimate = Record<ClassKey, [number, number, number, number]>;
export interface LandCover {
  n: number;
  imagery: string;
  latest: Estimate;
  trend: Record<string, Record<ClassKey, [number, number, number]>>;
  reference: Record<string, Record<ClassKey, number>>;
}
export interface LocalYear {
  year: number;
  res_m: number;
  share: Record<ClassKey, number>;
  unmapped_pct: number;
}

const regions = regionsJson.regions as Region[];
export const country = regionsJson.country as { code: string; name: string; area: number; bbox: Region["bbox"] };
const byCode = new Map(regions.map((r) => [r.code, r]));
const bySlug = new Map(regions.map((r) => [`${r.level}:${r.slug}`, r]));

export const LEVEL_PATH: Record<Level, string> = { 1: "provinces", 2: "districts", 3: "municipalities" };
export const LEVEL_NAME: Record<Level, string> = { 1: "Province", 2: "District", 3: "Local unit" };

export function regionsAt(level: Level): Region[] {
  return regions.filter((r) => r.level === level).sort((a, b) => a.name.localeCompare(b.name));
}
export function regionBySlug(level: Level, slug: string): Region | undefined {
  return bySlug.get(`${level}:${slug}`);
}
export function regionByCode(code: string | null | undefined): Region | undefined {
  return code ? byCode.get(code) : undefined;
}
export function childrenOf(code: string): Region[] {
  return regions.filter((r) => r.parent === code).sort((a, b) => a.name.localeCompare(b.name));
}
/** Province, district, … above a region, outermost first. */
export function ancestors(r: Region): Region[] {
  const out: Region[] = [];
  for (let p = regionByCode(r.parent); p; p = regionByCode(p.parent)) out.unshift(p);
  return out;
}
export function regionHref(r: Region): string {
  return `/${LEVEL_PATH[r.level]}/${r.slug}`;
}

/** What kind of local unit a name describes, from its official suffix. */
export function unitKind(r: Region): string {
  const m = r.name.match(/(Metropolitan City|Sub-Metropolitan City|Rural Municipality|Municipality|National Park|Wildlife Reserve|Conservation Area|Hunting Reserve|Lake)$/i);
  return m ? m[1] : "Local unit";
}

export const landcover = landcoverJson as unknown as {
  classes: { key: ClassKey; name: string; definition: string }[];
  years: number[];
  reference_years: number[];
  regions: Record<string, LandCover>;
};
export const CLASS_NAME = Object.fromEntries(landcover.classes.map((c) => [c.key, c.name])) as Record<ClassKey, string>;

export function landCoverOf(code: string): LandCover | undefined {
  return landcover.regions[code];
}

const local = localJson as unknown as Record<string, { years: Record<string, LocalYear | { error: string }> }>;
/** Reference-map land cover of a local unit, first and latest year (when available). */
export function localLandCover(code: string): LocalYear[] {
  const entry = local[code];
  if (!entry) return [];
  return Object.values(entry.years).filter((y): y is LocalYear => "share" in y).sort((a, b) => a.year - b.year);
}

export interface Town {
  town: string;
  lat: number;
  lon: number;
  population: number;
  kind: string;
}
const towns = townsJson as Record<string, Town>;
export function townOf(districtCode: string): Town | undefined {
  return towns[districtCode];
}
/** The district's town name when it is a real seat or a sizeable town, for page titles. */
export function placeLabel(d: Region): string {
  const t = towns[d.code];
  const notable = t && !t.town.startsWith(d.name) && (/^PPL(C|A)/.test(t.kind) || t.population >= 10000);
  return notable ? `${t.town}, ${d.name}` : d.name;
}

export const rivers = riversJson as unknown as {
  rivers: { key: string; name: string; gauge: string; lat: number; lon: number; province: string }[];
  climatology: Record<string, [number, number, number, number][]>;
  thresholds: Record<string, { q2: number; q10: number; mean: number }>;
  annual: Record<string, [number, number, number, string][]>;
};
export const climate = climateJson as unknown as Record<string, { name: string; unit: string; base_end: number; series: [number, number][] }>;
export const events = eventsJson as { year: number; end: number | null; title: string; category: string; detail: string }[];
export const national = nationalJson as unknown as Record<string, { name: string; unit: string; series: [number, number][] }>;
export const model = modelJson as unknown as {
  val_miou: number;
  baseline_miou: number;
  summary: string;
  updated: string;
  evaluation: string;
  results: { model: string; source: string; reference: string; overall: number; kappa: number; miou: number }[];
  inter_reference?: { overall: number; kappa: number };
};

export interface FireYear { total: number; months: number[]; districts: Record<string, number> }
export const hazards = hazardsJson as unknown as {
  fires: Record<string, Record<string, FireYear>>;
  quakes: Record<string, { m4: number; m5: number; m6: number }>;
};
