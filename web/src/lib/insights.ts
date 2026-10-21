/** Facts derived from the data for a region: ranks among its peers, comparisons and plain-language summaries. */
import { CLASS_KEYS, CLASS_NAME, childrenOf, type ClassKey, landCoverOf, localLandCover, type Region, regionsAt } from "./data";
import { fmt } from "./format";

export function latestShare(code: string): Record<ClassKey, number> | undefined {
  const lc = landCoverOf(code);
  return lc ? (Object.fromEntries(CLASS_KEYS.map((k) => [k, lc.latest[k][0]])) as Record<ClassKey, number>) : undefined;
}

/** 1-based rank of a region among its level for one class (1 = highest share), and the number ranked. */
export function rankOf(r: Region, k: ClassKey): { rank: number; of: number } | undefined {
  const peers = regionsAt(r.level)
    .map((p) => ({ code: p.code, v: r.level === 3 ? localLandCover(p.code).at(-1)?.share[k] : latestShare(p.code)?.[k] }))
    .filter((p): p is { code: string; v: number } => p.v != null)
    .sort((a, b) => b.v - a.v);
  const i = peers.findIndex((p) => p.code === r.code);
  return i < 0 ? undefined : { rank: i + 1, of: peers.length };
}

export function ordinal(n: number): string {
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return `${n}${s[(v - 20) % 10] ?? s[v] ?? s[0]}`;
}

export function dominant(share: Partial<Record<ClassKey, number>>): ClassKey {
  return CLASS_KEYS.reduce((a, b) => ((share[b] ?? 0) > (share[a] ?? 0) ? b : a));
}

/** "more than", "less than" or "about the same as" a reference value. */
export function compare(v: number, ref: number, tol = 0.15): string {
  if (Math.abs(v - ref) <= Math.max(1, ref * tol)) return "about the same as";
  return v > ref ? "more than" : "less than";
}

/** One-paragraph description; ``ref`` is the comparison area (default: Nepal) and ``refLabel`` its name. */
export function describeShare(name: string, share: Record<ClassKey, number>, ref?: Record<ClassKey, number>,
  refLabel = "the national"): string {
  const top = dominant(share);
  const parts = [
    `${CLASS_NAME[top]} is the largest land cover in ${name}, at ${fmt(share[top], 1)}% of its area.`,
    `Forest covers ${fmt(share.forest, 1)}%${ref ? ` (${compare(share.forest, ref.forest)} ${refLabel} ${fmt(ref.forest, 1)}%)` : ""},`,
    `farmland ${fmt(share.agriculture, 1)}%, rangeland ${fmt(share.rangeland, 1)}%, built-up land ${fmt(share.urban, 1)}%,`,
    `bare land, rock and snow ${fmt(share.barren, 1)}% and water ${fmt(share.water, 1)}%.`,
  ];
  return parts.join(" ");
}

/** Change between the first and latest reference-map years, in percentage points, largest first. */
export function changes(first: Record<ClassKey, number>, last: Record<ClassKey, number>) {
  return CLASS_KEYS.map((k) => ({ k, d: (last[k] ?? 0) - (first[k] ?? 0) })).sort((a, b) => Math.abs(b.d) - Math.abs(a.d));
}

export function signed(v: number, digits = 1): string {
  return `${v > 0 ? "+" : v < 0 ? "−" : "±"}${fmt(Math.abs(v), digits)}`;
}

/** A district's land cover from the same 10 m reference map as its local units: their area-weighted average. */
export function districtReferenceShare(districtCode: string): Record<ClassKey, number> | undefined {
  const units = childrenOf(districtCode).map((u) => ({ u, s: localLandCover(u.code).at(-1)?.share })).filter((x) => x.s);
  const area = units.reduce((s, x) => s + x.u.area, 0);
  if (!area) return undefined;
  return Object.fromEntries(CLASS_KEYS.map((k) => [k, units.reduce((s, x) => s + (x.s![k] ?? 0) * x.u.area, 0) / area])) as Record<ClassKey, number>;
}
