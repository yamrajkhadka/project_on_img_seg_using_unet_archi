import type { MetadataRoute } from "next";
import { landcover, regionsAt, rivers } from "@/lib/data";
import { siteConfig } from "@/lib/site";

// Written once at build time: the site is exported as static files.
export const dynamic = "force-static";

export default function sitemap(): MetadataRoute.Sitemap {
  const now = new Date();
  const u = (path: string, changeFrequency: MetadataRoute.Sitemap[number]["changeFrequency"], priority: number) =>
    ({ url: `${siteConfig.url}${path}`, lastModified: now, changeFrequency, priority });
  return [
    u("", "hourly", 1),
    ...["/alerts", "/air-quality", "/weather", "/earthquakes", "/fires", "/rivers"].map((p) => u(p, "hourly", 0.9)),
    ...["/map", "/change", "/model-lab", "/time-machine"].map((p) => u(p, "monthly", 0.8)),
    ...["/land-cover", "/provinces", "/districts", "/municipalities", "/climate", "/history", "/research"].map((p) => u(p, "monthly", 0.8)),
    ...["/about", "/contact", "/privacy", "/terms"].map((p) => u(p, "yearly", 0.3)),
    ...regionsAt(2).flatMap((d) => [u(`/air-quality/${d.slug}`, "hourly", 0.8), u(`/weather/${d.slug}`, "hourly", 0.8), u(`/districts/${d.slug}`, "monthly", 0.7)]),
    ...regionsAt(1).map((p) => u(`/provinces/${p.slug}`, "monthly", 0.7)),
    ...rivers.rivers.map((r) => u(`/rivers/${r.key}`, "daily", 0.7)),
    ...landcover.years.map((y) => u(`/land-cover/${y}`, "yearly", 0.6)),
    ...regionsAt(3).map((m) => u(`/municipalities/${m.slug}`, "monthly", 0.6)),
  ];
}
