import type { Metadata } from "next";
import { LineChart } from "@/components/charts";
import { PageHeader, Section, Source } from "@/components/ui";
import { events, national } from "@/lib/data";
import { fmt } from "@/lib/format";

export const metadata: Metadata = {
  title: "Nepal Since 1768: Timeline, Population, Forests & Land",
  description: "A timeline of Nepal from unification in 1768 to today, with population, urbanisation, forest and farmland statistics since 1960, and the earthquakes and floods that shaped the country.",
  alternates: { canonical: "/history" },
};

const CATEGORY: Record<string, string> = { state: "State & politics", land: "Land policy", disaster: "Disasters", conservation: "Conservation & milestones" };

export default function Page() {
  const shown = ["population", "urban_share", "forest_share", "agri_share", "electricity", "gdp_pc"].filter((k) => national[k]);
  return (
    <>
      <PageHeader eyebrow="History" title="Nepal since 1768"
        intro={<p>From Prithvi Narayan Shah&apos;s unification of Nepal in 1768 to the federal republic of today: the turning points, the land policies that changed the country&apos;s forests and fields, and the disasters it survived. New major earthquakes and floods are added automatically.</p>} />
      <Section title="Timeline">
        <ol className="relative space-y-4 border-l border-line pl-6">
          {events.map((e) => (
            <li key={`${e.year}-${e.title}`}>
              <div className="absolute -left-1.5 mt-1.5 h-3 w-3 rounded-full" style={{ background: e.category === "disaster" ? "var(--crimson)" : "var(--accent)" }} />
              <p className="text-sm text-muted">{e.year}{e.end ? `–${e.end}` : ""} · {CATEGORY[e.category] ?? e.category}</p>
              <p className="font-medium">{e.title}</p>
              <p className="text-muted">{e.detail}</p>
            </li>
          ))}
        </ol>
      </Section>
      {shown.map((k) => {
        const s = national[k];
        const last = s.series[s.series.length - 1];
        return (
          <Section key={k} title={s.name} note={`${s.unit}; World Bank World Development Indicators. Latest: ${fmt(last[1], last[1] < 100 ? 1 : 0)} (${last[0]}).`}>
            <LineChart series={s.series} label={`${s.name} of Nepal`} unit={s.unit} digits={last[1] < 100 ? 1 : 0} />
          </Section>
        );
      })}
      <Source>Curated timeline from the historical record; recent earthquakes: USGS; floods: GloFAS; statistics: World Bank WDI (FAO for forest and farmland).</Source>
    </>
  );
}
