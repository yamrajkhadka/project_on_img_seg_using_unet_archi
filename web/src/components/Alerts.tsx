import Link from "next/link";
import type { Alert } from "@/lib/alerts";

export function AlertList({ alerts, empty = true }: { alerts: Alert[]; empty?: boolean }) {
  if (!alerts.length) return empty ? <p className="rounded-xl border border-line bg-card p-4 text-sm" style={{ color: "var(--good)" }}>No alerts right now: air, fires, earthquakes, rain and rivers are within normal ranges.</p> : null;
  return (
    <ul className="grid gap-3 sm:grid-cols-2">
      {alerts.map((a) => (
        <li key={a.title} className="rounded-xl border bg-card p-4" style={{ borderColor: a.severity === "critical" ? "var(--bad)" : "var(--warn)" }}>
          <p className="text-xs font-medium uppercase tracking-wider" style={{ color: a.severity === "critical" ? "var(--bad)" : "var(--warn)" }}>{a.topic} · {a.severity === "critical" ? "Severe" : "Warning"}</p>
          <p className="mt-1 font-semibold"><Link href={a.href}>{a.title}</Link></p>
          <p className="text-sm text-muted">{a.detail}</p>
        </li>
      ))}
    </ul>
  );
}
