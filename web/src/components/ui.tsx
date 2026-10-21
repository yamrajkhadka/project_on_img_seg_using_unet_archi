import Link from "next/link";
import type { ReactNode } from "react";

export function PageHeader({ eyebrow, title, intro, children }: { eyebrow?: string; title: string; intro?: ReactNode; children?: ReactNode }) {
  return (
    <header className="mb-8">
      {eyebrow && <p className="mb-1 text-xs font-medium uppercase tracking-wider text-muted">{eyebrow}</p>}
      <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">{title}</h1>
      <div className="mt-2 h-1 w-24 rounded" style={{ background: "linear-gradient(90deg,var(--crimson) 0 62%,#003893 62% 100%)" }} />
      {intro && <div className="prose-site mt-4 text-[1.05rem] text-muted">{intro}</div>}
      {children}
    </header>
  );
}

export function Section({ id, title, children, note }: { id?: string; title: string; children: ReactNode; note?: ReactNode }) {
  return (
    <section id={id} className="mb-12 scroll-mt-20">
      <h2 className="mb-3 text-xl font-semibold tracking-tight">{title}</h2>
      {children}
      {note && <p className="mt-3 text-sm text-muted">{note}</p>}
    </section>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-xl border border-line bg-card p-4 ${className}`}>{children}</div>;
}

export function Stat({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: string }) {
  return (
    <Card>
      <div className="text-sm text-muted">{label}</div>
      <div className="mt-1 text-3xl font-semibold tabular-nums" style={tone ? { color: `var(--${tone})` } : undefined}>{value}</div>
      {sub && <div className="mt-1 text-sm text-muted">{sub}</div>}
    </Card>
  );
}

export function Badge({ children, tone = "muted" }: { children: ReactNode; tone?: string }) {
  return (
    <span className="inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium"
      style={{ color: `var(--${tone === "muted" ? "muted" : tone})`, borderColor: "currentColor" }}>
      {children}
    </span>
  );
}

export function Breadcrumbs({ items }: { items: { href?: string; label: string }[] }) {
  return (
    <nav aria-label="Breadcrumb" className="mb-4 text-sm text-muted">
      <ol className="flex flex-wrap gap-1">
        <li><Link href="/">Home</Link></li>
        {items.map((it) => (
          <li key={it.label} className="flex gap-1">
            <span aria-hidden>/</span>
            {it.href ? <Link href={it.href}>{it.label}</Link> : <span aria-current="page">{it.label}</span>}
          </li>
        ))}
      </ol>
    </nav>
  );
}

/** Structured data for search engines (schema.org JSON-LD). */
export function JsonLd({ data }: { data: object }) {
  return <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(data).replace(/</g, "\\u003c") }} />;
}

/** The exact numbers behind a chart, collapsed by default (accessibility and search engines read it too). */
export function TableTwin({ caption, head, rows }: { caption: string; head: string[]; rows: (string | number)[][] }) {
  return (
    <details className="mt-2">
      <summary className="cursor-pointer text-sm text-muted">Show the numbers</summary>
      <div className="mt-2 overflow-x-auto">
        <table className="data">
          <caption className="sr-only">{caption}</caption>
          <thead><tr>{head.map((h, i) => <th key={h} className={i ? "num" : ""}>{h}</th>)}</tr></thead>
          <tbody>{rows.map((r) => <tr key={String(r[0])}>{r.map((c, i) => <td key={i} className={i ? "num" : ""}>{c}</td>)}</tr>)}</tbody>
        </table>
      </div>
    </details>
  );
}

export function Source({ children }: { children: ReactNode }) {
  return <p className="mt-2 text-xs text-muted">Source: {children}</p>;
}

export function LinkGrid({ links }: { links: { href: string; label: string; sub?: string }[] }) {
  return (
    <ul className="grid gap-x-6 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
      {links.map((l) => (
        <li key={l.href}><Link href={l.href}>{l.label}</Link>{l.sub && <span className="text-sm text-muted"> · {l.sub}</span>}</li>
      ))}
    </ul>
  );
}
