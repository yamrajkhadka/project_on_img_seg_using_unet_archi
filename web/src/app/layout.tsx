import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import Link from "next/link";
import { AdSense } from "@/components/AdSense";
import { JsonLd } from "@/components/ui";
import { nav, siteConfig } from "@/lib/site";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });

export const metadata: Metadata = {
  metadataBase: new URL(siteConfig.url),
  title: { default: `${siteConfig.name}: Live Air Quality, Weather, Earthquakes & Land Cover`, template: `%s | ${siteConfig.name}` },
  description: siteConfig.description,
  applicationName: siteConfig.name,
  authors: [{ name: siteConfig.author, url: siteConfig.authorUrl }],
  keywords: ["Nepal", "air quality Nepal", "AQI Kathmandu", "Nepal weather", "Nepal earthquake today", "forest fire Nepal",
    "Nepal river level", "flood Nepal", "land cover Nepal", "forest cover Nepal", "districts of Nepal", "municipalities of Nepal"],
  openGraph: { type: "website", siteName: siteConfig.name, locale: "en_US", url: "/" },
  twitter: { card: "summary_large_image" },
  alternates: { canonical: "/" },
  verification: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION
    ? { google: process.env.NEXT_PUBLIC_GOOGLE_SITE_VERIFICATION } : undefined,
  other: { "google-adsense-account": siteConfig.adsenseClient },
};

export const viewport: Viewport = {
  themeColor: [{ media: "(prefers-color-scheme: light)", color: "#fbfbf9" }, { media: "(prefers-color-scheme: dark)", color: "#121211" }],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen">
        <AdSense client={siteConfig.adsenseClient} />
        <JsonLd data={{
          "@context": "https://schema.org", "@type": "WebSite", name: siteConfig.name, url: siteConfig.url,
          description: siteConfig.description, inLanguage: "en",
          publisher: { "@type": "Person", name: siteConfig.author, url: siteConfig.authorUrl },
        }} />
        <header className="sticky top-0 z-10 border-b border-line bg-[var(--bg)]/90 backdrop-blur">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
            <Link href="/" className="flex items-center gap-2 font-semibold text-fg no-underline">
              <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden><path d="M2 27 12 9l6 9 4-6 8 15Z" fill="#c8102e" /><path d="M2 27h28" stroke="#003893" strokeWidth="3" /></svg>
              {siteConfig.name}
            </Link>
            <nav aria-label="Main" className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
              {nav.map((n) => <Link key={n.href} href={n.href} className="text-muted no-underline hover:text-fg">{n.label}</Link>)}
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
        <footer className="border-t border-line">
          <div className="mx-auto grid max-w-6xl gap-6 px-4 py-8 text-sm text-muted sm:grid-cols-3">
            <div>
              <p className="font-semibold text-fg">{siteConfig.name}</p>
              <p className="mt-1">Built on the U-Net land-cover model of <a href={siteConfig.paper.url}>Khadka &amp; Pradhan (2026)</a>, Journal of Engineering Issues and Solutions.</p>
            </div>
            <ul className="space-y-1">
              <li><Link href="/provinces">Provinces</Link> · <Link href="/districts">Districts</Link> · <Link href="/municipalities">Municipalities</Link></li>
              <li><Link href="/climate">Climate</Link> · <Link href="/history">History</Link> · <Link href="/research">Research &amp; model</Link></li>
              <li><Link href="/map">AI map</Link> · <Link href="/change">Change detection</Link> · <Link href="/model-lab">Model lab</Link> · <Link href="/time-machine">Time machine</Link></li>
              <li><a href={siteConfig.appUrl}>Interactive analysis app</a></li>
            </ul>
            <ul className="space-y-1">
              <li><Link href="/about">About</Link> · <Link href="/contact">Contact</Link></li>
              <li><Link href="/privacy">Privacy policy</Link> · <Link href="/terms">Terms of use</Link></li>
              <li>© {new Date().getFullYear()} {siteConfig.author}</li>
              <li>
                More free sites: <a href="https://nepallaw.yamrajkhadka.com.np/">Nepal Law AI</a> ·{" "}
                <a href="https://filecraft.yamrajkhadka.com.np/">FileCraft</a> · <a href="https://calcnepal.pages.dev/">CalcNepal</a> ·{" "}
                <a href="https://lingua.yamrajkhadka.com.np/">Lingua</a> · <a href="https://blog.yamrajkhadka.com.np/blog/">Blog</a>
              </li>
            </ul>
          </div>
        </footer>
      </body>
    </html>
  );
}
