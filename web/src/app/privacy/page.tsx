import type { Metadata } from "next";
import { PageHeader } from "@/components/ui";
import { siteConfig } from "@/lib/site";

export const metadata: Metadata = { title: "Privacy Policy", description: "How Nepal Monitor handles data, cookies and advertising.", alternates: { canonical: "/privacy" } };

export default function Page() {
  return (
    <>
      <PageHeader title="Privacy policy" intro={<p>Last updated 21 October 2026.</p>} />
      <div className="prose-site">
        <p><b>What we collect.</b> Nepal Monitor has no accounts and does not ask for personal information. Our hosting provider (Vercel) keeps standard server logs, such as IP address, browser and pages requested, for security and reliability.</p>
        <p><b>Advertising.</b> We use Google AdSense to show ads. Google and its partners use cookies to serve ads based on your visits to this and other websites. You can opt out of personalised advertising at <a href="https://adssettings.google.com">Google Ads Settings</a> or <a href="https://www.aboutads.info/choices">aboutads.info</a>. Learn how Google uses information from sites that use its services at <a href="https://policies.google.com/technologies/partner-sites">policies.google.com/technologies/partner-sites</a>.</p>
        <p><b>Third-party data.</b> Pages load data on our server from public services (NASA, USGS, Open-Meteo, World Bank); your browser does not contact them, so they do not receive your information.</p>
        <p><b>Children.</b> The site is not directed at children under 13 and does not knowingly collect their data.</p>
        <p><b>Contact.</b> Questions about privacy:{" "}
          {siteConfig.contactEmail ? <a href={`mailto:${siteConfig.contactEmail}`}>{siteConfig.contactEmail}</a> : <a href={siteConfig.issuesUrl}>GitHub issues</a>}.</p>
      </div>
    </>
  );
}
