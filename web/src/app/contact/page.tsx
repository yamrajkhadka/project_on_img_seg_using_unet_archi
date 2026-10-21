import type { Metadata } from "next";
import { PageHeader } from "@/components/ui";
import { siteConfig } from "@/lib/site";

export const metadata: Metadata = { title: "Contact", description: "Contact Nepal Monitor.", alternates: { canonical: "/contact" } };

export default function Page() {
  return (
    <>
      <PageHeader title="Contact" />
      <div className="prose-site">
        <p>Corrections, data questions, research collaboration or advertising:{" "}
          {siteConfig.contactEmail ? <>email <a href={`mailto:${siteConfig.contactEmail}`}>{siteConfig.contactEmail}</a>.</> : <>open an issue on <a href={siteConfig.issuesUrl}>GitHub</a>.</>}</p>
        <p>Author: <a href={siteConfig.authorUrl}>{siteConfig.author}</a>. Code: <a href={siteConfig.codeUrl}>GitHub</a>.</p>
      </div>
    </>
  );
}
