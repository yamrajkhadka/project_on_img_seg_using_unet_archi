import { siteConfig } from "@/lib/site";

export const dynamic = "force-static";

/** Authorizes Google to sell ads on this site; AdSense checks it before paying out. */
export function GET() {
  const publisher = siteConfig.adsenseClient.replace(/^ca-/, "");
  return new Response(`google.com, ${publisher}, DIRECT, f08c47fec0942fa0\n`, { headers: { "Content-Type": "text/plain; charset=utf-8" } });
}
