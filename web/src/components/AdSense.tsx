"use client";

import { usePathname } from "next/navigation";
import { useEffect } from "react";

/**
 * Loads the AdSense script on every page except error pages: AdSense doesn't allow ads on screens without content
 * of their own, and Auto ads would place them there. Next marks 404 pages noindex, and no other page here is.
 */
export function AdSense({ client }: { client: string }) {
  const pathname = usePathname();
  useEffect(() => {
    if (document.querySelector('meta[name="robots"][content*="noindex"]') || document.getElementById("adsense")) return;
    const script = document.createElement("script");
    script.id = "adsense";
    script.async = true;
    script.crossOrigin = "anonymous";
    script.src = `https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${client}`;
    document.head.appendChild(script);
  }, [client, pathname]);
  return null;
}
