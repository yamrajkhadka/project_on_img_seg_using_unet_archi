import type { MetadataRoute } from "next";
import { siteConfig } from "@/lib/site";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: siteConfig.name, short_name: siteConfig.name, description: siteConfig.description, start_url: "/",
    display: "standalone", background_color: "#fbfbf9", theme_color: "#c8102e",
    icons: [{ src: "/icon.svg", sizes: "any", type: "image/svg+xml" }],
  };
}
