import { ImageResponse } from "next/og";
import { siteConfig } from "@/lib/site";

export const alt = `${siteConfig.name}: live air quality, weather, earthquakes, fires, rivers and land cover of Nepal`;
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function Image() {
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", justifyContent: "center", padding: 80, background: "#fbfbf9", color: "#1c1c1a" }}>
        <div style={{ display: "flex", fontSize: 34, color: "#c8102e", fontWeight: 600 }}>nepalmonitor.yamrajkhadka.com.np</div>
        <div style={{ display: "flex", fontSize: 92, fontWeight: 800, marginTop: 16 }}>{siteConfig.name}</div>
        <div style={{ display: "flex", fontSize: 40, marginTop: 20, color: "#5d5d58", lineHeight: 1.3 }}>Live air quality, weather, earthquakes, fires and rivers, and AI land-cover maps for every district and municipality of Nepal.</div>
        <div style={{ display: "flex", height: 14, width: 260, marginTop: 40, background: "linear-gradient(90deg,#c8102e 0 62%,#003893 62% 100%)" }} />
      </div>
    ),
    size,
  );
}
