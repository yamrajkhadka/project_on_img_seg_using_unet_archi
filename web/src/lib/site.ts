/** Site-wide settings: names, addresses and the research the site is built on. */
export const siteConfig = {
  name: "Nepal Monitor",
  url: "https://nepalmonitor.yamrajkhadka.com.np",
  tagline: "Live air quality, weather, earthquakes, fires, rivers and land cover for every part of Nepal",
  description:
    "Nepal Monitor tracks Nepal live and over time: air quality (AQI, PM2.5) and weather for every district, " +
    "earthquakes, forest fires, river levels and floods, and AI land-cover maps of forest, farmland and built-up " +
    "land for all 7 provinces, 77 districts and every municipality.",
  author: "Yamraj Khadka",
  authorUrl: "https://yamrajkhadka.com.np",
  appUrl: "https://4classifying-every-pixel-of-the-planets.streamlit.app/",
  codeUrl: "https://github.com/yamrajkhadka/project_on_img_seg_using_unet_archi",
  paper: {
    citation:
      "Khadka, Y. & Pradhan, R. (2026). Land Cover Segmentation from Satellite Imagery Using U-Net with Custom " +
      "Loss and Morphological Postprocessing. Journal of Engineering Issues and Solutions, 5(2), 240–253.",
    doi: "10.3126/joeis.v5i2.97825",
    url: "https://doi.org/10.3126/joeis.v5i2.97825",
  },
  // Public AdSense publisher ID (also on nepallaw.yamrajkhadka.com.np); override with NEXT_PUBLIC_ADSENSE_CLIENT.
  adsenseClient: process.env.NEXT_PUBLIC_ADSENSE_CLIENT ?? "ca-pub-8675936402393747",
  // Optional: shown on the contact and privacy pages only when set (as on nepallaw.yamrajkhadka.com.np).
  contactEmail: process.env.NEXT_PUBLIC_CONTACT_EMAIL?.trim() || undefined,
  issuesUrl: "https://github.com/yamrajkhadka/project_on_img_seg_using_unet_archi/issues",
} as const;

export const nav = [
  { href: "/alerts", label: "Alerts" },
  { href: "/air-quality", label: "Air quality" },
  { href: "/weather", label: "Weather" },
  { href: "/earthquakes", label: "Earthquakes" },
  { href: "/fires", label: "Fires" },
  { href: "/rivers", label: "Rivers" },
  { href: "/land-cover", label: "Land cover" },
  { href: "/districts", label: "Districts" },
  { href: "/map", label: "AI map" },
  { href: "/time-machine", label: "Time machine" },
] as const;
