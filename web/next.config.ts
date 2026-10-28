import type { NextConfig } from "next";

// Built as plain files (out/) and served by Cloudflare: see wrangler.toml. The live pages are rebuilt on a
// schedule by .github/workflows/deploy-website.yml.
const nextConfig: NextConfig = {
  output: "export",
};

export default nextConfig;
