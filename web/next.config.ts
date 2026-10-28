import type { NextConfig } from "next";

// Built as plain files (out/) and served by Cloudflare: see wrangler.toml. The live pages are rebuilt on a
// schedule by .github/workflows/deploy-website.yml.
const nextConfig: NextConfig = {
  output: "export",
  // A fixed build ID keeps unchanged pages byte-identical between the scheduled rebuilds, so each deploy uploads
  // only the pages whose data changed.
  generateBuildId: async () => "nepalmonitor",
};

export default nextConfig;
