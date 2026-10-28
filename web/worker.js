// Runs only for /opengraph-image (run_worker_first in wrangler.toml); every other request is a static file from out/.
// Next writes its generated share image as a file without an extension, which would go out with no content type,
// and some sites then refuse to show the preview.
const worker = {
  async fetch(request, env) {
    const res = await env.ASSETS.fetch(request);
    if (!res.ok) return res;
    const headers = new Headers(res.headers);
    headers.set("Content-Type", "image/png");
    return new Response(res.body, { status: res.status, headers });
  },
};

export default worker;
