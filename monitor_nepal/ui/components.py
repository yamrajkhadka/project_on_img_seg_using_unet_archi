"""Small self-contained HTML components."""
from __future__ import annotations

import base64
import html
import io
import uuid

import numpy as np
import streamlit as st
from PIL import Image


def _jpeg_uri(rgb: np.ndarray, max_side: int = 1400) -> str:
    im = Image.fromarray(rgb)
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def swipe(before: np.ndarray, after: np.ndarray, before_label: str, after_label: str, height_px: int = 520) -> None:
    """Drag-to-compare viewer: the 'before' image is revealed on the left of the handle."""
    uid = "sw" + uuid.uuid4().hex[:8]
    a, b = _jpeg_uri(before), _jpeg_uri(after)
    ratio = before.shape[0] / before.shape[1]
    st.html(f"""
<div id="{uid}" style="position:relative;width:100%;max-width:{int(height_px / ratio)}px;aspect-ratio:{1 / ratio};
     overflow:hidden;border-radius:8px;user-select:none;touch-action:none;cursor:ew-resize;background:#000">
  <img src="{b}" alt="after" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover">
  <div class="clip" style="position:absolute;inset:0;width:50%;overflow:hidden">
    <img src="{a}" alt="before" style="position:absolute;inset:0;width:{100 * 2}%;max-width:none;height:100%;object-fit:cover">
  </div>
  <div class="bar" style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;box-shadow:0 0 4px #000"></div>
  <div class="knob" style="position:absolute;top:50%;left:50%;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;
       background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.5);display:flex;align-items:center;justify-content:center;
       font:600 13px system-ui,sans-serif;color:#333">⇆</div>
  <span style="position:absolute;left:8px;top:8px;padding:2px 8px;border-radius:4px;background:rgba(0,0,0,.6);
       color:#fff;font:12px system-ui,sans-serif">{html.escape(before_label)}</span>
  <span style="position:absolute;right:8px;top:8px;padding:2px 8px;border-radius:4px;background:rgba(0,0,0,.6);
       color:#fff;font:12px system-ui,sans-serif">{html.escape(after_label)}</span>
</div>
<script>
(function() {{
  const root = document.getElementById("{uid}");
  if (!root) return;
  const clip = root.querySelector(".clip"), img = clip.querySelector("img");
  const bar = root.querySelector(".bar"), knob = root.querySelector(".knob");
  function set(x) {{
    const r = root.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (x - r.left) / r.width));
    clip.style.width = (f * 100) + "%";
    img.style.width = (f > 0 ? 100 / f : 100) + "%";
    bar.style.left = knob.style.left = (f * 100) + "%";
  }}
  let down = false;
  root.addEventListener("pointerdown", e => {{ down = true; root.setPointerCapture(e.pointerId); set(e.clientX); }});
  root.addEventListener("pointermove", e => {{ if (down) set(e.clientX); }});
  root.addEventListener("pointerup", () => down = false);
  set(root.getBoundingClientRect().left + root.getBoundingClientRect().width / 2);
}})();
</script>
""", unsafe_allow_javascript=True)
