"""Application shell: page config, navigation and the shared sidebar."""
from __future__ import annotations

import streamlit as st

from ..config import PAPER, ROOT
from .common import setup_page, sidebar_settings

VIEWS = ROOT / "views"
ASSETS = ROOT / "assets"


def main() -> None:
    st.set_page_config(page_title="Monitor Nepal · AI land-cover observatory", page_icon="🏔️",
                       layout="wide", initial_sidebar_state="auto",
                       menu_items={"About": f"**Monitor Nepal** — national land-cover monitoring with the U-Net of "
                                            f"{PAPER.short_authors} ({PAPER.year}). {PAPER.url}"})
    setup_page()
    st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large", link=PAPER.code_url)
    pages = {
        "Monitor": [
            st.Page(VIEWS / "live.py", title="Live Nepal", icon=":material/public:", default=True),
            st.Page(VIEWS / "landcover.py", title="Land-cover map", icon=":material/layers:", url_path="map"),
            st.Page(VIEWS / "time_machine.py", title="Time machine", icon=":material/history:", url_path="time"),
            st.Page(VIEWS / "change.py", title="Change detection", icon=":material/compare:", url_path="change"),
        ],
        "Research": [
            st.Page(VIEWS / "model_lab.py", title="Model lab · paper demo", icon=":material/science:",
                    url_path="model-lab"),
            st.Page(VIEWS / "methods.py", title="Methods, data & paper", icon=":material/menu_book:",
                    url_path="methods"),
        ],
    }
    nav = st.navigation(pages, position="sidebar")
    sidebar_settings()
    nav.run()
