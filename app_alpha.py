"""Monitor Nepal — AI land-cover observatory built on the U-Net of Khadka & Pradhan (2026).

This entry point keeps its original name on purpose: the demo URL printed in the paper
(https://4classifying-every-pixel-of-the-planets.streamlit.app) is the Streamlit Community Cloud app that
runs ``app_alpha.py``.  The original single-page demo is preserved verbatim in ``legacy/app_alpha.py``
and reproduced faithfully on the "Model lab" page.

Run locally:  streamlit run app_alpha.py
"""
from monitor_nepal.ui.app import main

main()
