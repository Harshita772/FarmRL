"""FarmRL dashboard: entry point and sidebar navigation.

    streamlit run app/streamlit_app.py      (or the root streamlit_app.py, which runs this file)

Pages are listed explicitly with st.navigation, so the sidebar shows a "Menu" heading and
the titles given here rather than names derived from the file names. Paths are absolute,
so the router works whichever entry point started it.
"""
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
for _p in [str(_APP_DIR), str(_APP_DIR.parent / "src"), str(_APP_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st

PAGES = _APP_DIR / "pages"
menu = [
    st.Page(PAGES / "0_Dashboard.py", title="Dashboard", default=True),
    st.Page(PAGES / "1_Simulate_a_season.py", title="Simulate a season", url_path="Simulate_a_season"),
    st.Page(PAGES / "2_Policy_explorer.py", title="Policy explorer", url_path="Policy_explorer"),
    st.Page(PAGES / "3_Training.py", title="Training", url_path="Training"),
    st.Page(PAGES / "4_Robustness.py", title="Robustness", url_path="Robustness"),
    st.Page(PAGES / "5_Map.py", title="Map", url_path="Map"),
    st.Page(PAGES / "6_Data.py", title="Data", url_path="Data"),
]
st.navigation({"Menu": menu}).run()
