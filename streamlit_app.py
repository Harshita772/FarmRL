"""FarmRL Streamlit root entrypoint for Streamlit Community Cloud & Hugging Face."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
SRC_DIR = ROOT / "src"

for p in [str(APP_DIR), str(SRC_DIR), str(ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Execute the primary dashboard application in app/
import runpy
runpy.run_path(str(APP_DIR / "streamlit_app.py"), run_name="__main__")
