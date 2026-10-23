from pathlib import Path
import runpy

target = Path(__file__).resolve().parents[1] / "app" / "pages" / "4_Robustness.py"
runpy.run_path(str(target), run_name="__main__")
