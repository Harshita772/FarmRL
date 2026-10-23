from pathlib import Path
import runpy

target = Path(__file__).resolve().parents[1] / "app" / "pages" / "1_Simulate_a_season.py"
runpy.run_path(str(target), run_name="__main__")
