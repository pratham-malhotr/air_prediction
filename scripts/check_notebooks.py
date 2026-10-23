"""Validate notebooks and execute the portable workflow with explicit synthetic overrides.

This smoke check uses the same code cells in a fresh directory. It does not connect to
Colab, download UCI, train XGBoost, or establish real-data accuracy.
"""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT / "Air_Prediction_Phase1_Colab.ipynb", ROOT / "notebooks" / "01_eda.ipynb"]
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for number, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:{number}", "exec")
        print(f"Validated {path.name}: {len(notebook.cells)} cells")
    notebook = nbformat.read(paths[0], as_version=4)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="notebook-smoke-", dir=ROOT / "artifacts") as directory:
        os.chdir(directory)
        namespace = {"__name__": "__notebook_smoke__"}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for number, cell in enumerate(notebook.cells):
                    if cell.cell_type != "code":
                        continue
                    source = cell.source
                    if "CONFIG_PATH =" in source:
                        source = source.replace("'phase1.toml'", "'demo.toml'").replace("/ 'phase1'", "/ 'demo'")
                    if "if not DATA_PATH.is_file():" in source:
                        source = "from air_prediction.demo import make_demo\nmake_demo(DATA_PATH)\n" + source
                    source = source.replace("RUN_TESTS = True", "RUN_TESTS = False")
                    source = source.replace("MODELS = ('xgboost', 'ridge')", "MODELS = ('ridge',)")
                    source = source.replace("RUN_ABLATIONS = True", "RUN_ABLATIONS = False")
                    exec(compile(source, f"notebook-cell-{number}", "exec"), namespace)
            manifest = namespace["result"]["manifest"]
            assert manifest["is_synthetic"] is True
            assert namespace["export_path"].is_file()
            print("Portable notebook cells executed successfully with synthetic data and Ridge; output export verified.")
            print("Real-data/XGBoost/hosted-Colab execution remains separate.")
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    main()
