"""Create a source ZIP including the portable notebooks, excluding local data and secrets."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from build_notebooks import ROOT, source_files


def main():
    destination = ROOT / "dist" / "air_prediction_phase1.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    notebooks = [ROOT / "Air_Prediction_Phase1_Colab.ipynb", ROOT / "notebooks" / "01_eda.ipynb"]
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in [*source_files(), *notebooks]:
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(destination)


if __name__ == "__main__":
    main()
