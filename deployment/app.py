"""Vercel WSGI entrypoint; copied into the prepared deployment directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from frontend.app import create_app

app = create_app(ROOT)
