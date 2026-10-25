"""Start the local Aircast dashboard and its trained-model prediction service."""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
import sys
from threading import Timer
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / ".dependencies"), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true", help="Leave the default browser closed")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port between 1024 and 65535")
    try:
        from werkzeug.serving import make_server
        from frontend.app import create_app
        app = create_app(ROOT)
    except ImportError as error:
        parser.exit(1, f"Missing dependency: {error}. Install the project with its web extra: python -m pip install -e '.[web]'\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"The saved project models or dataset could not be loaded: {error}\n")
    url = f"http://127.0.0.1:{args.port}"
    server = make_server("127.0.0.1", args.port, app, threaded=True)
    print(f"\nAircast is ready: {url}", flush=True)
    print("XGBoost and Ridge are loaded. Press Ctrl+C to stop the app.\n", flush=True)
    if not args.no_open:
        Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAircast stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
