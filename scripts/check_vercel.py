"""Check the hosted dashboard and compare its forecasts with the saved models."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / ".dependencies"), str(ROOT / "src")]


def vercel_command():
    if executable := shutil.which("vercel"):
        return [executable]
    cached = sorted((Path.home() / ".npm/_npx").glob("*/node_modules/vercel/dist/index.js"))
    if cached and (node := shutil.which("node")):
        version = subprocess.run([node, "--version"], capture_output=True, text=True, check=True).stdout
        flags = ["--no-experimental-webstorage"] if int(version.strip().lstrip("v").split(".")[0]) >= 26 else []
        return [node, *flags, str(cached[0])]
    if executable := shutil.which("npx"):
        return [executable, "--yes", "vercel@latest"]
    raise RuntimeError("Vercel CLI is unavailable. Use the deployment script first.")


class HostedApp:
    def __init__(self, url, directory):
        self.url, self.directory = url, Path(directory)
        self.command = vercel_command()
        self.environment = {**os.environ, "NO_UPDATE_NOTIFIER": "1", "VERCEL_TELEMETRY_DISABLED": "1"}

    def request(self, path, payload=None, upload=None, expected=200):
        with TemporaryDirectory(prefix="aircast-check-") as temporary:
            output = Path(temporary) / "response"
            options = ["--silent", "--show-error", "--max-time", "90", "--output", str(output), "--write-out", "%{http_code}"]
            if payload is not None or upload is not None:
                options += ["--header", "Origin: " + self.url]
            if payload is not None:
                body = Path(temporary) / "request.json"
                body.write_text(json.dumps(payload))
                options += ["--header", "Content-Type: application/json", "--request", "POST", "--data-binary", "@" + str(body)]
            if upload is not None:
                options += ["--form", "file=@" + str(upload)]
            command = [*self.command, "--cwd", str(self.directory), "curl", path, "--deployment", self.url, "--", *options]
            result = subprocess.run(command, env=self.environment, capture_output=True, text=True, timeout=120)
            status = result.stdout.strip()[-3:]
            if result.returncode or status != str(expected):
                raise RuntimeError(f"{path}: expected HTTP {expected}, received {status if status.isdigit() else 'no response'} (CLI exit {result.returncode}). Check Vercel authentication and deployment logs.")
            return output.read_bytes()


def verify(client, root=ROOT):
    from frontend.service import ForecastService
    from air_prediction.predict import predict_latest

    root = Path(root)
    local = ForecastService(root)
    checks = []

    def passed(message):
        checks.append(message)
        print("PASS " + message, flush=True)

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def read_json(path, **kwargs):
        try:
            return json.loads(client.request(path, **kwargs))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RuntimeError(f"{path} did not return JSON. Check deployment protection and function logs.") from error

    def compare(actual, expected):
        require(actual["target_time"] == expected["target_time"], "The hosted target time differs from the local forecast.")
        for model in ("xgboost", "ridge"):
            value = actual["predictions"][model]
            require(math.isfinite(value) and value >= 0, f"{model} returned an invalid concentration.")
            require(math.isclose(value, expected["predictions"][model], rel_tol=1e-6, abs_tol=1e-4), f"The hosted {model} forecast differs from the saved local model.")

    health = read_json("/api/health")
    require(health.get("ready") is True and set(health.get("models", [])) == {"xgboost", "ridge"} and health.get("horizon_hours") == 1, "The hosted models are not ready.")
    passed("Both models loaded; forecast horizon is one hour")
    require(b"Forecast the next hour." in client.request("/"), "The dashboard HTML is missing.")
    for asset in ("app.css", "app.js", "favicon.svg"):
        content = client.request("/static/" + asset)
        require(content == (root / "frontend/static" / asset).read_bytes(), f"The hosted {asset} is missing or outdated.")
    passed("Dashboard HTML and all three public assets are current")
    data = read_json("/api/bootstrap")
    expected = local.bootstrap()
    compare(data["forecast"], expected["forecast"])
    for model in ("xgboost", "ridge", "persistence", "seasonal_naive"):
        actual_metrics = next(row for row in data["metrics"] if row["model"] == model)
        expected_metrics = next(row for row in expected["metrics"] if row["model"] == model)
        for name in ("mae", "rmse", "r2", "n"):
            require(math.isclose(actual_metrics[name], expected_metrics[name], abs_tol=1e-9), f"The hosted {model} {name} is outdated.")
    passed("Sample predictions and all four measured comparisons match")
    inputs = {**data["context"]["defaults"], "pm2.5": 65}
    for model in ("xgboost", "ridge"):
        result = read_json("/api/forecast", payload={"mode": "manual", "model": model, "inputs": inputs})
        compare(result, local.forecast(inputs=inputs, selected=model))
        require(result["selected_model"] == model and result["latest"] == 65, "The manual input or model selection was not applied.")
    passed("Edited manual inputs and both model selections run correctly")
    csv = root / "data/raw/PRSA_data_2010.1.1-2014.12.31.csv"
    context = read_json("/api/history", upload=csv)
    result = read_json("/api/forecast", payload={"history_id": context["id"], "history": context["history"], "mode": "history"})
    for model in ("xgboost", "ridge"):
        prediction = predict_latest(root / "artifacts/phase1" / (model + ".joblib"), csv)
        require(math.isclose(result["predictions"][model], prediction["pm25_ug_m3"], rel_tol=1e-6, abs_tol=1e-4), f"The uploaded-history {model} forecast differs from predict_latest.")
    require(result["target_time"] == "2015-01-01T00:00:00+08:00", "The uploaded history forecast has the wrong target time.")
    passed("Original CSV upload and stateless forecasts match predict_latest")
    invalid = read_json("/api/forecast", payload={"mode": "manual", "inputs": {**inputs, "pm2.5": -1}}, expected=400)
    require("pm2.5" in invalid.get("error", {}).get("fields", {}), "Invalid PM2.5 input was not explained.")
    passed("Invalid measurements return a readable validation error")
    require(client.request("/report").startswith(b"%PDF-"), "The project report is not a PDF.")
    require(sha256(client.request("/api/example.csv")).digest() == sha256(csv.read_bytes()).digest(), "The example download differs from the original CSV.")
    passed("Project report and original CSV download are available")
    return {"status": "passed", "checks": checks, "uploaded_history_predictions": result["predictions"],
            "target_time": result["target_time"], "browser_rendering_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://aircast-pm25.vercel.app")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.path or parsed.query or parsed.fragment or parsed.username:
        parser.error("Use an HTTPS deployment origin without a path or credentials.")
    directory = ROOT / "dist/aircast-pm25"
    if not (directory / ".vercel/project.json").exists():
        parser.error("Deploy the prepared application first so its Vercel project is linked.")
    print("Checking " + url + " through the authenticated Vercel CLI…", flush=True)
    try:
        result = verify(HostedApp(url, directory))
    except (RuntimeError, subprocess.TimeoutExpired, OSError, KeyError, ValueError) as error:
        parser.exit(1, "Verification stopped: " + str(error) + "\n")
    result.update(url=url, checked_at=datetime.now(timezone.utc).isoformat())
    output = ROOT / "artifacts/deployment_verification.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("Hosted API and asset checks passed. Browser rendering still needs a visual check.")
    print("Results: " + str(output))


if __name__ == "__main__":
    main()
