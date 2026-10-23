"""HTTP endpoints for the local Aircast dashboard."""

from pathlib import Path
import os

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

from .service import ForecastService, InputError, MAX_UPLOAD_BYTES


def create_app(project_root=None, service=None):
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES, MAX_FORM_PARTS=10)
    if os.environ.get("VERCEL") == "1":
        app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
    engine = service if service is not None else ForecastService(root)
    app.extensions["forecast_service"] = engine

    @app.before_request
    def check_origin():
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
                raise InputError("Please submit measurements from this dashboard.", status=403)

    @app.after_request
    def response_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/bootstrap")
    def bootstrap():
        return jsonify(engine.bootstrap())

    @app.get("/api/health")
    def health():
        return jsonify(ready=True, models=list(engine.models), horizon_hours=1)

    @app.post("/api/forecast")
    def forecast():
        values = request.get_json(silent=True)
        if not isinstance(values, dict):
            raise InputError("Please send the forecast measurements as a JSON object.")
        return jsonify(engine.forecast(history_id=values.get("history_id", "sample"),
                                       mode=values.get("mode", "manual"), inputs=values.get("inputs"),
                                       selected=values.get("model", "xgboost"), history=values.get("history")))

    @app.post("/api/history")
    def upload_history():
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            raise InputError("Choose an hourly CSV file to upload.")
        return jsonify(engine.upload(upload.read(), upload.filename))

    @app.get("/api/example.csv")
    def example_csv():
        return send_file(root / "data/raw/PRSA_data_2010.1.1-2014.12.31.csv",
                         mimetype="text/csv", as_attachment=True, download_name="beijing_hourly_example.csv")

    @app.get("/report")
    def report():
        return send_file(root / "reports/ieee/main.pdf", mimetype="application/pdf")

    @app.errorhandler(InputError)
    def invalid_input(error):
        return jsonify(error={"message": str(error), "fields": error.fields}), error.status

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(error):
        return jsonify(error={"message": "This CSV is too large. Please keep uploads below 4 MB.", "fields": {}}), 413

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error={"message": error.description, "fields": {}}), error.code

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception("Could not complete the dashboard request")
        return jsonify(error={"message": "The forecast could not be completed. Please try again in a moment.", "fields": {}}), 500

    return app
