FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PORT=10000 MPLCONFIGDIR=/tmp/aircast-matplotlib
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml requirements-phase2.txt README.md ./
COPY src/ src/
COPY deployment/phase2/ deployment/phase2/
COPY scripts/restore_phase2_runtime.py scripts/
RUN python -S scripts/restore_phase2_runtime.py
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 && python -m pip install --no-cache-dir -r deployment/phase2/requirements.txt
COPY frontend/ frontend/
COPY scripts/serve_phase2_production.py scripts/
RUN useradd --system --uid 10001 --create-home aircast
USER aircast
EXPOSE 10000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import json,os,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','10000')+'/api/health',timeout=4)); assert r['ready'] and 'lstm' in r['models']"
CMD ["python", "scripts/serve_phase2_production.py"]
