"""Exercise an actual Gunicorn/container server against saved-model reference values."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import argparse
import json
import math
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def check(url, expected_path, output):
    url = url.rstrip('/')
    expected = json.loads(Path(expected_path).read_text())
    references = expected['cli_dashboard_forecasts']
    assert set(references) == {'lstm','bilstm','hybrid','ridge','xgboost'}

    def request(path, body=None):
        headers = {'Origin':url}
        data = None
        if body is not None:
            headers['Content-Type'] = 'application/json'
            data = json.dumps(body).encode()
        with urlopen(Request(url+path, data=data, headers=headers), timeout=30) as response:
            payload = response.read()
            assert response.headers.get('X-Content-Type-Options') == 'nosniff'
            if path.startswith('/api/'):
                assert response.headers.get('Cache-Control') == 'no-store'
            return payload

    deadline = time.monotonic()+90
    while True:
        try:
            health = json.loads(request('/api/health'))
            break
        except (URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    assert health['ready'] and set(health['models']) == set(references)
    bootstrap = json.loads(request('/api/bootstrap'))
    assert bootstrap['forecast']['selected_model'] == expected['default_model']
    assert len(bootstrap['metrics']) == 7
    assert request('/report').startswith(b'%PDF')
    forecasts = {}
    for model, value in references.items():
        result = json.loads(request('/api/forecast', {'model':model,'mode':'history'}))
        assert result['selected_model'] == model
        assert math.isfinite(result['forecast']) and abs(result['forecast']-value) < 1e-4, (model,result['forecast'],value)
        forecasts[model] = result['forecast']
    try:
        request('/api/forecast', {'mode':'manual','inputs':{'pm2.5':-1}})
    except HTTPError as error:
        assert error.code == 400
    else:
        raise AssertionError('Invalid measurements were accepted')
    result = {'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),
              'url':url,'source_code_sha256':expected['source_code_sha256'],
              'forecasts':forecasts,'tolerance':1e-4,
              'checks':['HTTP readiness with five loaded models','validation-selected default',
                        'seven measured comparisons','Phase 2 PDF','all five saved forecasts',
                        'same-origin POST','invalid-input rejection','response headers']}
    destination = Path(output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8017')
    parser.add_argument('--expected',default=str(ROOT/'artifacts/phase2_research/verification.json'))
    parser.add_argument('--output',default=str(ROOT/'artifacts/phase2_production/verification.json'))
    parser.add_argument('--launch',action='store_true',help='Launch and stop a local Gunicorn server for the check')
    args = parser.parse_args(argv)
    process = None
    try:
        if args.launch:
            parsed = urlparse(args.url)
            if parsed.hostname != '127.0.0.1' or not parsed.port:
                raise ValueError('--launch requires an explicit 127.0.0.1 port')
            environment = {**os.environ,'PORT':str(parsed.port),'AIRCAST_HOST':'127.0.0.1'}
            process = subprocess.Popen([sys.executable,str(ROOT/'scripts/serve_phase2_production.py')],env=environment)
        check(args.url,args.expected,args.output)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__=='__main__':
    main()
