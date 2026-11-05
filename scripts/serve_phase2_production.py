"""Start the verified Phase 2 bundle under Gunicorn; PORT defaults to 10000."""
from pathlib import Path
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def server_arguments(environment=None):
    environment=os.environ if environment is None else environment
    try:
        port=int(environment.get('PORT','10000'))
    except (TypeError,ValueError) as exc:
        raise ValueError('PORT must be an integer') from exc
    if not 1024<=port<=65535:
        raise ValueError('PORT must be between 1024 and 65535')
    return [sys.executable,'-m','gunicorn','--chdir',str(ROOT),
            '--pythonpath',str(ROOT/'src'),'--bind',f"{environment.get('AIRCAST_HOST','0.0.0.0')}:{port}",
            '--workers','1','--threads','1','--timeout','120','--access-logfile','-',
            'deployment.phase2.wsgi:create_application()']


def main():
    from importlib.util import find_spec
    if find_spec('gunicorn') is None:
        raise SystemExit('Install deployment/phase2/requirements.txt before production serving')
    arguments=server_arguments()
    # No preload: initialize PyTorch inside the worker after Gunicorn forks.
    os.execv(sys.executable,arguments)


if __name__=='__main__':
    main()
