"""Run audited Bi-LSTM and hybrid backtests without modifying Phase 1 outputs."""

import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(ROOT / 'data/raw/PRSA_data_2010.1.1-2014.12.31.csv'))
    parser.add_argument('--config', default=str(ROOT / 'configs/phase1.toml'))
    parser.add_argument('--phase2-config', default=str(ROOT / 'configs/phase2.toml'))
    parser.add_argument('--baseline-dir', default=str(ROOT / 'artifacts/phase1'))
    parser.add_argument('--output', default=str(ROOT / 'artifacts/phase2'))
    args = parser.parse_args(argv)
    try:
        from air_prediction.config import load_config
        from air_prediction.phase2 import load_phase2_settings, run_phase2
        dl_config, residual, fraction = load_phase2_settings(args.phase2_config)
        result = run_phase2(args.data, load_config(args.config), args.output, args.baseline_dir,
                            dl_config, residual, fraction)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Phase 2 failed: {exc}\n')
    print(result['metrics'].to_string(index=False))
    print(f"Results saved to {result['output']}")


if __name__ == '__main__':
    main()
