"""Run independent neural selection, paired ablations and hybrid calibration."""
import os
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(ROOT / 'data/raw/PRSA_data_2010.1.1-2014.12.31.csv'))
    parser.add_argument('--config', default=str(ROOT / 'configs/phase1.toml'))
    parser.add_argument('--research-config', default=str(ROOT / 'configs/phase2_research.toml'))
    parser.add_argument('--baseline-dir', default=str(ROOT / 'artifacts/phase1'))
    parser.add_argument('--output', default=str(ROOT / 'artifacts/phase2_research'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    from air_prediction.config import load_config
    from air_prediction.phase2_research import load_research_settings, run_research
    try:
        result = run_research(args.data, load_config(args.config), args.output,
                              args.baseline_dir, load_research_settings(args.research_config), args.resume)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f'Research run failed: {exc}\n')
    print(result['metrics'].to_string(index=False))
    print(f"Saved research results to {result['output']}")


if __name__ == '__main__':
    main()
