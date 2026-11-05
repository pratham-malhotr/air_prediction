"""Build shareable Phase 2 comparison figures and compact measured outputs."""
import argparse
import os
from pathlib import Path
import sys
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from air_prediction.metrics import regression_metrics, score_predictions

LABELS = {'persistence': 'Persistence', 'seasonal_naive': 'Daily seasonal naive',
          'ridge': 'Ridge', 'xgboost': 'XGBoost', 'bilstm': 'Bi-LSTM', 'lstm': 'LSTM', 'hybrid': 'Bi-LSTM + XGBoost'}
COLORS = {'persistence': '#94a3b8', 'seasonal_naive': '#cbd5e1', 'ridge': '#16a34a',
          'xgboost': '#2563eb', 'bilstm': '#9333ea', 'lstm': '#0891b2', 'hybrid': '#ea580c'}


def build(run, destination):
    run, destination = Path(run), Path(destination)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('is_synthetic', True):
        raise ValueError('Figures require a completed real-data experiment')
    predictions = pd.read_csv(run / 'predictions.csv', parse_dates=['origin_time', 'target_time'], low_memory=False)
    metrics = pd.read_csv(run / 'metrics.csv')
    pd.testing.assert_frame_equal(score_predictions(predictions), metrics, check_exact=False, rtol=1e-10)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('metrics.csv', 'fold_metrics.csv', 'errors_by_pollution_regime.csv', 'selection.json',
                 'protocol.json', 'fold_audit.json', 'run_manifest.json', 'RESULTS.md'):
        if (run / name).resolve() != (destination / name).resolve():
            shutil.copyfile(run / name, destination / name)
    for name in ('search_metrics.csv', 'ablation_metrics.csv', 'ablation_seed_metrics.csv',
                 'hybrid_search_metrics.csv', 'hybrid_seed_metrics.csv', 'seed_stability.csv',
                 'seed_metrics.csv', 'hybrid_fold_audit.json', 'hybrid_comparison_audit.json', 'errors_by_hour.csv'):
        if (run / name).exists() and (run / name).resolve() != (destination / name).resolve():
            shutil.copyfile(run / name, destination / name)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180})

    def save(fig, name):
        fig.savefig(destination / f'{name}.png', bbox_inches='tight')
        fig.savefig(destination / f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    order = metrics.loc[metrics.phase == 'validation'].sort_values('rmse', ascending=False).model
    for ax, phase in zip(axes, ('validation', 'holdout')):
        # Shared y axes must retain identical model order in both periods.
        table = metrics.loc[metrics.phase == phase].set_index('model').reindex(order).reset_index()
        ax.barh([LABELS.get(name, name) for name in table.model], table.rmse,
                color=[COLORS[name] for name in table.model])
        for i, value in enumerate(table.rmse):
            ax.text(value + .6, i, f'{value:.2f}', va='center')
        ax.set_xlim(0, table.rmse.max() * 1.17)
        ax.set_xlabel('RMSE (µg/m³; lower is better)')
        ax.set_title('2013 validation' if phase == 'validation' else '2014 exploratory comparison')
    fig.suptitle(f'Phase 2: identical evaluation hours for all {metrics.model.nunique()} models')
    fig.tight_layout()
    save(fig, 'model_comparison')

    quarters = pd.read_csv(run / 'fold_metrics.csv').query("phase == 'holdout'")
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ('persistence', 'ridge', 'xgboost', 'bilstm', 'lstm', 'hybrid'):
        table = quarters.loc[quarters.model == name].sort_values('fold')
        if table.empty:
            continue
        ax.plot(range(1, len(table) + 1), table.rmse, marker='o', label=LABELS[name], color=COLORS[name])
    ax.set(xticks=[1, 2, 3, 4], xticklabels=['Q1', 'Q2', 'Q3', 'Q4'],
           ylabel='RMSE (µg/m³)', title='2014 quarterly errors (daily seasonal naive omitted for scale)')
    ax.legend(ncol=2)
    ax.grid(axis='y', alpha=.2)
    save(fig, 'quarterly_rmse')

    regimes = pd.read_csv(run / 'errors_by_pollution_regime.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, regime in zip(axes, ('below_training_p90', 'above_training_p90')):
        table = regimes.loc[regimes.pollution_regime == regime].sort_values('rmse')
        ax.barh([LABELS[name] for name in table.model], table.rmse,
                color=[COLORS[name] for name in table.model])
        ax.set_xlabel('RMSE (µg/m³)')
        ax.set_title(f'{regime.replace("_", " ")} (n={int(table.n.iloc[0])})')
    fig.suptitle(f'Pollution regimes: threshold {manifest["training_p90_pm25"]:g} µg/m³ from 2010–2012 only')
    fig.tight_layout()
    save(fig, 'pollution_regimes')

    holdout = predictions.loc[predictions.phase == 'holdout']
    start = holdout.target_time.max() - pd.Timedelta(days=7)
    week = holdout.loc[holdout.target_time > start]
    grid = pd.date_range(start + pd.Timedelta(hours=1), holdout.target_time.max(), freq='h')
    actual = week.loc[week.model == 'bilstm'].set_index('target_time').reindex(grid)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(actual.index, actual.actual, color='#0f172a', label='Observed', linewidth=1.8)
    for name in ('ridge', 'bilstm', 'lstm', 'hybrid'):
        table = week.loc[week.model == name].set_index('target_time').reindex(grid)
        if table.prediction.isna().all():
            continue
        ax.plot(table.index, table.prediction, label=LABELS[name], color=COLORS[name], alpha=.8)
    ax.set(ylabel='PM2.5 (µg/m³)', title='Last seven days of 2014: scored target hours only (missing targets create gaps)')
    ax.legend(ncol=4)
    fig.autofmt_xdate()
    save(fig, 'forecast_last_week')

    audit = json.loads((run / 'fold_audit.json').read_text())
    if any('role' in fold for fold in audit):
        audit = [fold for fold in audit if fold.get('phase') == 'holdout' and fold.get('role') == 'selected_bilstm']
    fig, ax = plt.subplots(figsize=(10, 5))
    for fold in audit:
        ax.plot(range(1, len(fold['training_loss']) + 1), fold['training_loss'], label=fold['fold'])
    ax.set(xlabel='Epoch', ylabel='Training Huber loss (standardized targets)',
           title='Base-model training losses; evaluation data do not control stopping')
    ax.legend(ncol=2)
    save(fig, 'training_losses')
    if (run / 'search_metrics.csv').exists():
        search = pd.read_csv(run / 'search_metrics.csv').sort_values('rmse_mean')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(search.candidate, search.rmse_mean, color='#9333ea')
        ax.set(xlabel='Pooled 2013 validation RMSE (µg/m³)', title='Eight independent Bi-LSTM candidates; fixed screening seed')
        ax.invert_yaxis()
        save(fig, 'validation_search')

        for filename, chart, title in (
            ('ablation_metrics.csv', 'attention_ablation', 'Matched architecture and attention comparison'),
            ('seed_stability.csv', 'seed_stability', 'Selected models: validation stability across fixed seeds')):
            table = pd.read_csv(run / filename).sort_values('rmse_mean')
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(table.candidate, table.rmse_mean, xerr=table.rmse_std, capsize=4, color='#0891b2')
            ax.set(xlabel='Mean pooled 2013 RMSE ± seed standard deviation (µg/m³)', title=title)
            ax.invert_yaxis()
            fig.text(.5, .005, 'Error bars describe variation across seeds; they are not confidence intervals.', ha='center', fontsize=9)
            save(fig, chart)

        hybrid = pd.read_csv(run / 'hybrid_search_metrics.csv').sort_values('rmse_mean').head(8)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.barh(hybrid.candidate, hybrid.rmse_mean, xerr=hybrid.rmse_std, capsize=3, color='#ea580c')
        ax.set(xlabel='Mean 2013 RMSE ± seed standard deviation (µg/m³)',
               title='Best eight hybrid candidates; full search retained in the CSV')
        ax.invert_yaxis()
        save(fig, 'hybrid_selection')

        # Isolate the correction using its own frozen base, rather than the
        # independently trained Bi-LSTM with a different training window.
        rows = []
        for phase in ('validation', 'holdout'):
            table = predictions.loc[(predictions.phase == phase) & (predictions.model == 'hybrid')]
            for regime in ('all_hours', 'below_training_p90', 'above_training_p90'):
                subset = table
                if regime == 'below_training_p90':
                    subset = subset.loc[subset.actual < manifest['training_p90_pm25']]
                elif regime == 'above_training_p90':
                    subset = subset.loc[subset.actual >= manifest['training_p90_pm25']]
                for name, column in (('frozen_base', 'base_prediction'), ('hybrid', 'prediction')):
                    rows.append({'phase': phase, 'model': name, 'pollution_regime': regime,
                                 **regression_metrics(subset.actual, subset[column])})
        correction = pd.DataFrame(rows)
        correction.to_csv(destination / 'correction_effect.csv', index=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        regimes = ('all_hours', 'below_training_p90', 'above_training_p90')
        for shift, name, label, color in ((-.18, 'frozen_base', 'Same frozen neural base', '#9333ea'),
                                          (.18, 'hybrid', 'Base + selected correction', '#ea580c')):
            values = correction.loc[(correction.phase == 'holdout') & (correction.model == name)]
            values = values.set_index('pollution_regime').reindex(regimes)
            ax.bar(np.arange(3) + shift, values.rmse, width=.36, label=label, color=color)
        ax.set(xticks=np.arange(3), xticklabels=['All scored hours', 'Below training p90', 'At/above training p90'],
               ylabel='2014 RMSE (µg/m³)', title='Isolated effect of the selected correction on its own frozen base')
        ax.legend()
        save(fig, 'correction_effect')

    print(f'Wrote compact results and PNG/PDF figures to {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=str(ROOT / 'artifacts/phase2'))
    parser.add_argument('--output', default=str(ROOT / 'docs/results/phase2'))
    args = parser.parse_args()
    build(args.run, args.output)
