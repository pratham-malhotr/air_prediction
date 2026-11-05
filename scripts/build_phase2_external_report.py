"""Independently recompute external scores and publish a compact transfer supplement."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
sys.path[:0] = [str(ROOT),str(ROOT/'src')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from air_prediction.metrics import score_predictions

LABELS = {'lstm':'LSTM','bilstm':'Bi-LSTM','hybrid':'Hybrid','ridge':'Ridge',
          'xgboost':'XGBoost','persistence':'Persistence','seasonal_naive':'Daily seasonal naive'}


def build():
    run = ROOT/'artifacts/phase2_external'
    manifest = json.loads((run/'run_manifest.json').read_text())
    protocol = json.loads((run/'protocol.json').read_text())
    assert manifest['status']=='complete' and manifest['no_external_fitting']
    assert sha256((run/'protocol.json').read_bytes()).hexdigest()==manifest['protocol_sha256']
    assert sha256((ROOT/'scripts/validate_phase2_external.py').read_bytes()).hexdigest()==manifest['evaluator_sha256']
    for model,digest in protocol['model_sha256'].items():
        assert sha256((run/'reference_models'/f'{model}.joblib').read_bytes()).hexdigest()==digest
    predictions = pd.read_csv(run/'predictions.csv.gz',float_precision='round_trip',parse_dates=['origin_time','target_time'])
    assert not predictions.duplicated(['station','model','origin_time']).any()
    assert predictions.groupby(['station','origin_time']).model.nunique().eq(7).all()
    assert predictions.groupby(['station','origin_time']).actual.nunique().eq(1).all()
    assert predictions.target_time.between(protocol['target_start'],protocol['target_end_exclusive'],inclusive='left').all()
    assert (predictions.target_time-predictions.origin_time).eq(pd.Timedelta(hours=1)).all()
    pooled = pd.read_csv(run/'pooled_metrics.csv',float_precision='round_trip')
    station = pd.read_csv(run/'station_metrics.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(score_predictions(predictions,('model',)),pooled,check_exact=False,rtol=1e-11)
    pd.testing.assert_frame_equal(score_predictions(predictions,('station','model')),station,check_exact=False,rtol=1e-11)
    assert pooled.n.nunique()==1 and station.station.nunique()==12
    audits = pd.DataFrame(json.loads((run/'station_audit.json').read_text()))
    assert audits.common_target_hours.sum()==pooled.n.iloc[0]
    provenance = json.loads((ROOT/'data/raw/external/uci501.provenance.json').read_text())
    assert provenance['sha256']==manifest['archive_sha256']
    assert pd.Timestamp(provenance['downloaded_utc'])>pd.Timestamp(protocol['declared_utc'])
    destination = ROOT/'docs/results/phase2_external'
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('protocol.json','run_manifest.json','pooled_metrics.csv','station_metrics.csv',
                 'macro_metrics.csv','year_metrics.csv','regime_metrics.csv','station_audit.json'):
        shutil.copyfile(run/name,destination/name)
    shutil.copyfile(ROOT/'data/raw/external/uci501.provenance.json',destination/'dataset_provenance.json')
    order = pooled.sort_values('rmse').model.tolist()
    compact = [name for name in order if name!='seasonal_naive']
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False}):
        fig,(left,right)=plt.subplots(1,2,figsize=(11,5),gridspec_kw={'width_ratios':[1,1.8]},layout='constrained')
        table=pooled.set_index('model').loc[compact]
        left.barh([LABELS[name] for name in compact],table.rmse,color=['#2563eb' if name=='lstm' else '#94a3b8' for name in compact])
        for index,value in enumerate(table.rmse):
            left.text(value+.2,index,f'{value:.3f}',va='center',fontsize=8)
        left.set_xlim(0,table.rmse.max()+3.5)
        left.invert_yaxis()
        left.set(xlabel='Pooled RMSE (µg/m³)',title='Frozen external transfer · 2015–2017')
        table=station.pivot(index='station',columns='model',values='rmse').reindex(columns=compact)
        heat=right.imshow(table.to_numpy(),aspect='auto',cmap='Blues')
        right.set(xticks=range(len(compact)),xticklabels=[LABELS[name] for name in compact],
                  yticks=range(len(table)),yticklabels=table.index,title='Station RMSE on common observed hours')
        for y in range(len(table)):
            for x in range(len(compact)):
                color='white' if heat.norm(table.iloc[y,x])>.6 else '#111827'
                right.text(x,y,f'{table.iloc[y,x]:.1f}',ha='center',va='center',fontsize=8,color=color)
        fig.colorbar(heat,ax=right,label='RMSE (µg/m³)',shrink=.7)
        for suffix in ('png','pdf'):
            fig.savefig(destination/f'external_transfer_rmse.{suffix}',dpi=220)
        plt.close(fig)
    score_rows='\n'.join(f'| {LABELS[row.model]} | {row.mae:.3f} | {row.rmse:.3f} | {row.r2:.4f} | {row.bias:.3f} |'
                          for row in pooled.sort_values('rmse').itertuples())
    macro=pd.read_csv(run/'macro_metrics.csv').set_index('model')
    station_rows='\n'.join(f'| {name} | {int(count):,} |' for name,count in zip(audits.station,audits.common_target_hours))
    preferred=pooled.set_index('model').loc['lstm']
    persistence=pooled.set_index('model').loc['persistence']
    best=pooled.sort_values('rmse').iloc[0]
    mae_leader=pooled.sort_values('mae').iloc[0]
    narrative=f'''# Frozen external transfer evaluation

This supplement evaluates the previously selected models on all 12 stations in [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) (DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), CC BY 4.0). Targets run from 1 January 2015 through 28 February 2017. The declared model copies, feature mapping, station set and scoring policy were frozen before the archive was downloaded. No model, imputer, encoder, hyperparameter, correction weight or seed was fitted or selected on these observations.

## Transfer conditions

These are different monitoring sites with weather from nearby meteorological stations. `PM2.5`, `TEMP`, `DEWP`, `PRES` and compass `wd` are mapped directly. `WSPM` measures instantaneous wind speed and cannot replace the accumulated `Iws` input; `RAIN` is precipitation amount and cannot replace rain duration `Ir`. `Iws`, `Ir` and `Is` therefore remain missing, using the existing fitted handling. Unseen wind categories retain their labels and follow the fitted unknown-category path. The fraction outside the original wind vocabulary ranges from {audits.unseen_wind_fraction.min():.1%} to {audits.unseen_wind_fraction.max():.1%} across stations.

This is external station and feature transfer evidence. It does not establish blind accuracy at the original station or present-day performance. The five-page main paper retains the original historical study; this is a separately declared supplement.

## Results

Every method is scored on the same **{int(preferred.n):,} station-hours**, requiring observed next-hour targets, current PM2.5 and daily seasonal baseline values. Missing scoring labels are excluded. Features and sequence windows keep the complete hourly grid. Errors are in µg/m³; bias is prediction minus observation.

| Model | MAE | RMSE | R² | Bias |
|---|---:|---:|---:|---:|
{score_rows}

The original validation preference remains LSTM. Its external RMSE is {preferred.rmse:.3f}, versus {persistence.rmse:.3f} for persistence. The lowest observed pooled RMSE is {best.rmse:.3f} for {LABELS[best.model]}; this table does not trigger model reselection. {LABELS[mae_leader.model]} has the lowest pooled MAE, {mae_leader.mae:.3f}, so the RMSE ranking does not imply improvement on every error measure. LSTM station-macro RMSE is {macro.loc['lstm','rmse_mean']:.3f} with station SD {macro.loc['lstm','rmse_std']:.3f}. Station SD describes geographical variation, not an uncertainty interval or a statistical-significance test. Pooled scores weight observed station-hours; macro scores weight stations equally.

![Frozen external pooled and station RMSE](results/phase2_external/external_transfer_rmse.png)

The chart omits daily seasonal naive to keep the six stronger methods legible; its measured score remains in the table and machine-readable evidence. High-pollution analysis uses the original training-only threshold of {protocol['threshold_pm25']:g} µg/m³. [Year](results/phase2_external/year_metrics.csv), [pollution regime](results/phase2_external/regime_metrics.csv), [station](results/phase2_external/station_metrics.csv) and [macro](results/phase2_external/macro_metrics.csv) tables retain the detailed comparisons.

| Station | Common target hours |
|---|---:|
{station_rows}

## Provenance and reproduction

The [frozen protocol](results/phase2_external/protocol.json) records exact model bytes and both training and runtime source hashes. The reference copies were taken from the first published research run. The runtime differs only in lossless loading of the historical baseline comparison CSV; model inference and fitted parameters are unchanged. These exact reference copies remain in the runnable ZIP under `artifacts/phase2_external/reference_models/` even after the portability rerun updates the active research bundle. [Portability verification](results/phase2_research/portability_verification.json) confirms exact equality of the newly trained neural tensors, preprocessing, target statistics, hybrid tree bytes and saved forecasts. Reproduce that comparison with `python scripts/verify_phase2_portability.py`.

The [dataset provenance](results/phase2_external/dataset_provenance.json) records the official URL, download time and archive hash. The [completed manifest](results/phase2_external/run_manifest.json) links the protocol, evaluator, archive and model hashes. The full predictions and frozen reference weights are included in the runnable release; no new labels are committed into the source repository.

```bash
python scripts/validate_phase2_external.py --output artifacts/phase2-external-new --plan-only
python scripts/validate_phase2_external.py --output artifacts/phase2-external-new --resume
python scripts/build_phase2_external_report.py
```

The final report builder independently recomputes pooled and station scores, checks common targets, timestamps, hashes and the pre-download declaration. It summarizes the canonical `artifacts/phase2_external/` run; preserve that completed directory and choose a new output for another evaluation. Repeating this now is reproduction, not another untouched external test. No transfer result establishes a causal weather effect or universal architectural superiority.
'''
    (ROOT/'docs/phase2_external_validation.md').write_text(narrative)
    verification={'status':'passed','verified_utc':datetime.now(timezone.utc).isoformat(),
                  'station_hours_per_model':int(preferred.n),'stations':12,'models':7,
                  'protocol_sha256':manifest['protocol_sha256'],
                  'predictions_sha256':sha256((run/'predictions.csv.gz').read_bytes()).hexdigest(),
                  'checks':['completed frozen protocol','reference model and evaluator hashes',
                            'all common station-hours and targets','declared one-hour target interval',
                            'recomputed pooled and station scores','archive provenance and pre-download declaration']}
    for folder in (run,destination):
        (folder/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':
    build()
