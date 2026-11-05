"""Build the Phase 2 IEEE report, editable deck, PDF, guide and executed notebook."""
from pathlib import Path
import argparse
import base64
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
sys.path.insert(0, str(ROOT / 'src'))
# Reuse already-installed document tooling without overriding the numerical environment.
if (ROOT / '.dependencies/documents').exists():
    sys.path.append(str(ROOT / '.dependencies/documents'))

import pandas as pd
import joblib
import build_presentation as layout
from build_phase2_report import build as build_figures, LABELS


def build_article_figures(run, output):
    """Draw at final IEEE column width so labels remain legible in print."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    from build_phase2_report import COLORS
    folder = output / 'figures'
    def save(fig, name):
        fig.savefig(folder / ('article_' + name + '.pdf'))
        fig.savefig(folder / ('article_' + name + '.png'), dpi=240)
        plt.close(fig)
    with plt.rc_context({'font.size': 7.2, 'axes.titlesize': 7.5, 'axes.labelsize': 7.2,
                         'legend.fontsize': 6.2, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.8,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        search = pd.read_csv(run / 'search_metrics.csv').sort_values('rmse_mean')
        short = {'level_24h_attention': 'Level 24h / attn', 'level_24h_longer_training': 'Level 24h / 10 epochs',
                 'level_48h_attention': 'Level 48h / attn', 'level_24h_wider': 'Level 24h / 64 units',
                 'level_24h_no_attention': 'Level 24h / no attn', 'delta_24h_no_attention': 'Delta 24h / no attn',
                 'delta_24h_longer_training': 'Delta 24h / 10 epochs', 'delta_48h_no_attention': 'Delta 48h / no attn'}
        fig, ax = plt.subplots(figsize=(3.5, 1.95), layout='constrained')
        ax.barh([short[name] for name in search.candidate], search.rmse_mean, color='#9333ea')
        ax.invert_yaxis()
        ax.set_xlabel('Pooled 2013 RMSE (µg/m³)')
        save(fig, 'validation_search')
        ablation = pd.read_csv(run / 'ablation_metrics.csv').sort_values('rmse_mean')
        fig, ax = plt.subplots(figsize=(3.5, 1.65), layout='constrained')
        names = [name.replace('bilstm', 'Bi-LSTM').replace('lstm', 'LSTM').replace('_no_attention', ' / no attn').replace('_attention', ' / attn') for name in ablation.candidate]
        ax.barh(names, ablation.rmse_mean, xerr=ablation.rmse_std, capsize=2, color='#0891b2')
        ax.invert_yaxis()
        ax.set_xlabel('2013 RMSE mean ± seed SD (µg/m³)')
        save(fig, 'attention_ablation')
        quarter = pd.read_csv(run / 'fold_metrics.csv').query("phase == 'holdout'")
        fig, ax = plt.subplots(figsize=(3.5, 2.1), layout='constrained')
        for name in ('persistence', 'ridge', 'xgboost', 'bilstm', 'lstm', 'hybrid'):
            values = quarter.loc[quarter.model == name].sort_values('fold')
            ax.plot(range(1, 5), values.rmse, marker='o', markersize=2, linewidth=.8,
                    label=LABELS[name].replace('Bi-LSTM + XGBoost', 'Hybrid'), color=COLORS[name])
        ax.set(xticks=[1, 2, 3, 4], xticklabels=['Q1', 'Q2', 'Q3', 'Q4'], ylabel='RMSE (µg/m³)')
        ax.legend(ncol=3, loc='upper right')
        save(fig, 'quarterly_rmse')
        regimes = pd.read_csv(run / 'errors_by_pollution_regime.csv')
        names = ['lstm', 'bilstm', 'hybrid', 'ridge', 'xgboost', 'persistence']
        fig, ax = plt.subplots(figsize=(3.5, 2.0), layout='constrained')
        for offset, regime, color, label in ((-.18, 'below_training_p90', '#94a3b8', 'Below 218'),
                                             (.18, 'above_training_p90', '#2563eb', 'At/above 218')):
            table = regimes.loc[regimes.pollution_regime == regime].set_index('model').reindex(names)
            ax.barh(np.arange(len(names)) + offset, table.rmse, height=.36, color=color, label=label)
        ax.set(yticks=np.arange(len(names)), yticklabels=['LSTM','Bi-LSTM','Hybrid','Ridge','XGBoost','Persistence'], xlabel='2014 RMSE (µg/m³)')
        ax.invert_yaxis()
        ax.legend(loc='lower right')
        save(fig, 'pollution_regimes')
        correction = pd.read_csv(output / 'correction_effect.csv').query("phase == 'holdout'")
        fig, ax = plt.subplots(figsize=(3.5, 1.8), layout='constrained')
        groups = ['all_hours', 'below_training_p90', 'above_training_p90']
        for offset, name, color, label in ((-.18, 'frozen_base', '#9333ea', 'Frozen base'),
                                           (.18, 'hybrid', '#ea580c', '+ Correction')):
            table = correction.loc[correction.model == name].set_index('pollution_regime').reindex(groups)
            ax.bar(np.arange(3)+offset, table.rmse, width=.36, color=color, label=label)
        ax.set(xticks=np.arange(3), xticklabels=['All hours','Below 218','At/above 218'], ylabel='2014 RMSE (µg/m³)')
        ax.legend(loc='upper left')
        save(fig, 'correction_effect')
        predictions = pd.read_csv(run / 'predictions.csv', parse_dates=['target_time'], low_memory=False)
        holdout = predictions.loc[predictions.phase == 'holdout']
        stop = holdout.target_time.max()
        grid = pd.date_range(stop - pd.Timedelta(days=7) + pd.Timedelta(hours=1), stop, freq='h')
        actual = holdout.loc[holdout.model == 'lstm'].set_index('target_time').reindex(grid)
        fig, ax = plt.subplots(figsize=(3.5, 2.0), layout='constrained')
        ax.plot(actual.index, actual.actual, color='#111827', linewidth=1, label='Observed')
        for name in ('lstm', 'hybrid', 'ridge'):
            table = holdout.loc[holdout.model == name].set_index('target_time').reindex(grid)
            ax.plot(table.index, table.prediction, linewidth=.6, alpha=.8,
                    color=COLORS[name], label='Hybrid' if name == 'hybrid' else LABELS[name])
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.set_ylabel('PM2.5 (µg/m³)')
        ax.legend(ncol=2, loc='upper left')
        save(fig, 'forecast_last_week')


def tex(value):
    return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def score_sentence(table, column='rmse'):
    return '; '.join(f'{LABELS.get(row.model, row.model)} {getattr(row, column):.3f}' for row in table.itertuples())


def build_paper(run, output, manifest, scores):
    holdout = scores.loc[scores.phase == 'holdout'].sort_values('rmse')
    values = holdout.set_index('model')
    selection = manifest['selection']
    configs = selection['neural_configurations']
    b = configs['bilstm']
    h = selection['hybrid']
    parameter_counts = {name: sum(parameter.numel() for parameter in
                                  joblib.load(run / f'{name}.joblib')['pipeline'].model.parameters())
                        for name in ('bilstm', 'lstm')}
    leader = holdout.iloc[0]
    stability = pd.read_csv(run / 'seed_stability.csv')
    regimes = pd.read_csv(run / 'errors_by_pollution_regime.csv')
    peak = regimes.loc[regimes.pollution_regime == 'above_training_p90'].set_index('model')
    low = regimes.loc[regimes.pollution_regime == 'below_training_p90'].set_index('model')
    correction = pd.read_csv(output / 'correction_effect.csv') if (output / 'correction_effect.csv').exists() else None
    abstract = (f'We extend a chronological Beijing PM2.5 forecasting study with independently trained LSTM and Bi-LSTM models, '
        f'temporal-attention ablations, three-seed validation and a calibrated XGBoost correction head. Eight recurrent configurations '
        f'are screened using 2013 data, followed by matched architecture/attention comparisons and hybrid selection. The final 2014 '
        f'comparison uses the same {int(leader.n):,} observed target hours for every method. Bi-LSTM achieves RMSE '
        f'{values.loc["bilstm", "rmse"]:.3f}, LSTM {values.loc["lstm", "rmse"]:.3f}, hybrid '
        f'{values.loc["hybrid", "rmse"]:.3f} and Ridge {values.loc["ridge", "rmse"]:.3f} micrograms per cubic meter. '
        f'The validation-based preference is {LABELS.get(selection["preferred_model"], selection["preferred_model"])}. '
        f'Saved predictions, fold boundaries, source/data hashes and model artifacts support inspection and reproduction. '
        f'The 2014 results remain exploratory because their Phase 1 counterparts were previously viewed.')
    selected_settings = (f'The screening winner is \\texttt{{{tex(selection["search_winner"])}}}. '
        f'The selected independent Bi-LSTM uses {b["seq_len"]} input hours, {b["hidden_dim"]} hidden units per direction, '
        f'{b["epochs"]} epochs, {"enabled" if b["use_attention"] else "disabled"} attention and a {b["target_mode"]} target. '
        f'The selected hybrid reserves {100*h["calibration_fraction"]:.0f}\\% of eligible origins for calibration and uses '
        f'correction weight {h["correction_weight"]:g}. Its head configuration is recorded in the selection JSON. '
        f'The selected Bi-LSTM and LSTM contain {parameter_counts["bilstm"]:,} and '
        f'{parameter_counts["lstm"]:,} trainable parameters, respectively. '
        f'The model preferred by mean validation RMSE is {tex(LABELS.get(selection["preferred_model"], selection["preferred_model"]))}.')
    discussion = (f'The lowest observed pooled 2014 RMSE is {leader.rmse:.3f} for {tex(LABELS[leader.model])}. '
        f'The independent Bi-LSTM RMSE is {values.loc["bilstm", "rmse"]:.3f}, compared with '
        f'{values.loc["persistence", "rmse"]:.3f} for persistence and {values.loc["ridge", "rmse"]:.3f} for Ridge. '
        f'The selected hybrid RMSE is {values.loc["hybrid", "rmse"]:.3f}. '
        'These are measured historical rankings, not evidence of statistical significance. The first corrected baseline used '
        'a shared reduced-training base, whereas this research run both changes model selection and trains standalone '
        'networks independently; differences between runs cannot be attributed to one factor alone.')
    stability_discussion = ('The seed table reports the variation of pooled 2013 errors for the selected configurations. '
        'Every seed uses the same target hours and chronological folds. A small seed standard deviation does not eliminate '
        'selection bias from repeatedly using one validation year. The 2014 table is a single predeclared seed, rather than '
        'a seed chosen after looking at final errors.')
    peak_discussion = (f'The independent Bi-LSTM has high-pollution RMSE {peak.loc["bilstm", "rmse"]:.3f}, '
        f'LSTM {peak.loc["lstm", "rmse"]:.3f}, hybrid {peak.loc["hybrid", "rmse"]:.3f}, '
        f'Ridge {peak.loc["ridge", "rmse"]:.3f} and persistence {peak.loc["persistence", "rmse"]:.3f}. '
        f'Below the threshold, Bi-LSTM and hybrid RMSE are {low.loc["bilstm", "rmse"]:.3f} and '
        f'{low.loc["hybrid", "rmse"]:.3f}. The corresponding peak biases are '
        f'{peak.loc["bilstm", "bias"]:.3f} and {peak.loc["hybrid", "bias"]:.3f}; negative bias indicates underprediction. '
        'This stratification describes where errors occur. It does not establish a causal reason for a model failure. '
        'A correction that helps ordinary conditions can still degrade the pollution tail. The selected correction weight '
        'and regularization are chosen from validation, with no revision motivated by these 2014 regime scores.')
    if correction is not None:
        correction = correction.loc[correction.phase == 'holdout'].set_index(['model', 'pollution_regime'])
        peak_discussion += (f' Isolating the correction on the same frozen base, pooled RMSE changes from '
            f'{correction.loc[("frozen_base", "all_hours"), "rmse"]:.3f} to '
            f'{correction.loc[("hybrid", "all_hours"), "rmse"]:.3f}; below-threshold RMSE changes from '
            f'{correction.loc[("frozen_base", "below_training_p90"), "rmse"]:.3f} to '
            f'{correction.loc[("hybrid", "below_training_p90"), "rmse"]:.3f}, while peak RMSE changes from '
            f'{correction.loc[("frozen_base", "above_training_p90"), "rmse"]:.3f} to '
            f'{correction.loc[("hybrid", "above_training_p90"), "rmse"]:.3f}. '
            'This isolates the contribution of the correction from differences in standalone training history.')
    conclusion = (f'Phase 2 delivers independent recurrent benchmarks, validation selection, attention and directionality ablations, '
        f'seed stability, chronological residual calibration and runnable saved models. Mean 2013 validation RMSE prefers '
        f'{tex(LABELS.get(selection["preferred_model"], selection["preferred_model"]))}; the lowest observed 2014 pooled RMSE '
        f'is {leader.rmse:.3f} for {tex(LABELS[leader.model])}. '
        'The result is conditional on this archive, feature set and finite search. The scientific outcome is the measured '
        'comparison, including unsuccessful candidates and peak behavior, rather than an assumption that added complexity wins. '
        'A fresh untouched evaluation and broader data are required before making new generalization claims.')
    result_rows = '\n'.join(f'{tex(LABELS[row.model])} & {row.mae:.3f} & {row.rmse:.3f} & {row.r2:.4f} \\\\'
                            for row in holdout.itertuples())
    stability_rows = '\n'.join(f'{tex(LABELS[row.candidate])} & {row.rmse_mean:.3f} & {row.rmse_std:.3f} \\\\'
                               for row in stability.itertuples())
    replacements = {'ABSTRACT': tex(abstract), 'SELECTED_SETTINGS': selected_settings,
                    'RESULT_ROWS': result_rows, 'RESULT_DISCUSSION': discussion,
                    'STABILITY_ROWS': stability_rows, 'STABILITY_DISCUSSION': stability_discussion,
                    'PEAK_DISCUSSION': peak_discussion, 'CONCLUSION': conclusion}
    template = (ROOT / 'reports/phase2/report_template.tex').read_text()
    for key, value in replacements.items():
        template = template.replace('@@' + key + '@@', value)
    if '@@' in template:
        raise ValueError('Unfilled report template field')
    (output / 'Aircast_Phase2_Report.tex').write_text(template)
    (output / 'research_summary.md').write_text(
        '# Phase 2 selected research findings\n\n' + abstract + '\n\n' +
        '2014 RMSE: ' + score_sentence(holdout) + '.\n\n' +
        peak_discussion.replace('\\%', '%') + '\n\n' + conclusion.replace('\\%', '%') + '\n')


class Phase2Deck(layout.Deck):
    def __init__(self, output):
        layout.OUTPUT = output
        layout.TOTAL_SLIDES = 12
        super().__init__()
        self.prs.core_properties.title = 'Aircast | Phase 2: Independent Neural and Hybrid Forecasting'
        self.prs.core_properties.subject = 'Validation selection, attention ablations, seed stability and exploratory historical results'
        self.prs.core_properties.keywords = 'PM2.5, Phase 2, LSTM, Bi-LSTM, attention, hybrid, chronological validation'
        self.pdf.setTitle(self.prs.core_properties.title)
        self.output = output

    def start(self, *args, **kwargs):
        kwargs.setdefault('source', 'CSAI PROJECT 19 / PHASE 2 · DEEP LEARNING AND HYBRID FORECASTING')
        return super().start(*args, **kwargs)

    def save(self):
        if self.page != 12:
            raise ValueError('The Phase 2 deck must contain twelve slides')
        self.prs.save(self.output / 'Aircast_Phase2_Presentation.pptx')
        self.pdf.save()
        (self.output / 'presenter_guide.md').write_text(
            '# Phase 2 presentation speaker guide\n\nThe same notes are embedded in the editable PowerPoint. '
            'Explain the 2014 exploratory status, validation-only selection, seed variation and peak errors. '
            'No individual contributions or email addresses are inferred.\n\n' + '\n'.join(self.notes))


def build_deck(run, output, manifest, scores):
    d = Phase2Deck(output)
    val = scores.loc[scores.phase == 'validation'].set_index('model')
    holdout = scores.loc[scores.phase == 'holdout'].sort_values('rmse')
    selection = manifest['selection']
    b, h = selection['neural_configurations']['bilstm'], selection['hybrid']
    green, muted, ink = layout.GREEN, layout.MUTED, layout.INK
    d.start('Research question', dark=True)
    d.text('Beyond the latest reading.', 56, 127, 1160, 55, layout.WHITE, True)
    d.text('Independent recurrent models and calibrated hybrid corrections\nfor one-hour PM2.5 forecasts.', 60, 225, 1050, 30, 'BED1D2')
    d.text('Measured comparisons. Validation-based selection. Explicit limitations.', 60, 340, 1120, 24, layout.MINT, True)
    for i, (name, number) in enumerate(layout.TEAM):
        d.text(f'{name}  /  {number}', 60, 418 + 34*i, 1100, 20, layout.WHITE)
    d.note('Scope and team', 'Phase 2 asks whether independent recurrent networks and a calibrated correction improve on strong classical references. The team is carried forward exactly from the existing project. All 2014 results are exploratory because the year was already inspected in Phase 1.')

    d.start('01 / Audit and repairs', 'Fix the experiment before interpreting scores', 'The initial code and its benchmark needed separate scientific and inference corrections.')
    items = [('Training boundary', 'Fit preprocessing only on actual training windows; never learn from prediction batches.'),
             ('Independent benchmark', 'Standalone LSTM and Bi-LSTM use the full eligible outer training window.'),
             ('Honest hybrid targets', 'Freeze the base before a later calibration block; fit only out-of-sample residuals.'),
             ('Verified inference', 'Share sequence-aware CLI/dashboard prediction and verify saved model hashes.')]
    for i, (title, detail) in enumerate(items):
        y = 220 + 98*i
        d.text(title, 72, y, 320, 24, green, True)
        d.text(detail, 420, y, 770, 22, ink)
    d.note('Audit findings', 'The first audit fixed full-dataset scaling, batch-dependent imputation, ignored masks, dropped wind direction, in-sample residual fitting and suppressed errors. This continuation also fixed bidirectional state extraction without attention and added independent full-window benchmarks. A zero or invalid training mask fails rather than silently relaxing the boundary.')

    d.start('02 / Information boundaries', 'Observed history in. A future concentration out.', 'All hours follow the Beijing local clock; the target is one hour after the origin.')
    for x, title, detail in [(56, 'TRAIN + PREPROCESS', '2010–2012 initially\nAvailable labels only\nPreserve gaps and valid peaks'),
                              (454, 'SELECT ON 2013', 'Four expanding quarters\nScreen candidates + ablations\nThree predefined seeds'),
                              (852, 'COMPARE ON 2014', 'Freeze selection first\nSame observed target hours\nExploratory historical result')]:
        d.panel(x, 235, 372, 250, title)
        d.text(detail, x+22, 313, 328, 24, ink)
    d.text('No target-hour measured weather. No random train/test split. No filled evaluation labels.', 64, 540, 1150, 24, green, True)
    d.note('Data and timing', 'UCI 381 has 43,824 hourly rows and 2,067 missing concentrations. The same engineered features and common evaluation mask are used for all models. At each quarterly boundary, only labels available at the first forecast origin enter training. 2014 remains exploratory even though individual fits respect chronology.')

    d.start('03 / Neural formulation', 'A sequence model can predict a level or a change', 'The delta formulation adds a learned correction to the latest observed concentration.')
    d.panel(56, 225, 568, 325, 'Recurrent representation')
    d.text('24 or 48 observed hours\nLSTM or Bi-LSTM\n32 or 64 hidden units per direction\nAttention or final hidden states', 80, 300, 520, 25, ink)
    d.panel(648, 225, 576, 325, 'Standardized target + reconstruction')
    d.text('Level: next concentration\nDelta: next minus current concentration\nTraining-only target statistics\nReconstruct and clip at zero', 674, 300, 520, 25, ink)
    d.text('Without attention, both terminal Bi-LSTM states summarize the full observed window.', 64, 594, 1150, 23, green, True)
    d.note('Neural architecture', 'Input and target scaling are fitted only on training windows. Delta targets use current pollution as a fixed offset; missing current pollution uses its saved training median. Standardized Huber loss, AdamW, gradient clipping and fixed epochs train the model. The bidirectional network reads backward within the observed window, never beyond the forecast origin. Directionality comparisons have matching hidden width but different parameter counts.')

    d.start('04 / Candidate screening', 'Eight candidates. One validation year.', 'Sequence length, width, duration, attention and target formulation; screening seed fixed at 42.')
    d.figure('validation_search', 62, 214, 1150, 420)
    d.note('Candidate search', 'The complete grid is saved before results are inspected. Initial screening uses seed 42 on all four 2013 quarters. This is a finite staged search, not exhaustive optimization or nested cross-validation. Unsuccessful candidates remain in the CSV and checkpoint outputs. The winning template supplies the matched ablations.')

    d.start('05 / Matched ablations', 'Does attention or a backward direction help?', 'Each matched variant is run at seeds 42, 137 and 2026 on identical validation hours.')
    d.figure('attention_ablation', 62, 212, 1150, 420)
    d.note('Ablations and seed variability', 'Compare LSTM and Bi-LSTM with attention enabled and disabled at the selected template settings. Selection uses mean pooled validation RMSE across seeds. Error bars show sample standard deviation across seeds; they are not confidence intervals. Attention weights do not establish causal feature importance.')

    d.start('06 / Hybrid selection', 'Calibrate later. Shrink the correction.', f'Selected calibration: {100*h["calibration_fraction"]:.0f}% of eligible origins; correction weight: {h["correction_weight"]:g}.')
    d.figure('hybrid_selection', 62, 212, 1150, 420)
    d.note('Hybrid calibration', 'Reserve the last 10% or 20% of training origins for calibration. The base is trained earlier with a label-availability gap and then frozen. Two XGBoost settings and weights 0, .25, .5 and 1 are scored using three validation seeds. The zero-weight control can reject an unhelpful correction. The hybrid intentionally has less base-training data than the independent neural benchmark.')

    d.start('07 / Historical comparison', 'Report every model on the same target hours', f'{int(holdout.iloc[0].n):,} common 2014 hours. RMSE and MAE in µg/m³. Comparison seed fixed before scoring.')
    columns = [(78, 'Model', 360), (480, 'MAE', 160), (710, 'RMSE', 160), (940, 'R²', 160)]
    for x, label, width in columns:
        d.text(label, x, 220, width, 22, green, True)
    d.line(72, 259, 1200, 259)
    for i, row in enumerate(holdout.itertuples()):
        y = 281 + i*43
        for (x, _, width), value in zip(columns, (LABELS[row.model], f'{row.mae:.3f}', f'{row.rmse:.3f}', f'{row.r2:.4f}')):
            d.text(value, x, y, width, 22, ink, i == 0)
    d.text(f'Validation-based model preference: {LABELS[selection["preferred_model"]]}.', 78, 617, 1120, 21, green, True)
    d.note('Final score table', 'These rows use the identical observed 2014 targets. Classical predictions are reused from the audited Phase 1 experiment and checked for exact origins, labels and fold alignment. The preferred model comes from validation, not this final ranking. Small differences do not establish significance, and a lower RMSE does not guarantee a lower MAE or better peak behavior.')

    d.start('08 / Quarter and regime', 'A pooled score hides changing behavior', 'Quarterly and pollution-regime diagnostics retain identical hours within each comparison.')
    d.figure('quarterly_rmse', 56, 225, 568, 365)
    d.figure('correction_effect', 650, 225, 574, 365)
    d.text('High pollution: at least 218 µg/m³, fixed from the 2010–2012 training distribution.', 64, 617, 1150, 21, green, True)
    d.note('Failure analysis', 'The high-pollution threshold is fixed using initial training data only. There are 865 peak hours and 7,666 lower-pollution hours in the common 2014 set. Compare peak RMSE and bias with Ridge and persistence, not just with a weaker neural base. Seasonal changes and differences between full and reduced training windows complicate attribution; the diagnostics are descriptive, not causal.')

    d.start('09 / Stability and forecast trace', 'Repeatability and a concrete forecast sequence', 'The last week is selected by time; missing target hours remain visible as gaps.')
    d.figure('seed_stability', 56, 222, 566, 370)
    d.figure('forecast_last_week', 650, 222, 574, 370)
    d.text('Predeclared deployment seed 42. Seed variation is measured on validation.', 64, 617, 1150, 21, green, True)
    d.note('Seed stability and example trace', 'The left panel shows variation across the three predefined validation seeds for selected model configurations. The right panel shows a time-selected forecast interval rather than a hand-picked success. We do not choose a deployment seed using holdout performance. No calibrated prediction intervals are supplied.')

    d.start('10 / Reproducible demonstration', 'Saved models, shared features, checked results', 'The notebook and package retain enough evidence to inspect and repeat the experiment.')
    for i, (title, detail) in enumerate([
        ('Traceable experiment', 'Input/source/model hashes, frozen selection, seed scores and fold audits.'),
        ('Runnable local demo', 'Independent Bi-LSTM, LSTM and hybrid alongside classical models.'),
        ('Consistent forecasts', f'CLI and dashboard share features and retain 168 + {b["seq_len"]} raw hourly slots.'),
        ('Submission artifacts', 'IEEE report, editable slides, slide PDF, notebook and packaged weights.')]):
        d.text(title, 72, 224+98*i, 340, 24, green, True)
        d.text(detail, 430, 224+98*i, 760, 22, ink)
    d.note('Demo and reproducibility', 'The local dashboard verifies model hashes from a completed real-data manifest before exposing models. The hosted classical deployment does not include PyTorch. The notebook loads measured results by default and has an explicit switch to rerun the complete experiment. The archive excludes environments, credentials and training checkpoints, and includes source, model weights, the attributed dataset and comparison evidence.')

    d.start('11 / Conclusion and discussion', 'The measured result is the contribution.', dark=True)
    d.text(f'Validation preference: {LABELS[selection["preferred_model"]]}', 60, 223, 1120, 35, layout.MINT, True)
    d.text('Independent benchmarks, honest calibration, matched ablations\nand reproducible evidence are now in place.', 60, 300, 1100, 30, layout.WHITE)
    d.text('2014 is exploratory. New blind accuracy claims need untouched data.', 60, 421, 1120, 25, 'BED1D2')
    d.text('Thank you. Questions?', 60, 535, 1120, 42, layout.MINT, True)
    d.note('Conclusions and viva prompts', 'Explain why the standalone model is independent, why calibration residuals must be out of sample, why the bidirectional window is causal, why all models share a target mask, and why 2014 remains exploratory. Discuss seed variation without claiming a confidence interval. Do not assume a hybrid must beat Ridge. All five team members and enrollment numbers are listed on the opening slide, with no invented contribution allocations.')
    d.save()


def build_notebook(output):
    cells = []
    def md(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)})
    def code(source):
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': source.splitlines(keepends=True),
                      'execution_count': None, 'outputs': []})
    md('# Aircast Phase 2: independent recurrent models and calibrated hybrid\n\n'
       'This notebook inspects the measured experiment and runs saved-model inference. '
       'Set `RUN_FULL_RESEARCH=True` to repeat training into a fresh directory. '
       'Model selection uses 2013. The 2014 comparison is exploratory because its Phase 1 results were already known.\n')
    code("from pathlib import Path\nimport os, sys, json\nROOT = Path.cwd()\nif not (ROOT / 'src/air_prediction').exists():\n    ROOT = ROOT.parent\nassert (ROOT / 'src/air_prediction').exists(), 'Open this notebook from the project checkout'\nos.chdir(ROOT)\nfor key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):\n    os.environ.setdefault(key, '1')\nos.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))\nsys.path.insert(0, str(ROOT / 'src'))\ntry:\n    from IPython.display import display\nexcept ImportError:\n    display = print\nprint('Project:', ROOT)\n")
    md('## Environment\n\nUse Python 3.13 and `requirements-phase2.txt` to match the recorded models. '
       'On macOS, XGBoost also needs `brew install libomp`. Installation is opt-in below.\n')
    code("INSTALL_DEPENDENCIES = False\nif INSTALL_DEPENDENCIES:\n    import subprocess\n    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-phase2.txt'])\nimport pandas as pd\nfrom air_prediction.config import load_config\nfrom air_prediction.phase2_research import load_research_settings, run_research\nfrom air_prediction.predict import predict_latest\nfrom air_prediction.data import load_dataset\nCONFIG = load_config(ROOT / 'configs/phase1.toml')\nSETTINGS = load_research_settings(ROOT / 'configs/phase2_research.toml')\nDATA = ROOT / 'data/raw/PRSA_data_2010.1.1-2014.12.31.csv'\nprint(load_dataset(DATA).audit)\n")
    md('## Reproduce the experiment\n\nThe full run trains eight screening candidates, paired attention/directionality variants at three seeds, '
       'and chronological hybrid candidates. Standalone neural models use every eligible training row. '
       'The hybrid base is frozen before its calibration block. Completed runs are never overwritten.\n')
    code("RUN_FULL_RESEARCH = False\nRUN = ROOT / 'artifacts/phase2_research'\nif RUN_FULL_RESEARCH:\n    from datetime import datetime\n    RUN = ROOT / 'artifacts' / ('phase2_notebook_' + datetime.now().strftime('%Y%m%d_%H%M%S'))\n    run_research(DATA, CONFIG, RUN, ROOT / 'artifacts/phase1', SETTINGS)\nassert (RUN / 'run_manifest.json').exists(), 'Run training or extract the submission package with saved artifacts'\nmanifest = json.loads((RUN / 'run_manifest.json').read_text())\nassert manifest['status'] == 'complete'\nassert load_dataset(DATA).audit['sha256'] == manifest['dataset_audit']['sha256']\nselection = json.loads((RUN / 'selection.json').read_text())\nassert selection['holdout_used_for_selection'] is False\nprint('Run:', RUN)\nprint('Validation preference:', selection['preferred_model'])\nprint('Selected configurations:', selection['neural_configurations'])\n")
    md('## Verify evidence and inspect scores\n\nEvery model uses identical target hours; the deployment fits are separate from the fits used for scoring.\n')
    code("from hashlib import sha256\nfrom air_prediction.metrics import score_predictions\nassert sha256((RUN / 'selection.json').read_bytes()).hexdigest() == manifest['selection_sha256']\nfor name, expected in manifest['model_sha256'].items():\n    assert sha256((RUN / (name + '.joblib')).read_bytes()).hexdigest() == expected\npredictions = pd.read_csv(RUN / 'predictions.csv', parse_dates=['origin_time', 'target_time'], low_memory=False)\nmetrics = pd.read_csv(RUN / 'metrics.csv')\npd.testing.assert_frame_equal(score_predictions(predictions), metrics, check_exact=False, rtol=1e-10)\ndisplay(metrics)\n")
    code("for name in ('search_metrics.csv', 'ablation_metrics.csv', 'seed_stability.csv', 'errors_by_pollution_regime.csv'):\n    print('\\n' + name)\n    display(pd.read_csv(RUN / name))\n")
    md('## Figures\n\nSeed error bars describe variation across predefined seeds and are not confidence intervals.\n')
    code("FIGURES = ROOT / 'docs/results/phase2_research'\nif RUN_FULL_RESEARCH:\n    import subprocess\n    FIGURES = RUN / 'report'\n    subprocess.check_call([sys.executable, 'scripts/build_phase2_report.py', '--run', str(RUN), '--output', str(FIGURES)])\ntry:\n    from IPython.display import Image as NotebookImage\nexcept ImportError:\n    NotebookImage = None\nfor name in ('model_comparison', 'attention_ablation', 'pollution_regimes'):\n    path = FIGURES / (name + '.png')\n    if NotebookImage is not None:\n        display(NotebookImage(filename=str(path)))\n    else:\n        print(path)\n")
    md('## Run the saved local models\n\nThe sample ends at 2014-12-31 23:00; forecasts refer to the following hour. '
       'These are historical examples, not live air quality or a new measured accuracy result.\n')
    code("for name in manifest['models']:\n    result = predict_latest(RUN / (name + '.joblib'), DATA)\n    assert pd.notna(result['pm25_ug_m3'])\n    print(json.dumps(result, indent=2))\n")
    md('## Verification and demonstration\n\nRun `pytest -q -W error::DeprecationWarning` for automated checks and '
       '`bash scripts/start_frontend.sh` for the local dashboard. '
       'The existing hosted classical deployment remains separate.\n\n'
       'Dataset: UCI Beijing PM2.5, DOI 10.24432/C5JS49, CC BY 4.0. '
       'Methods, measured limitations, references and the original team details are in the Phase 2 report.\n')
    namespace = {'__name__': '__notebook__'}
    count = 0
    for cell in cells:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            exec(compile(''.join(cell['source']), f'<phase2 notebook cell {count}>', 'exec'), namespace)
        cell['execution_count'] = count
        if captured.getvalue():
            cell['outputs'].append({'output_type': 'stream', 'name': 'stdout',
                                    'text': captured.getvalue().splitlines(keepends=True)})
        if 'for name in (\'model_comparison\'' in ''.join(cell['source']):
            for name in ('model_comparison', 'attention_ablation', 'pollution_regimes'):
                cell['outputs'].append({'output_type': 'display_data', 'metadata': {}, 'data': {
                    'image/png': base64.b64encode((namespace['FIGURES'] / (name + '.png')).read_bytes()).decode(),
                    'text/plain': [name]}})
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': sys.version.split()[0]}}}
    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'phase2-{index:02d}'
    path = ROOT / 'notebooks/Air_Prediction_Phase2.ipynb'
    path.write_text(json.dumps(notebook, indent=1) + '\n')
    print(f'Executed {count} notebook cells and saved {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=str(ROOT / 'artifacts/phase2_research'))
    parser.add_argument('--output', default=str(ROOT / 'reports/phase2'))
    parser.add_argument('--no-compile', action='store_true')
    args = parser.parse_args(argv)
    run, output = Path(args.run).resolve(), Path(args.output).resolve()
    manifest = json.loads((run / 'run_manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('is_synthetic', True):
        raise ValueError('Submission documents require a completed real-data research run')
    compact = ROOT / 'docs/results/phase2_research'
    build_figures(run, compact)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'figures').mkdir(exist_ok=True)
    for path in compact.iterdir():
        if path.suffix in {'.png', '.pdf'}:
            shutil.copyfile(path, output / 'figures' / path.name)
    shutil.copyfile(compact / 'correction_effect.csv', output / 'correction_effect.csv')
    build_article_figures(run, output)
    scores = pd.read_csv(run / 'metrics.csv')
    build_paper(run, output, manifest, scores)
    build_deck(run, output, manifest, scores)
    build_notebook(output)
    if not args.no_compile:
        environment = os.environ.copy()
        environment['TECTONIC_CACHE_DIR'] = str(ROOT / '.cache/tectonic')
        subprocess.run(['tectonic', '--keep-logs', '--outdir', str(output),
                        str(output / 'Aircast_Phase2_Report.tex')], cwd=output,
                       env=environment, check=True)
    print(f'Built Phase 2 submission documents in {output}')


if __name__ == '__main__':
    main()
