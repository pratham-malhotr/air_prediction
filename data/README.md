# Dataset provenance

Use UCI **Beijing PM2.5**, dataset 381, explicitly selected for this project:
https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data

Expected file: `raw/PRSA_data_2010.1.1-2014.12.31.csv`.

Both datasets are committed to the repository's `main` branch:

| File | Contents |
|---|---|
| [Raw dataset](raw/PRSA_data_2010.1.1-2014.12.31.csv) | The original 43,824 hourly records, with unchanged source bytes |
| [Feature-engineered dataset](processed/feature_engineered.csv) | 41,612 supervised rows with 70 features, `origin_time`, `target_time`, and `target_pm25` |
| [Feature dictionary](processed/feature_dictionary.csv) | Description and observation availability for each feature |
| [Export provenance](processed/feature_engineered.provenance.json) | Source/export checksums, feature configuration, row-selection policy, and dataset attribution |

The engineered export predicts PM2.5 one hour after `origin_time`. It includes rows after the 168-hour history warmup with an observed future target. Missing feature inputs remain empty; wind direction remains categorical. Imputation, scaling, and wind encoding are fitted inside each training fold. `target_pm25` and the timestamps are metadata/labels, not additional model inputs. Keep rows in chronological order and use the configured time splits rather than a random split.

Regenerate the full export from the repository root, using the installed project environment:

```bash
PYTHONPATH=src python scripts/prepare_data.py --feature-output data/processed/feature_engineered.csv
```

The preparation artifacts also include `feature_preview.csv`, which contains only 48 example rows. `cleaned_hourly.csv.gz` is the cleaned observation series; the full engineered dataset is `data/processed/feature_engineered.csv`.

Dataset citation: Chen, S. (2015), *Beijing PM2.5*, UCI Machine Learning Repository, DOI [10.24432/C5JS49](https://doi.org/10.24432/C5JS49). Dataset license: CC BY 4.0. Credit the source when redistributing the CSV or its derivatives. This dataset license is not a declaration of a separate license for all project software.

The downloader checks the official archive's documented schema, row count, and complete 2010–2014 hourly timestamp grid before saving the CSV and its SHA256/provenance sidecar. A checksum records the file used; it is not an independently published authenticity signature. A manual CSV without a sidecar is labeled `user_supplied` and its checksum is still recorded in the run audit.

Do not remove missing rows before making lags. The loader inserts missing hourly slots and leaves all values unfilled. Timestamp values use the Beijing local clock. The PM2.5 and weather observation locations differ.

Synthetic fixtures are generated separately under a demo run directory and have `source=synthetic` in their provenance. They are never downloaded as the official dataset and never substituted when a real download fails.

The analysed file was supplied locally and copied without changing its bytes. Its schema and complete hourly coverage match the documented archive. The provenance sidecar records `source=uci381_user_supplied`, the dataset citation, the acquisition method, and SHA256 `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c`. This records the actual file used; it does not claim an independent byte-for-byte comparison with a newly downloaded UCI copy.
