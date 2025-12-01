## Project Overview

This repository prepares structured electronic health record (EHR) data and trains classical ensemble models to predict multiple cardiovascular outcomes (AF, AS, MI, CI). The workflow covers patient filtering, leakage removal, balanced sampling, cross-validation materialization, model fitting, post-hoc evaluation, and visualization of threshold/ confidence behavior. Five core modules underpin the project:

1. `config/` – scenario-specific configuration namespaces that hold file paths, feature lists, and plotting constants.
2. `dataloader/` – patient filtering, leakage removal, balanced sampling, and dataset serialization.
3. `models/` – TF-IDF feature extraction plus Gradient Boosting/LightGBM/XGBoost classifiers, ROC-based metrics, and JSON artifacts.
4. `utils/` – reusable helpers (logging, JSON I/O, plotting routines) and metric post-processing.
5. `main.py` – the single CLI entry that dispatches to any preparation, training, evaluation, or plotting mode.

**Tech stack:** Python ≥3.9, NumPy, pandas, scikit-learn, LightGBM, XGBoost, SciPy, matplotlib, joblib.


## Project Structure

| Path | Description |
| ---- | ----------- |
| `config/` | `config.py` defines `SimpleNamespace` objects per mode (data prep, CV generation, training, plotting) and exposes `MODE_CONFIGS`, `AVAILABLE_MODES`. |
| `dataloader/` | `data_loader.py` houses `prepare_patient_info`, `prepare_data`, `create_train_test`, and CSV/PKL generators for cross-validation/test assets. |
| `models/` | `model.py` contains training, evaluation, threshold metric generation, statistical aggregation, and confidence artifact preparation. |
| `utils/` | Visualization and helper scripts (`plot_risk_confidence_mcf*.py`, `plot_risk_threshold_metrics.py`, statistical plotters, `utils.py` for logging/helpers). |
| `main.py` | CLI entry; parses `--run_mode`, validates availability, loads the matching config, and executes the mapped callable. |
| `translation_mapping.md` | Historical mapping notes between Chinese and English keys. |

Key files:

- `main.py`: `python main.py --run_mode <mode>` orchestrates every workflow.
- `config/config.py`: authoritative place for data paths, feature lists, run parameters, plotting constants.
- `dataloader/data_loader.py`: input CSV readers and pickle writers.
- `models/model.py`: training/inference utilities, JSON writers, ROC/threshold calculators.
- `utils/utils.py`: logging setup, helper math, JSON serialization.
- Plot scripts under `utils/`: standalone visualization entry points (confidence scatter, threshold curves, logistic boundary fitting, etc.).


## Core Mathematical Logic

- **TF-IDF vectorization + tree ensembles:** Each patient row is converted to a textual observation string that concatenates categorical features. `TfidfVectorizer` maps this string into a sparse vector space. Gradient Boosting, LightGBM, or XGBoost classifiers learn decision surfaces over these vectors. The probability output is `p(y=1|x) = \mathrm{model}(\mathrm{tfidf}(x))`.
- **ROC, AUROC, and density-driven threshold search:** `generate_threshold_metrics` computes ROC curves via `roc_curve`, integrates to AUROC, and enumerates thresholds with resolution `1/density` to log accuracy, precision (PPV), sensitivity (TPR), and F1 across decision boundaries.
- **Logistic regression confidence boundary:** Plot scripts apply feature mapping \(\phi(x_1, x_2)\) and optimize the logistic loss \(J(\theta) = -\frac{1}{m}\sum y\log h_\theta + (1-y)\log (1-h_\theta) + \lambda \|\theta_{1:}\|^2 / (2m)\) to separate high/low confidence regions, then fit polynomial/linear curves serving as interpretable risk thresholds.
- **Statistical coefficient aggregation:** Confidence/scoring artifacts compute density-binned statistics (mean, variance, weighted sums) to inspect calibration quality and to combine metrics such as `accuracy × sensitivity`.


## Scripts & Functions

### Data preparation
- `dataloader/data_loader.py`
  - `prepare_patient_info(cfg)`: reads ICD csvs, filters AF patients, renames columns, caches `PATIENTID`/`STAYID` pickles.
  - `prepare_data(cfg)`: loads raw semicolon CSV, removes leakage codes inside `DIAGNOSIS_HISTORY`, drops `PREDICTION_TARGET`, serializes cleaned DataFrame.
  - `create_train_test(cfg)`: labels positives via overlap with patient info, balances control/ case ratio (1:5), emits refreshed `df_training_AF4.csv` & `df_eval_AF4.csv`.
  - `generate_cross_training_csv(cfg)`: removes low-quality `ADMISSION_YEAR_GROUP`, stratifies by patient IDs, writes fold-specific train/validation CSVs.
  - `generate_common_testing_csv(cfg)`: filters and persists the held-out evaluation CSV.
  - `generate_cross_data_pickle(cfg)`: calls `csv_to_data_frame` to convert CSVs into TF-IDF-ready observation strings, dumping `(train_df, test_df)` pickles per fold.

### Model training & evaluation
- `models/model.py`
  - `train_ensemble_models(cfg)`: clones a `TfidfVectorizer`, trains per fold with the selected algorithm (`LGBM`, `GBDT`, `XGBt`), and saves `(vectorizer, model, labels, predictions)` via joblib.
  - `reevaluate_models(cfg)`: reloads artifacts, recomputes evaluations if new metrics are needed.
  - `generate_threshold_metrics(cfg)`: aggregates predictions over folds/random states, bins via density, writes JSON threshold dictionaries.
  - `print_threshold_evaluation`, `compute_statistical_coefficients`, `prepare_confidence_artifacts`: consume stored metrics to emit CSV/JSON summaries for reporting.

### Visualization utilities
- `utils/plot_risk_confidence_mcf*.py`, `plot_risk_confidence_mcf_scf.py`, `plot_risk_confidence_mcf_isp.py`: scatter risk vs. confidence, overlay fitted boundaries (linear/quadratic/polynomial) and highlight misclassified vs. correct samples.
- `utils/plot_risk_threshold_metrics.py`: plots accuracy, precision, recall, F1, and `accuracy × sensitivity` over classification thresholds.
- `utils/plot_statistical_coefficients.py`: depicts statistical coefficient curves produced by `compute_statistical_coefficients`.
- `utils/plot_logistic_nonlinear_boundary_mcf_scf.py`: runs the full logistic fitting pipeline described earlier.

### Utility helpers
- `utils/utils.py`: logging factory, JSON serialization (`load_json_data`, `save_json_data`), probability weighting helpers, and reporting utilities (`test_data_label_index`, `calc_pred_pro_weighted_sum`, etc.).


## Running Instructions

### Environment & dependencies

```bash
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scikit-learn lightgbm xgboost scipy matplotlib joblib
```

If you track dependencies centrally, create a `requirements.txt` with the packages above and run `pip install -r requirements.txt`.

### Hyperparameters & CLI options

| Argument / Config | Location | Description | Typical values |
| ----------------- | -------- | ----------- | -------------- |
| `--run_mode` | CLI + `config/config.py` | Selects which pipeline stage to execute. | `af_data_prep`, `generate_cross_training_csv`, `generate_common_testing_csv`, `generate_cross_data_pickle`, `train_models`, `reevaluate_models`, `generate_threshold_metrics`, `print_threshold_evaluation`, `compute_statistical_coefficients`, `prepare_confidence_artifacts`. |
| `cfg.model_name` | `train_model_config` | Chooses classifier class. | `LGBM`, `GBDT`, `XGBt`. |
| `cfg.folds` | configs | Number of CV folds used throughout. Default `6`. |
| `cfg.great_random_state` / `ADAPTIVE_RANDOM_STATES` | configs | Seeds used for shuffling/metric sampling per label. |
| `cfg.density` | metric/plot configs | Controls discretization of thresholds/confidence curves. |
| `cfg.features_training` | cross-data config | Ordered list of categorical columns concatenated into TF-IDF observations. |

### Typical workflow

1. **Prepare raw data**
   ```bash
   python main.py --run_mode af_data_prep
   python main.py --run_mode generate_cross_training_csv
   python main.py --run_mode generate_common_testing_csv
   python main.py --run_mode generate_cross_data_pickle
   ```
2. **Train / evaluate models**
   ```bash
   python main.py --run_mode train_models          # fits TF-IDF + ensemble models per fold
   python main.py --run_mode reevaluate_models     # recompute metrics if needed
   python main.py --run_mode generate_threshold_metrics
   python main.py --run_mode compute_statistical_coefficients
   python main.py --run_mode prepare_confidence_artifacts
   ```
3. **Visualize**
   Run any plot script directly, e.g.
   ```bash
   python utils/plot_risk_confidence_mcf.py
   python utils/plot_risk_threshold_metrics.py
   python utils/plot_logistic_nonlinear_boundary_mcf_scf.py
   ```

### Mode routing

`main.py` maps each `--run_mode` to the corresponding function inside `dataloader.data_loader` or `models.model`. The CLI validates your choice against `AVAILABLE_MODES`; unsupported values raise a descriptive error before execution.


## Notes

- **Data schema:** All CSV/PKL columns must already use English keys such as `PATIENTID`, `STAYID`, `ADMISSION_YEAR_GROUP`, `PREDICTION_TARGET`, `DIAGNOSIS_HISTORY`, etc. Legacy Chinese keys are no longer recognized and will trigger `KeyError`.
- **File paths:** Update the CSV/PKL directories referenced in `config/config.py` (`disease_source_csv`, `cross_training_csv_pickle`, `plot_data_archive`, etc.) to match your filesystem.
- **Common issues:**
  - `KeyError: 'PATIENTID'` – confirm inputs were translated and that semicolon CSVs contain the expected headers.
  - `FileNotFoundError` for pickle/CSV paths – double-check `cfg` paths, ensure folders exist, and Windows backslashes are escaped correctly.
  - Memory pressure when plotting dense scatter figures – reduce `const.density` inside plotting scripts.
- **Extensibility tips:** place new model trainers inside `models/model.py` or add separate modules that follow the same TF-IDF input contract. Register new run modes by extending `MODE_CONFIGS`, adding a callable to `TASKS`, and exposing it through `AVAILABLE_MODES`. Visualizations should reside in `utils/` and reuse the JSON artifacts written by the training pipeline.


