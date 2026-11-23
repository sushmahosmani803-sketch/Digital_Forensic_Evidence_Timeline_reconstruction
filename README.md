# AI-Based Digital Evidence Timeline Reconstruction

Rebuilds 4,615 insider-threat log events (CERT dataset: logons, USB devices, files, email, web)
into one chronological timeline. It then scores every event with machine learning so an
investigator can start with the most suspicious activity.

## Quick start

```bash
pip install -r requirements.txt
python pipeline.py            # optional: rebuilds everything in outputs/ (~10 s)
streamlit run app.py          # opens the dashboard at http://localhost:8501
python -m pytest              # runs the test suite
```

On Windows you can also double-click `run_dashboard.bat`.

## What changed from version 1

| Area | Before | Now |
|---|---|---|
| Dashboard | Crashed: the report CSV header began with a pasted PowerShell fragment, so `df["timestamp"]` failed | Header is repaired automatically; new dark "evidence file" interface with 5 tabs |
| Total events | Hard-coded `4615` | Calculated from the data |
| Code layout | 16 one-off scripts, copy-pasted logic, one file duplicated twice | Reusable `evidence/` package + one `pipeline.py` command |
| Tests | None | 73 pytest tests: unit, integration, output checks and dashboard smoke tests |
| Best F1 (stratified test) | 0.757 (baseline Random Forest) | **0.930** (enhanced RF), 0.934 hybrid score, AUC 0.989 |
| Future-period test (chronological) | F1 0.268 | F1 0.520, AUC 0.927 |
| Leakage | Not documented | Label-revealing columns excluded and verified by tests |

## Dashboard tabs

* **Overview**: weekly activity, risk distribution, an hour-by-weekday heatmap of flagged events, sources.
* **Timeline reconstruction**: pick a user to see their events in evidence-source lanes, a chronological
  event chain, sessions (split after N idle minutes) and silent periods longer than 3 days.
* **People of interest**: threat ranking of all users and a chart of volume against risk.
* **Model performance**: F1, accuracy, precision, recall and AUC for 5 models, ROC curves, confusion matrices, feature importance.
* **Evidence locker**: full-text search, CSV export, a downloadable investigation report, and upload of new raw
  evidence to be scored live.

The sidebar filters every tab by date range, source, risk level, user and minimum risk.

## How events are scored

1. **Normalise.** The raw export mixes a 9-field and a 7-field layout; both are mapped to one schema (`evidence/normalize.py`).
2. **Features.** Hour, weekday, off-hours (before 07:00 or after 19:00), weekend, seconds since the user's previous and next event,
   the user's event count that day, and whether the PC is not the user's usual one (`evidence/features.py`).
3. **Models** (`evidence/models.py`):
   * Random Forest (supervised, 300 trees, balanced classes)
   * Isolation Forest (unsupervised, 5% contamination)
   * **Hybrid risk score = 0.7 × RF probability + 0.3 × Isolation Forest risk.** RF probabilities are *out-of-fold*
     (5-fold cross-validation), so no event is scored by a model that saw it during training.
4. **Risk levels:** Critical ≥ 0.80, High ≥ 0.60, Medium ≥ 0.35, otherwise Low. Events with a score of 0.60 or more are flagged.

### Leakage control

In the raw file every label-1 row has 9 fields and every label-0 row has 7, and only label-1 rows have a `resource`.
A model given `record_format`, `resource`, `case_id` or `event_id` would score near 100% without learning anything.
These columns are therefore excluded (`config.LEAKAGE_COLUMNS`), and `tests/test_features.py` and
`tests/test_models.py` fail if one is ever used.

## Project structure

```
app.py                 Streamlit dashboard
pipeline.py            End-to-end command: normalise -> features -> models -> outputs
evidence/              Core library (config, normalize, features, models, timeline, report, charts)
data/                  Raw CERT export (input)
outputs/               Generated results used by the dashboard
  scored_evidence.csv            every event with features and risk scores
  anomaly_investigation_report.csv  flagged events, ordered by user then time
  user_risk_profiles.csv         per-user threat ranking
  model_metrics.json             all evaluation results
  normalized_evidence.csv        cleaned, unified dataset
tests/                 pytest suite
legacy_scripts/        The original version-1 scripts, kept for reference
```

## Tests

```
python -m pytest -v
```

| File | Covers |
|---|---|
| `test_normalize.py` | both raw layouts, source inference, case/label split, mixed date formats, full-dataset counts |
| `test_features.py` | time features, per-user gaps, preserved row order, no leaky features |
| `test_models.py` | splits, quality thresholds (F1 ≥ 0.88, AUC ≥ 0.97), enhanced beats baseline, risk bands |
| `test_timeline.py` | chronological reconstruction, filters, sessions, gaps, user profiles |
| `test_report.py` | corrupted-header repair, report contents and ordering |
| `test_pipeline.py` | full pipeline on a sample, output files, scoring of new evidence |
| `test_outputs.py` | the shipped outputs are consistent |
| `test_charts.py`, `test_app.py` | every chart builds; the dashboard runs with Streamlit's AppTest (needs streamlit + plotly installed) |
