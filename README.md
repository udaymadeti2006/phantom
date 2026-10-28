# TEAM PHANTOM
# PhantomStock R1 — Baseline Model

## What it does
Produces daily top-20 shelf-check alerts for PhantomStock R1. For target date D, features use only records through D-1. The baseline combines a transparent zero-sales/positive-stock rule with an Isolation Forest anomaly score. 

## Project layout
- `submission/R1_predictions.csv`: test-window predictions (2026-09-03 through 2026-09-30).
- `FULL_CODE/baseline_model.py`: code to reproduce dev and test predictions.
- `FULL_CODE/requirements.txt`: Python dependencies.

## Setup and run
Use Python 3.10+ in a virtual environment, then install dependencies:

```powershell
python -m pip install -r .\FULL_CODE\requirements.txt
python .\FULL_CODE\baseline_model.py
```


## Output format
`date, store_id, sku_id, rank, confidence, reason`. It targets all 25 stores, the 28 test dates, and up to 20 candidates per store-day.

## AI usage and pre-built code
This baseline was prepared with AI coding assistance and uses pandas, NumPy, and scikit-learn.

