# PhantomPulse AI — R1 Phantom Stock-Out Detector

## What this solution does

PhantomPulse combines three ideas into one workflow:

1. **Inventory Reality Gap Engine:** compares a SKU-store's current sales behavior with its own recent and same-weekday demand history, while considering recorded stock persistence and promotions.
2. **Lost Revenue Detective:** estimates the sales opportunity associated with a suspicious zero-sales run. This is an estimate conditional on a genuine issue, not confirmed lost or recovered revenue.
3. **Self-Learning Investigation Workflow:** records verified physical inspection outcomes so future model versions can be trained and evaluated. The current code provides the prediction pipeline and manager dashboard; inspection outcomes should be collected in a separate feedback table and used only after review.

The competition ranking is by predicted phantom-stock-out risk to target the official Precision@20 metric. Financial-value ranking is a separate business view and is not used to replace the official ranking. The dashboard includes an inspection-feedback form that appends manager outcomes to `outputs/inspection_feedback.csv`.

## Important findings from the supplied pack

The pack contains:
- 7,970,112 rows in `daily_sales_inventory.parquet`, covering 25 stores and 548 days.
- A 110,400-row CSV sample for two development stores.
- Public development truth for 13 stores, with `phantom` and `misplaced` counted as positive.
- `delisted` labels as decoys: these must not be treated as hits.
- An official evaluator that averages hits divided by 20 over each scored store-day. Lists with fewer than 20 rows are penalized, so the code ranks all eligible SKU-store candidates and emits 20 rows per store-day.
- A strict no-look-ahead rule: the list for date D uses sales and inventory through D-1. The promotion/holiday calendar for D is allowed because it is known in advance.

### Preliminary experiment on the included 2-store sample

I tested a temporal holdout on the two-store CSV sample: train on July 2026 and evaluate on 1 August–2 September 2026 (66 store-days). The included demand-aware HistGradientBoosting script reached:
- Precision@20: **0.4705**
- Row recall: **0.4401**
- Episode recall: **0.3385**

The simple zero-sales-run baseline with a 10-day threshold reached:
- Precision@20: **0.3591**
- Row recall: **0.3359**
- Episode recall: **0.2462**

This is encouraging but **not a final competition result**: it covers only two stores and a temporal holdout, not the 12 hidden test stores. The full Parquet file could not be executed in this environment because the Parquet engine (`pyarrow`) was unavailable. The code below is designed to run on the full organizer data after installing requirements. Always use the official evaluator for final validation.

## Files expected at project root

Copy the organizer's `data/` folder and `evaluate.py` into this project root. Do not commit the data files into the team submission zip.

Expected inputs:
- `data/daily_sales_inventory.parquet`
- `data/dev_truth.csv`
- `data/store_master.csv`
- `data/sku_master.csv`
- `data/holiday_calendar.csv`
- `evaluate.py`

The project zip intentionally excludes the organizer's data and any trained model.

## Setup

From the project root:

```bash
pip install -r FULL_CODE/requirements.txt
```

The full Parquet file needs `pyarrow`. If you only want to test the pipeline first, use the supplied 2-store CSV sample.

## Recommended run order

### 1. Quick temporal validation on the 2-store CSV sample

```bash
python FULL_CODE/phantompulse.py --mode temporal-validate --input-file data/daily_sales_inventory_sample_2stores.csv
```

Then score the output with the official evaluator:

```bash
python evaluate.py --submission outputs/validation_predictions.csv --truth outputs/validation_truth.csv --out outputs/validation_score.json
```

### 2. Validate generalization to unseen development stores

This holds out one development store from each available store format, trains on the other development stores, and evaluates on the held-out stores:

```bash
python FULL_CODE/phantompulse.py --mode validate
python evaluate.py --submission outputs/validation_predictions.csv --truth outputs/validation_truth.csv --out outputs/validation_score.json
```

Review Precision@20, row recall and episode recall before final submission.

### 3. Generate the official test-window submission

```bash
python FULL_CODE/phantompulse.py --mode submit
```

This writes:
- `submission/R1_predictions.csv` — required competition file.
- `outputs/alert_details.csv` — ranked alerts with feature evidence and financial estimates.
- `models/phantompulse_model.joblib` — trained model for reproducibility.

Submit mode requires the full 25-store Parquet panel and checks that there are exactly 20 predictions for every store on every date from 2026-09-03 through 2026-09-30.

### 4. Launch the dashboard

```bash
streamlit run FULL_CODE/dashboard.py
```

The dashboard displays the daily queue, evidence for a selected SKU, sales versus expected demand, recorded inventory history, and estimated sales opportunity. The chart stops before the alert date so it does not display same-day information that was unavailable when the morning list was created.

## Model details

### Feature groups
- Zero-sales streak through D-1.
- Rolling 7/14/28/56-day sales summaries.
- Historical demand before the current zero-sales run.
- Same-weekday historical demand from prior weeks, excluding values inside the current run.
- Zero-sales frequency and sales variability.
- Previous-close recorded inventory, inventory change and short-term persistence.
- Inventory-to-demand coverage and estimated demand gap.
- Promotion status scheduled for D.
- Previous-day store-wide sales and zero-sales share (helps detect store closure or feed issues).
- Weekday, month, holiday flag.
- SKU attributes (category, subcategory, price, margin, shelf-life class, case pack).
- Store attributes (format, region, city tier, selling area).

### Why this differs from a fixed threshold
A 5-day zero-sales streak means different things for a fast-moving SKU and a slow-moving SKU. The model learns from the public labels which combinations of demand, inventory, promotion, and product/store attributes tend to indicate `phantom` or `misplaced` cases. `delisted` is a negative class for the primary binary target.

### No-look-ahead guarantee
Every feature row for date D is built from the panel row for D-1 and earlier history. Only calendar information already known for D (promotion and holiday) is added. Same-day sales and same-day inventory for D are never used to rank D.

### Why 20 rows are always produced
The evaluator divides hits by 20 for every scored store-day, including lists with fewer than 20 rows. The model therefore ranks candidates and selects 20 per store-day instead of requiring a hard threshold that could leave empty slots.

### Confidence field
The `confidence` field contains the model's ranking score in the range 0–1. It should not be described as a calibrated real-world probability unless a separate calibration check is performed.

## Financial estimates

For each alert, the detail report estimates:

`estimated_missed_units_to_date = baseline_daily_demand × zero_sales_streak`

`estimated_lost_revenue_to_date = estimated_missed_units_to_date × unit_price_inr`

`estimated_recoverable_revenue_to_date = estimated_lost_revenue_to_date × assumed_recovery_rate`

The default recovery-rate assumption is 60% and can be changed with `--recovery-rate`. It is an illustrative assumption, not a measured fact. Estimates are conditional on a genuine issue and can overstate recoverable demand. Do not sum repeated daily alerts as if each were a new incident; ongoing incidents overlap. A credible annual estimate should deduplicate incidents and use a pilot-measured recovery rate and contribution margin.

## Self-learning feedback (next deployment step)

After a manager physically checks a flagged SKU, the dashboard form appends date, store, SKU, predicted rank/score, outcome, resolution date, recovered units and notes to `outputs/inspection_feedback.csv`.

Do not treat uninspected SKUs as confirmed negatives. The inspected set is selected by the model, so future training must account for selection bias. The current static competition data ends on 2026-09-30; feedback is a deployment workflow for future data, not a way to obtain hidden test labels. Retrain only after enough reviewed outcomes exist, then deploy a new model only if it improves held-out Precision@20 and recall.

## Submission checklist

- [ ] Copy `data/` and `evaluate.py` locally for running, but exclude data and model files from the final team zip.
- [ ] Run grouped-store validation and inspect the official score.
- [ ] Run `python FULL_CODE/phantompulse.py --mode submit`.
- [ ] Confirm `submission/R1_predictions.csv` has 14,000 rows and 20 rows per store-day.
- [ ] Run the official evaluator on an appropriate development validation output.
- [ ] Include the full `FULL_CODE/` directory and your README in the team zip.
- [ ] Do not claim that sales data proves theft; alerts require physical verification.
