#!/usr/bin/env python3
"""PhantomPulse AI: demand-aware phantom stock-out ranking.

Important design rules:
- A prediction for date D uses sales and inventory only through D-1.
- The promotion and holiday calendar for D may be used because it is known in advance.
- The official competition list is ranked by phantom-stock-out risk, not revenue.
- Exactly up to 20 SKUs are emitted per store-day; submit mode expects 25 stores.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DEV_START, DEV_END = "2026-06-04", "2026-09-02"
TEST_START, TEST_END = "2026-09-03", "2026-09-30"
K = 20
POSITIVE_TYPES = {"phantom", "misplaced"}
ALL_LABEL_TYPES = {"phantom", "misplaced", "delisted"}

NUMERIC_COLS = [
    "zero_run", "avg7", "avg14", "avg28", "avg56", "pre_run_rate",
    "same_weekday_rate", "zero_rate28", "sales_std28",
    "sales_nonzero_days28", "stock", "stock_prev", "stock_change",
    "stock_persistence7", "stock_per_demand", "gap_run",
    "prior_day_sales", "days_since_sale", "on_promo_target",
    "prev_store_sales", "prev_store_zero_share", "unit_price_inr",
    "unit_margin_inr", "case_pack", "city_tier", "size_sqft",
    "weekday", "month", "day_of_month", "is_weekend", "is_holiday",
]
CAT_COLS = ["category", "sub_category", "shelf_life_class", "format", "region"]


def read_panel(input_file: Path) -> pd.DataFrame:
    cols = ["date", "store_id", "sku_id", "units_sold", "on_hand_eod", "on_promo"]
    if input_file.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_file, columns=cols)
    elif input_file.suffix.lower() == ".csv":
        df = pd.read_csv(input_file, usecols=cols, parse_dates=["date"])
    else:
        raise ValueError("Input must be the organizer Parquet file or a CSV sample.")
    df["date"] = pd.to_datetime(df["date"])
    # Categoricals reduce memory on the 8-million-row panel.
    for col in ("store_id", "sku_id"):
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0).astype("float32")
    df["on_hand_eod"] = pd.to_numeric(df["on_hand_eod"], errors="coerce").astype("float32")
    df["on_promo"] = pd.to_numeric(df["on_promo"], errors="coerce").fillna(0).astype("int8")
    return df


def _rolling(series: pd.Series, group: pd.Series, window: int, min_periods: int = 1,
             operation: str = "mean") -> pd.Series:
    grouped = series.groupby(group, sort=False, observed=True)
    roll = grouped.rolling(window, min_periods=min_periods)
    if operation == "mean":
        result = roll.mean()
    elif operation == "std":
        result = roll.std()
    elif operation == "sum":
        result = roll.sum()
    else:
        raise ValueError(operation)
    return result.reset_index(level=0, drop=True)


def _holiday_dates_for_region(holiday_df: pd.DataFrame, region: str) -> set:
    applicable = {
        "national": {"North", "South", "East", "West", "Central"},
        "south": {"South"},
        "north_west": {"North", "West"},
        "west_south": {"West", "South"},
    }
    result = set()
    for row in holiday_df.itertuples(index=False):
        scope = str(row.scope).lower()
        if region in applicable.get(scope, set()):
            result.add(pd.Timestamp(row.date))
    return result


def build_features(panel: pd.DataFrame, sku_master: pd.DataFrame,
                   store_master: pd.DataFrame, holiday_calendar: pd.DataFrame,
                   start_date: str = DEV_START, end_date: str = TEST_END) -> pd.DataFrame:
    """Create one row for prediction date D using panel data through D-1."""
    sku_keep = [
        "sku_id", "category", "sub_category", "unit_price_inr",
        "unit_margin_inr", "shelf_life_class", "case_pack",
    ]
    store_keep = [
        "store_id", "format", "region", "city_tier", "size_sqft", "label_split",
    ]
    sku_meta = sku_master[sku_keep].drop_duplicates("sku_id")
    store_meta = store_master[store_keep].drop_duplicates("store_id")
    holiday_calendar = holiday_calendar.copy()
    holiday_calendar["date"] = pd.to_datetime(holiday_calendar["date"])

    wanted_start = pd.Timestamp(start_date)
    wanted_end = pd.Timestamp(end_date)
    chunks = []

    # Process one store at a time to avoid building all 8 million rows of features at once.
    meta_by_store = store_meta.set_index("store_id").to_dict("index")

    for store_key, store_panel in panel.groupby("store_id", sort=False, observed=True):
        store_id = str(store_key)
        d = store_panel[
            ["date", "store_id", "sku_id", "units_sold", "on_hand_eod", "on_promo"]
        ].copy()
        d = d.sort_values(["sku_id", "date"]).reset_index(drop=True)
        sku_group = d["sku_id"]
        g = d.groupby("sku_id", sort=False, observed=True)

        sales = d["units_sold"].astype("float32")
        zero = sales.eq(0)
        # Each non-zero-sales day starts a new segment; zeros accumulate within a segment.
        segment = sales.ne(0).groupby(sku_group, sort=False, observed=True).cumsum()
        zero_run = zero.groupby([sku_group, segment], sort=False, observed=True).cumsum().astype("int16")
        d["zero_run"] = zero_run.astype("float32")

        d["avg7"] = _rolling(sales, sku_group, 7, 1, "mean").astype("float32")
        d["avg14"] = _rolling(sales, sku_group, 14, 1, "mean").astype("float32")
        d["avg28"] = _rolling(sales, sku_group, 28, 1, "mean").astype("float32")
        d["avg56"] = _rolling(sales, sku_group, 56, 1, "mean").astype("float32")
        roll28_min7 = _rolling(sales, sku_group, 28, 7, "mean")
        prev28_rate = roll28_min7.groupby(sku_group, sort=False, observed=True).shift(1)
        rate_at_run_start = prev28_rate.where(zero_run.eq(1))
        d["pre_run_rate"] = rate_at_run_start.groupby(
            [sku_group, segment], sort=False, observed=True
        ).ffill().astype("float32")

        # For prediction date D, row D-1 has same-weekday historical lags of 6, 13, 20, 27 rows.
        same_sum = np.zeros(len(d), dtype="float32")
        same_count = np.zeros(len(d), dtype="float32")
        run_array = zero_run.to_numpy()
        for lag in (6, 13, 20, 27):
            shifted = g["units_sold"].shift(lag).to_numpy(dtype="float32")
            valid = (run_array <= lag) & np.isfinite(shifted)
            same_sum[valid] += shifted[valid]
            same_count[valid] += 1
        d["same_weekday_rate"] = np.divide(
            same_sum, same_count, out=np.full(len(d), np.nan, dtype="float32"),
            where=same_count > 0
        )

        d["zero_rate28"] = _rolling(zero.astype("float32"), sku_group, 28, 1, "mean").astype("float32")
        d["sales_std28"] = _rolling(sales, sku_group, 28, 2, "std").astype("float32")
        d["sales_nonzero_days28"] = _rolling(sales.gt(0).astype("float32"), sku_group, 28, 1, "sum").astype("float32")

        stock = d.groupby("sku_id", sort=False, observed=True)["on_hand_eod"].ffill()
        d["stock"] = stock.astype("float32")
        d["stock_prev"] = stock.groupby(sku_group, sort=False, observed=True).shift(1).astype("float32")
        stock_change = stock.groupby(sku_group, sort=False, observed=True).diff()
        d["stock_change"] = stock_change.astype("float32")
        same_stock_transition = stock_change.eq(0).astype("float32").where(stock_change.notna())
        d["stock_persistence7"] = _rolling(
            same_stock_transition, sku_group, 7, 2, "mean"
        ).astype("float32")

        base_rate = d["pre_run_rate"].fillna(d["avg28"]).clip(lower=0.05)
        d["stock_per_demand"] = (d["stock"] / base_rate).clip(lower=0, upper=365).astype("float32")
        d["gap_run"] = (base_rate * d["zero_run"]).clip(lower=0, upper=10000).astype("float32")
        d["prior_day_sales"] = sales
        d["days_since_sale"] = d["zero_run"]

        # Known calendar information for D is safe; sales/stock from D are not used.
        d["on_promo_target"] = g["on_promo"].shift(-1).astype("float32")
        daily_sales = d.groupby("date", observed=True)["units_sold"].sum()
        daily_zero_share = d.assign(_zero=zero.astype("float32")).groupby("date", observed=True)["_zero"].mean()
        d["prev_store_sales"] = d["date"].map(daily_sales).astype("float32")
        d["prev_store_zero_share"] = d["date"].map(daily_zero_share).astype("float32")

        d["pred_date"] = d["date"] + pd.Timedelta(days=1)
        d["weekday"] = d["pred_date"].dt.dayofweek.astype("int8")
        d["month"] = d["pred_date"].dt.month.astype("int8")
        d["day_of_month"] = d["pred_date"].dt.day.astype("int8")
        d["is_weekend"] = d["weekday"].isin([5, 6]).astype("int8")
        region = str(meta_by_store.get(store_id, {}).get("region", ""))
        holiday_dates = _holiday_dates_for_region(holiday_calendar, region)
        d["is_holiday"] = d["pred_date"].isin(holiday_dates).astype("int8")

        mask = d["pred_date"].between(wanted_start, wanted_end)
        keep_features = [
            "pred_date", "store_id", "sku_id", "zero_run", "avg7", "avg14", "avg28", "avg56",
            "pre_run_rate", "same_weekday_rate", "zero_rate28", "sales_std28",
            "sales_nonzero_days28", "stock", "stock_prev", "stock_change",
            "stock_persistence7", "stock_per_demand", "gap_run", "prior_day_sales",
            "days_since_sale", "on_promo_target", "prev_store_sales",
            "prev_store_zero_share", "weekday", "month", "day_of_month",
            "is_weekend", "is_holiday",
        ]
        chunks.append(d.loc[mask, keep_features].copy())

    features = pd.concat(chunks, ignore_index=True)
    features = features.rename(columns={"pred_date": "date"})
    features = features.merge(sku_meta, on="sku_id", how="left")
    features = features.merge(store_meta, on="store_id", how="left")
    for col in CAT_COLS:
        features[col] = features[col].fillna("Unknown").astype(str)
    # Numeric features use float32 to keep the training matrix compact.
    for col in NUMERIC_COLS:
        if col in features:
            features[col] = pd.to_numeric(features[col], errors="coerce").astype("float32")
    return features


def attach_labels(features: pd.DataFrame, truth_path: Path) -> pd.DataFrame:
    truth = pd.read_csv(truth_path, parse_dates=["date"], dtype={"store_id": str, "sku_id": str})
    truth = truth[["date", "store_id", "sku_id", "label_type", "episode_id"]].drop_duplicates(
        ["date", "store_id", "sku_id"]
    )
    out = features.merge(truth, on=["date", "store_id", "sku_id"], how="left")
    out["label_type"] = out["label_type"].fillna("normal")
    out["y"] = out["label_type"].isin(POSITIVE_TYPES).astype("int8")
    return out


def build_model() -> Pipeline:
    # Dense one-hot output is used because sklearn's histogram gradient booster requires dense input.
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_COLS),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
            ]), CAT_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    classifier = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.08, max_leaf_nodes=15,
        l2_regularization=2.0, random_state=42
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def fit_model(train: pd.DataFrame) -> Pipeline:
    model = build_model()
    X = train[NUMERIC_COLS + CAT_COLS].copy()
    for c in NUMERIC_COLS:
        X[c] = X[c].astype("float32")
    model.fit(X, train["y"].astype(int))
    return model


def predict_risk(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    X = frame[NUMERIC_COLS + CAT_COLS].copy()
    for c in NUMERIC_COLS:
        X[c] = X[c].astype("float32")
    return model.predict_proba(X)[:, 1]


def base_demand(row) -> float:
    for name in ("pre_run_rate", "same_weekday_rate", "avg28", "avg14", "avg7"):
        value = getattr(row, name, np.nan)
        if pd.notna(value) and np.isfinite(value):
            return max(0.0, float(value))
    return 0.0


def make_reason(row) -> str:
    stock_text = "unknown" if pd.isna(row.stock) else f"{int(max(0, row.stock))} units at the previous close"
    rate = base_demand(row)
    run = int(max(0, row.zero_run)) if pd.notna(row.zero_run) else 0
    if run >= 2 and rate >= 0.5:
        reason = f"No sales for {run} days versus historical demand of about {rate:.1f}/day; system showed {stock_text}."
    elif int(row.on_promo_target) == 1 and rate >= 0.5:
        reason = f"Promotion is scheduled today, but recent sales were unusually weak versus about {rate:.1f}/day demand; prior stock was {stock_text}."
    elif pd.notna(row.stock_persistence7) and row.stock_persistence7 >= 0.8 and rate >= 0.5:
        reason = f"Recorded stock stayed nearly unchanged while historical demand is about {rate:.1f}/day; prior stock was {stock_text}."
    else:
        reason = f"Ranked highly by the demand-and-inventory risk model; recent zero-sales streak {run} days; prior stock was {stock_text}."
    return reason


def rank_top20(frame: pd.DataFrame, risk_col: str = "risk") -> pd.DataFrame:
    out = []
    for (date, store_id), group in frame.groupby(["date", "store_id"], sort=True):
        top = group.sort_values([risk_col, "sku_id"], ascending=[False, True], kind="stable").head(K).copy()
        top["rank"] = np.arange(1, len(top) + 1)
        top["confidence"] = top[risk_col].clip(0, 1).astype(float)
        top["reason"] = [make_reason(row) for row in top.itertuples(index=False)]
        out.append(top)
    if not out:
        return pd.DataFrame(columns=["date", "store_id", "sku_id", "rank", "confidence", "reason"])
    result = pd.concat(out, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    return result


def local_metrics(predictions: pd.DataFrame, truth: pd.DataFrame) -> dict:
    # Ranked feature frames still contain the training label columns; remove them before joining truth.
    p = predictions.drop(columns=["label_type", "episode_id", "y"], errors="ignore").copy()
    t = truth.copy()
    p["date"] = pd.to_datetime(p["date"])
    t["date"] = pd.to_datetime(t["date"])
    dates = sorted(t["date"].unique())
    stores = sorted(t["store_id"].astype(str).unique())
    scope = pd.MultiIndex.from_product([dates, stores], names=["date", "store_id"])
    positives = t[t["label_type"].isin(POSITIVE_TYPES)].copy()
    joined = p.merge(t[["date", "store_id", "sku_id", "label_type", "episode_id"]],
                     on=["date", "store_id", "sku_id"], how="left")
    joined["hit"] = joined["label_type"].isin(POSITIVE_TYPES)
    hits = joined.groupby(["date", "store_id"])["hit"].sum().reindex(scope, fill_value=0)
    total_hits = int(joined["hit"].sum())
    pos_episode_ids = set(positives["episode_id"].dropna().astype(str))
    caught = set(joined.loc[joined["hit"], "episode_id"].dropna().astype(str))
    return {
        "primary_score_precision_at_20": float((hits / K).mean()) if len(scope) else None,
        "row_recall": float(total_hits / len(positives)) if len(positives) else None,
        "episode_recall": float(len(caught & pos_episode_ids) / len(pos_episode_ids)) if pos_episode_ids else None,
        "hits": total_hits,
        "positive_rows": int(len(positives)),
        "positive_episodes": int(len(pos_episode_ids)),
        "scored_store_days": int(len(scope)),
        "mean_rows_per_store_day": float(len(p) / len(scope)) if len(scope) else None,
    }


def add_financial_estimates(details: pd.DataFrame, recovery_rate: float) -> pd.DataFrame:
    out = details.copy()
    base = out.apply(base_demand, axis=1).astype(float)
    run = pd.to_numeric(out["zero_run"], errors="coerce").fillna(0).clip(lower=0)
    out["estimated_missed_units_to_date"] = (base * run).clip(lower=0)
    out["estimated_lost_revenue_to_date_inr"] = (
        out["estimated_missed_units_to_date"] * pd.to_numeric(out["unit_price_inr"], errors="coerce").fillna(0)
    )
    out["assumed_recovery_rate"] = recovery_rate
    out["estimated_recoverable_revenue_to_date_inr"] = (
        out["estimated_lost_revenue_to_date_inr"] * recovery_rate
    )
    return out


def write_validation(predictions: pd.DataFrame, truth: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep the validation submission in the official format; feature/label columns are for analysis only.
    submission_cols = ["date", "store_id", "sku_id", "rank", "confidence", "reason"]
    predictions[submission_cols].to_csv(out_dir / "validation_predictions.csv", index=False)
    truth.to_csv(out_dir / "validation_truth.csv", index=False)
    metrics = local_metrics(predictions, truth)
    (out_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_dir / 'validation_predictions.csv'} and validation_truth.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["submit", "validate", "temporal-validate"], default="submit")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--input-file", default=None,
                    help="Optional CSV sample or Parquet panel. Default: data/daily_sales_inventory.parquet")
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--submission-dir", default=str(ROOT / "submission"))
    ap.add_argument("--recovery-rate", type=float, default=0.60,
                    help="Illustrative recoverable share used only in the financial detail report.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    input_file = Path(args.input_file) if args.input_file else data_dir / "daily_sales_inventory.parquet"
    out_dir = Path(args.out_dir)
    submission_dir = Path(args.submission_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading panel: {input_file}")
    panel = read_panel(input_file)
    stores = pd.read_csv(data_dir / "store_master.csv", dtype={"store_id": str})
    skus = pd.read_csv(data_dir / "sku_master.csv", dtype={"sku_id": str})
    holidays = pd.read_csv(data_dir / "holiday_calendar.csv", parse_dates=["date"])
    truth_path = data_dir / "dev_truth.csv"
    truth = pd.read_csv(truth_path, parse_dates=["date"], dtype={"store_id": str, "sku_id": str})

    print(f"Panel rows: {len(panel):,}; stores present: {panel['store_id'].nunique()}; "
          f"SKU-store pairs: {panel.groupby(['store_id','sku_id'], observed=True).ngroups:,}")
    features = build_features(panel, skus, stores, holidays, DEV_START, TEST_END)
    del panel
    labelled = attach_labels(features, truth_path)
    dev_start, dev_end = pd.Timestamp(DEV_START), pd.Timestamp(DEV_END)
    test_start, test_end = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)
    dev_stores = set(stores.loc[stores["label_split"] == "dev_labelled", "store_id"].astype(str))
    dev_rows = labelled[
        labelled["date"].between(dev_start, dev_end) &
        labelled["store_id"].astype(str).isin(dev_stores)
    ].copy()

    if args.mode == "temporal-validate":
        train_mask = dev_rows["date"].between(pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-31"))
        val_mask = dev_rows["date"].between(pd.Timestamp("2026-08-01"), dev_end)
        train = dev_rows.loc[train_mask].copy()
        val = dev_rows.loc[val_mask].copy()
        if train.empty or val.empty:
            raise RuntimeError("Temporal validation has no train or validation rows. Check the input data dates.")
        model = fit_model(train)
        val["risk"] = predict_risk(model, val)
        predictions = rank_top20(val)
        truth_val = truth[
            truth["date"].between(pd.Timestamp("2026-08-01"), dev_end) &
            truth["store_id"].astype(str).isin(set(val["store_id"].astype(str)))
        ].copy()
        write_validation(predictions, truth_val, out_dir)
        return

    if args.mode == "validate":
        # Hold out one labelled store from each format to emulate unseen-store evaluation.
        dev_meta = stores[stores["label_split"] == "dev_labelled"].sort_values("store_id")
        heldout = []
        for fmt in sorted(dev_meta["format"].unique()):
            candidates = dev_meta[dev_meta["format"] == fmt]["store_id"].astype(str).tolist()
            present = set(labelled["store_id"].astype(str).unique())
            candidates = [s for s in candidates if s in present]
            if candidates:
                heldout.append(candidates[0])
        if not heldout:
            raise RuntimeError("No dev stores available for grouped validation.")
        train = dev_rows[~dev_rows["store_id"].astype(str).isin(heldout)].copy()
        val = dev_rows[dev_rows["store_id"].astype(str).isin(heldout)].copy()
        if train.empty or val.empty:
            raise RuntimeError(
                "Grouped validation needs the full panel with multiple development stores. "
                "Use --mode temporal-validate for the 2-store CSV sample."
            )
        model = fit_model(train)
        val["risk"] = predict_risk(model, val)
        predictions = rank_top20(val)
        truth_val = truth[truth["store_id"].astype(str).isin(heldout)].copy()
        write_validation(predictions, truth_val, out_dir)
        print(f"Grouped holdout stores: {heldout}")
        return

    # Final submission: train on all public-label stores and score every store in the test window.
    if panel_store_count(input_file) < 25:
        raise RuntimeError(
            "Submit mode requires the full 25-store Parquet panel. You supplied a sample/partial file. "
            "Use --mode temporal-validate to experiment with the 2-store CSV sample."
        )
    model = fit_model(dev_rows)
    test_rows = labelled[labelled["date"].between(test_start, test_end)].copy()
    test_rows["risk"] = predict_risk(model, test_rows)
    ranked = rank_top20(test_rows)
    # Official submission has only required columns. Financial details go to outputs/.
    sub_cols = ["date", "store_id", "sku_id", "rank", "confidence", "reason"]
    sub = ranked[sub_cols].copy()
    # Sanity checks required by the competition.
    sub["date"] = sub["date"].astype(str)
    expected_dates = set(pd.date_range(TEST_START, TEST_END).strftime("%Y-%m-%d"))
    if set(sub["date"].unique()) != expected_dates:
        raise RuntimeError("Submission does not contain every test date.")
    if sub["store_id"].nunique() != 25:
        raise RuntimeError(f"Expected 25 stores in submission; found {sub['store_id'].nunique()}.")
    counts = sub.groupby(["date", "store_id"]).size()
    if not (counts == K).all():
        raise RuntimeError("Every store-day must contain exactly 20 rows.")
    if sub.duplicated(["date", "store_id", "sku_id"]).any():
        raise RuntimeError("Duplicate SKU in a store-day.")
    out_path = submission_dir / "R1_predictions.csv"
    sub.to_csv(out_path, index=False)
    placeholder = submission_dir / ".gitkeep"
    if placeholder.exists():
        placeholder.unlink()

    detail_cols = [
        "date", "store_id", "sku_id", "rank", "confidence", "reason", "risk",
        "zero_run", "pre_run_rate", "same_weekday_rate", "avg28", "stock",
        "stock_persistence7", "on_promo_target", "unit_price_inr", "unit_margin_inr",
        "estimated_missed_units_to_date", "estimated_lost_revenue_to_date_inr",
        "assumed_recovery_rate", "estimated_recoverable_revenue_to_date_inr",
    ]
    details = ranked.copy()
    details["date"] = pd.to_datetime(details["date"])
    details = add_financial_estimates(details, args.recovery_rate)
    detail_cols = [c for c in detail_cols if c in details.columns]
    details[detail_cols].to_csv(out_dir / "alert_details.csv", index=False)
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "phantompulse_model.joblib")
    print(f"Wrote official submission: {out_path} ({len(sub):,} rows)")
    print(f"Wrote manager details: {out_dir / 'alert_details.csv'}")
    print("Reminder: confidence is a model ranking score unless separately calibrated.")
    print("Financial figures are estimates conditional on an incident; they are not verified recovered revenue.")


def panel_store_count(input_file: Path) -> int:
    if input_file.suffix.lower() == ".parquet":
        return int(pd.read_parquet(input_file, columns=["store_id"])["store_id"].nunique())
    return int(pd.read_csv(input_file, usecols=["store_id"])["store_id"].nunique())


if __name__ == "__main__":
    main()
