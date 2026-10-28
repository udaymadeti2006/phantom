#!/usr/bin/env python3
"""Official scorer for R1 -- The Phantom Stock-Out Detector (TalentAP Mega Hackathon 2026).

Usage
-----
    python evaluate.py --submission submission/R1_predictions.csv --truth data/dev_truth.csv [--out score.json]

The SAME file is used by the organisers with the hidden test truth.
Requires only Python >= 3.9, numpy and pandas.

Submission format (CSV, one row per flagged SKU)
------------------------------------------------
    date, store_id, sku_id, rank, confidence, reason [, predicted_type]
* date            YYYY-MM-DD -- the business day the list is for
* store_id/sku_id ids exactly as in the data files
* rank            integer 1..20, unique within a (date, store_id) list
* confidence      number between 0 and 1
* reason          one-line plain-English reason (text; may not be blank in spirit -- judges read it)
* predicted_type  OPTIONAL stretch column: phantom | misplaced | delisted (blank allowed)
At most MAX_PER_STORE_DAY rows per (date, store_id).  No duplicate (date, store_id, sku_id).

Scoring
-------
Only (date, store_id) pairs present in the truth file are scored ("store-days" = every date in the truth
file x every store in the truth file).  Rows for other dates/stores are ignored, so you may submit lists
for every store and every date.

* primary_score = mean over scored store-days of  hits / K   (K = 20; a list with fewer than 20 rows is
  still divided by 20; a store-day with no rows scores 0).  Range 0..1, higher is better.
  A "hit" is a submitted row whose (date, store_id, sku_id) has label_type in POSITIVE_TYPES.
* row_recall      = hits / all positive rows in the truth
* episode_recall  = positive episodes flagged on at least one of their days / all positive episodes
* ceiling         = best possible primary_score for this truth (some store-days have < 20 positives)
* type_accuracy   = (stretch, only if a predicted_type column is given) among your flagged rows that are
                    phantom / misplaced / delisted in truth AND have a non-blank predicted_type, the share where
                    predicted_type equals the true type (blank predicted_type rows are left out)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- constants
K = 20                                   # manager's daily verification budget per store
MAX_PER_STORE_DAY = 20                   # more rows than this for one store-day is a format error
POSITIVE_TYPES = ("phantom", "misplaced")  # counted as hits
DECOY_TYPES = ("delisted",)              # labelled in truth, but NOT a hit
VALID_TYPES = POSITIVE_TYPES + DECOY_TYPES
REQUIRED_COLS = ["date", "store_id", "sku_id", "rank", "confidence", "reason"]
OPTIONAL_COLS = ["predicted_type"]
TRUTH_COLS = ["date", "store_id", "sku_id", "label_type", "episode_id"]


class FormatError(Exception):
    pass


def _raise_csv_field_limit() -> None:
    lim = sys.maxsize
    while True:                       # sys.maxsize overflows a C long on some platforms (Windows)
        try:
            csv.field_size_limit(lim)
            return
        except OverflowError:
            lim //= 10


_raise_csv_field_limit()


def fail(msg: str) -> None:
    print("\n*** SUBMISSION FORMAT ERROR ***")
    print(msg)
    print("\nNothing was scored. Fix the problem above and run evaluate.py again.")
    sys.exit(2)


def _strict_dates(s: pd.Series, what: str) -> pd.Series:
    s = s.astype(str).str.strip()
    ok = s.str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    parsed = pd.to_datetime(s.where(ok), format="%Y-%m-%d", errors="coerce")
    bad = parsed.isna()
    if bad.any():
        ex = s[bad].head(5).tolist()
        raise FormatError(f"{what}: {int(bad.sum())} value(s) in column 'date' are not valid YYYY-MM-DD dates, "
                          f"e.g. {ex}. Write dates like 2026-09-03.")
    return parsed.dt.strftime("%Y-%m-%d")


def load_truth(path: str) -> pd.DataFrame:
    try:
        t = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:  # noqa: BLE001
        raise FormatError(f"Could not read truth file '{path}': {type(e).__name__}: {str(e)[:200]}")
    missing = [c for c in TRUTH_COLS if c not in t.columns]
    if missing:
        raise FormatError(f"Truth file '{path}' is missing columns {missing}. Did you pass the submission as --truth?")
    t["date"] = _strict_dates(t["date"], "truth file")
    bad = ~t["label_type"].isin(VALID_TYPES)
    if bad.any():
        raise FormatError(f"Truth file has unknown label_type values {sorted(t.label_type[bad].unique())[:5]}")
    return t


def _read_text(path: str, warnings: list[str]) -> str:
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        raise FormatError(f"Submission file '{path}' was not found. Check the path "
                          f"(expected something like submission/R1_predictions.csv).")
    except IsADirectoryError:
        raise FormatError(f"'{path}' is a folder, not a file. Pass the CSV itself, e.g. submission/R1_predictions.csv.")
    except OSError as e:
        raise FormatError(f"Could not open '{path}': {e}")
    if b"\x00" in raw[:65536]:
        raise FormatError(f"'{path}' looks like a binary file (zip/xlsx/parquet?), not a CSV. Save it as CSV (UTF-8).")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        warnings.append("File is not UTF-8; read it as Windows-1252 (Excel default). Prefer 'CSV UTF-8' when saving.")
        text = raw.decode("cp1252", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")      # old-Mac / mixed line endings


def load_submission(path: str) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    text = _read_text(path, warnings)
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except csv.Error as e:
        raise FormatError(f"Could not parse the file as CSV ({e}). Save it as a plain comma-separated UTF-8 file.")
    rows = [r for r in rows if any(x.strip() for x in r)]
    if not rows:
        raise FormatError("The submission file is empty.")
    n_head = len(rows[0])
    bad = [i for i, r in enumerate(rows[1:], start=2) if len(r) != n_head]
    if bad:
        raise FormatError(f"{len(bad)} line(s) have a different number of fields than the header ({n_head}), first at "
                          f"line {bad[0]} ({len(rows[bad[0] - 1])} fields). Usually a 'reason' text contains a comma: wrap it "
                          f"in double quotes (pandas' to_csv does this for you).")
    try:
        s = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except Exception as e:  # noqa: BLE001
        raise FormatError(f"Could not read '{path}' as CSV: {e}. Save it as a plain comma-separated UTF-8 file.")
    s.columns = [c.strip().lower() for c in s.columns]
    missing = [c for c in REQUIRED_COLS if c not in s.columns]
    if missing:
        raise FormatError(f"Missing required column(s): {missing}.\nYour columns: {list(s.columns)}\n"
                          f"Required header: {','.join(REQUIRED_COLS)}  (optional extra column: predicted_type)")
    extra = [c for c in s.columns if c not in REQUIRED_COLS + OPTIONAL_COLS]
    if extra:
        warnings.append(f"Ignoring unknown column(s): {extra}")
    if len(s) == 0:
        raise FormatError("The submission has a header but no rows.")
    s["date"] = _strict_dates(s["date"], "submission")
    for c in ("store_id", "sku_id"):
        s[c] = s[c].astype(str).str.strip()
        blank = s[c] == ""
        if blank.any():
            raise FormatError(f"Column '{c}' is blank in {int(blank.sum())} row(s) (first at data row "
                              f"{int(np.argmax(blank.values)) + 2}).")
    r = pd.to_numeric(s["rank"].str.strip(), errors="coerce")
    bad = r.isna() | (r != np.floor(r)) | (r < 1) | (r > MAX_PER_STORE_DAY)
    if bad.any():
        ex = s.loc[bad, "rank"].head(5).tolist()
        raise FormatError(f"Column 'rank' must be a whole number from 1 to {MAX_PER_STORE_DAY}; "
                          f"{int(bad.sum())} bad value(s), e.g. {ex}.")
    s["rank"] = r.astype(int)
    cf = pd.to_numeric(s["confidence"].str.strip(), errors="coerce")
    bad = cf.isna() | (cf < 0) | (cf > 1)
    if bad.any():
        ex = s.loc[bad, "confidence"].head(5).tolist()
        raise FormatError(f"Column 'confidence' must be a number between 0 and 1; {int(bad.sum())} bad value(s), "
                          f"e.g. {ex}.")
    s["confidence"] = cf.astype(float)
    blank_reason = s["reason"].astype(str).str.strip() == ""
    if blank_reason.any():
        warnings.append(f"{int(blank_reason.sum())} row(s) have an empty 'reason' -- the score is unaffected, "
                        f"but judges expect a one-line plain-English reason for every flag.")
    if "predicted_type" in s.columns:
        s["predicted_type"] = s["predicted_type"].astype(str).str.strip().str.lower()
        bad = ~s["predicted_type"].isin(list(VALID_TYPES) + [""])
        if bad.any():
            raise FormatError(f"Column 'predicted_type' may only contain {list(VALID_TYPES)} or be blank; "
                              f"found {sorted(s.predicted_type[bad].unique())[:5]}.")
    dup = s.duplicated(["date", "store_id", "sku_id"], keep=False)
    if dup.any():
        ex = s.loc[dup, ["date", "store_id", "sku_id"]].head(3).to_dict("records")
        raise FormatError(f"{int(dup.sum())} rows repeat the same (date, store_id, sku_id). Each SKU may appear at "
                          f"most once per store-day. Examples: {ex}")
    n = s.groupby(["date", "store_id"]).size()
    over = n[n > MAX_PER_STORE_DAY]
    if len(over):
        ex = [f"{d} / {st}: {c} rows" for (d, st), c in over.head(3).items()]
        raise FormatError(f"{len(over)} store-day(s) have more than {MAX_PER_STORE_DAY} rows -- the manager can only "
                          f"check {MAX_PER_STORE_DAY} SKUs per store per day. Examples: {ex}")
    rdup = s.duplicated(["date", "store_id", "rank"], keep=False)
    if rdup.any():
        ex = s.loc[rdup, ["date", "store_id", "rank"]].head(3).to_dict("records")
        raise FormatError(f"Ranks must be unique within each (date, store_id) list. Examples of repeats: {ex}")
    return s, warnings


def score(sub: pd.DataFrame, truth: pd.DataFrame) -> dict:
    dates = sorted(truth["date"].unique())
    stores = sorted(truth["store_id"].unique())
    scope = pd.MultiIndex.from_product([dates, stores], names=["date", "store_id"])
    pos = truth[truth["label_type"].isin(POSITIVE_TYPES)]

    in_scope = sub.set_index(["date", "store_id"]).index.isin(scope)
    n_ignored = int((~in_scope).sum())
    s = sub[in_scope].merge(truth[["date", "store_id", "sku_id", "label_type", "episode_id"]],
                            on=["date", "store_id", "sku_id"], how="left")
    s["hit"] = s["label_type"].isin(POSITIVE_TYPES)

    hits = s.groupby(["date", "store_id"])["hit"].sum().reindex(scope, fill_value=0)
    p_at_k = hits / K
    sub_sd = s.groupby(["date", "store_id"]).size().reindex(scope, fill_value=0)
    missing_sd = sub_sd[sub_sd == 0].index.tolist()
    n_pos_sd = pos.groupby(["date", "store_id"]).size().reindex(scope, fill_value=0)
    ceiling = float((np.minimum(n_pos_sd, K) / K).mean())

    total_hits = int(s["hit"].sum())
    n_pos_rows = len(pos)
    pos_eps = pos["episode_id"].unique()
    caught_eps = s.loc[s["hit"], "episode_id"].unique()
    res = {
        "primary_score": round(float(p_at_k.mean()), 6),
        "primary_metric": f"mean precision@{K} per store-day",
        "n_store_days_scored": int(len(scope)),
        "n_dates": len(dates),
        "n_stores": len(stores),
        "n_store_days_missing": len(missing_sd),
        "ceiling_precision_at_20": round(ceiling, 6),
        "primary_score_as_share_of_ceiling": round(float(p_at_k.mean()) / ceiling, 6) if ceiling > 0 else None,
        "row_recall": round(total_hits / n_pos_rows, 6) if n_pos_rows else None,
        "episode_recall": round(len(caught_eps) / len(pos_eps), 6) if len(pos_eps) else None,
        "n_hits": total_hits,
        "n_positive_rows_in_truth": int(n_pos_rows),
        "n_positive_episodes_in_truth": int(len(pos_eps)),
        "n_submitted_rows_scored": int(len(s)),
        "n_submitted_rows_ignored_out_of_scope": n_ignored,
        "share_of_flags_on_delisted_decoys": round(float(s["label_type"].isin(DECOY_TYPES).mean()), 6) if len(s) else 0.0,
        "mean_rows_per_scored_store_day": round(float(sub_sd.mean()), 3),
        "precision_at_20_by_date": {d: round(float(v), 4) for d, v in p_at_k.groupby(level="date").mean().items()},
        "precision_at_20_by_store": {st: round(float(v), 4) for st, v in p_at_k.groupby(level="store_id").mean().items()},
        "_missing_store_days_examples": [f"{d} / {st}" for d, st in missing_sd[:5]],
    }
    if "predicted_type" in s.columns:
        typed = s[s["label_type"].isin(VALID_TYPES) & (s["predicted_type"] != "")]
        res["type_accuracy"] = round(float((typed["predicted_type"] == typed["label_type"]).mean()), 6) if len(typed) else None
        res["type_accuracy_n_rows"] = int(len(typed))
        res["type_accuracy_by_true_type"] = {
            t: round(float((g["predicted_type"] == t).mean()), 4) for t, g in typed.groupby("label_type")}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score an R1 Phantom Stock-Out submission.")
    ap.add_argument("--submission", required=True, help="path to your CSV (e.g. submission/R1_predictions.csv)")
    ap.add_argument("--truth", required=True, help="truth CSV (data/dev_truth.csv for self-checks)")
    ap.add_argument("--out", default="score.json", help="where to write the JSON result (default score.json)")
    a = ap.parse_args(argv)
    try:
        truth = load_truth(a.truth)
        sub, warnings = load_submission(a.submission)
        res = score(sub, truth)
    except FormatError as e:
        fail(str(e))
    except Exception as e:  # noqa: BLE001  -- never show a traceback to participants
        fail(f"Unexpected problem while reading or scoring the files ({type(e).__name__}: {str(e)[:300]}). "
             f"Check that --submission is your CSV and --truth is data/dev_truth.csv, and that the CSV follows the format "
             f"in README.md.")

    print("=" * 66)
    print(" R1 Phantom Stock-Out Detector -- evaluation")
    print("=" * 66)
    for wmsg in warnings:
        print(f"WARNING: {wmsg}")
    if res["n_store_days_missing"]:
        print(f"WARNING: {res['n_store_days_missing']} of {res['n_store_days_scored']} scored store-days have NO rows "
              f"in your submission and score 0. First few: {res['_missing_store_days_examples']}")
    if res["n_submitted_rows_scored"] == 0:
        print("WARNING: NONE of your rows match a (date, store_id) in the truth file. Check the date format and the "
              "store_id spelling (e.g. STR-XXXX, upper case) and that you predicted the right window.")
    if res["n_submitted_rows_ignored_out_of_scope"]:
        print(f"note: {res['n_submitted_rows_ignored_out_of_scope']} row(s) are for dates/stores not in this truth "
              f"file and were ignored (that is fine).")
    print(f"Scored store-days          : {res['n_store_days_scored']} ({res['n_dates']} dates x {res['n_stores']} stores)")
    print(f"PRIMARY  precision@{K}       : {res['primary_score']:.4f}   (0..1, higher is better)")
    print(f"  best possible (ceiling)  : {res['ceiling_precision_at_20']:.4f}")
    print(f"Row recall                 : {res['row_recall']:.4f}  ({res['n_hits']} of {res['n_positive_rows_in_truth']} positive rows)")
    print(f"Episode recall             : {res['episode_recall']:.4f}  (of {res['n_positive_episodes_in_truth']} positive episodes)")
    print(f"Flags spent on delisted    : {res['share_of_flags_on_delisted_decoys']:.4f}")
    if "type_accuracy" in res:
        ta = res["type_accuracy"]
        print(f"Stretch type accuracy      : {ta if ta is None else round(ta, 4)}  (n={res['type_accuracy_n_rows']})")
    res["missing_store_days_examples"] = res.pop("_missing_store_days_examples", [])
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
