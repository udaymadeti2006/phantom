#!/usr/bin/env python3
"""Simple Streamlit manager view for PhantomPulse AI."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "daily_sales_inventory.parquet"
DETAILS = ROOT / "outputs" / "alert_details.csv"
SUBMISSION = ROOT / "submission" / "R1_predictions.csv"

if not st.runtime.exists():
    raise SystemExit("This dashboard must be run with Streamlit, for example: streamlit run FULL_CODE/dashboard.py")

st.set_page_config(page_title="PhantomPulse AI", layout="wide")
st.title("PhantomPulse AI")
st.caption("Demand-aware inventory investigation — alerts are candidates for physical verification, not proof of missing stock.")

if not DETAILS.exists():
    st.error("No alert details found. Run `python FULL_CODE/phantompulse.py --mode submit` first.")
    st.stop()

alerts = pd.read_csv(DETAILS, parse_dates=["date"])
if alerts.empty:
    st.warning("No alerts are available.")
    st.stop()

min_date, max_date = alerts["date"].min().date(), alerts["date"].max().date()
selected_date = st.date_input("Business date", value=max_date, min_value=min_date, max_value=max_date)
day = alerts[alerts["date"].dt.date == selected_date].copy()
stores = sorted(day["store_id"].astype(str).unique())
selected_store = st.selectbox("Store", stores)
store_day = day[day["store_id"].astype(str) == selected_store].sort_values("rank")

c1, c2, c3 = st.columns(3)
c1.metric("Recommended checks", len(store_day))
c2.metric("High-ranked candidates", int((store_day["rank"] <= 5).sum()))
c3.metric("Top-20 estimated opportunity", f"₹{store_day['estimated_lost_revenue_to_date_inr'].sum():,.0f}")

st.subheader("Daily inspection queue")
show_cols = ["rank", "sku_id", "confidence", "zero_run", "stock", "reason"]
st.dataframe(store_day[show_cols], use_container_width=True, hide_index=True)

options = store_day.apply(lambda r: f"#{int(r['rank'])} — {r['sku_id']} — risk {r['confidence']:.3f}", axis=1).tolist()
selected_label = st.selectbox("Investigate a candidate", options)
selected_idx = options.index(selected_label)
alert = store_day.iloc[selected_idx]
sku_id = str(alert["sku_id"])
alert_date = pd.Timestamp(alert["date"])
st.subheader(f"Evidence card: {sku_id}")
st.write(alert["reason"])
m1, m2, m3 = st.columns(3)
m1.metric("Model risk score", f"{float(alert['confidence']):.3f}")
m2.metric("Zero-sales streak through previous close", f"{int(alert['zero_run'])} days")
m3.metric("Recorded stock at previous close", "Unknown" if pd.isna(alert["stock"]) else f"{int(alert['stock'])} units")

st.info(
    f"Estimated demand shortfall to date: {float(alert['estimated_missed_units_to_date']):.1f} units. "
    f"Estimated sales opportunity: ₹{float(alert['estimated_lost_revenue_to_date_inr']):,.0f}. "
    "This is conditional on the issue being real and may not all be recoverable. It is not confirmed revenue."
)

st.subheader("Record physical inspection")
feedback_path = ROOT / "outputs" / "inspection_feedback.csv"
with st.form("inspection_feedback_form"):
    outcome = st.selectbox(
        "Inspection outcome",
        ["confirmed_phantom", "confirmed_misplaced", "available", "delisted",
         "inventory_record_error", "inconclusive"]
    )
    resolution_date = st.date_input("Resolution date (if resolved)", value=selected_date)
    recovered_units = st.number_input("Units recovered / corrected (if known)", min_value=0, value=0, step=1)
    notes = st.text_input("Manager notes (optional)")
    submitted = st.form_submit_button("Save inspection result")
    if submitted:
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_row = pd.DataFrame([{
            "inspection_date": str(selected_date),
            "store_id": selected_store,
            "sku_id": sku_id,
            "alert_rank": int(alert["rank"]),
            "predicted_risk_score": float(alert["confidence"]),
            "outcome": outcome,
            "resolution_date": str(resolution_date),
            "recovered_units": int(recovered_units),
            "notes": notes,
        }])
        feedback_row.to_csv(
            feedback_path, mode="a", header=not feedback_path.exists(), index=False
        )
        st.success(f"Inspection result saved to {feedback_path.name}. Review outcomes before using them for retraining.")

if not DATA.exists():
    st.warning("The organizer Parquet file is not in data/. Copy it from the provided dataset pack to show charts.")
    st.stop()

start = alert_date - pd.Timedelta(days=84)
try:
    hist = pd.read_parquet(
        DATA,
        columns=["date", "store_id", "sku_id", "units_sold", "on_hand_eod", "on_promo"],
        filters=[("date", ">=", start), ("date", "<", alert_date)]
    )
except Exception:
    # Fallback if the installed Parquet engine does not support filters.
    hist = pd.read_parquet(DATA, columns=["date", "store_id", "sku_id", "units_sold", "on_hand_eod", "on_promo"])
    hist = hist[(hist["date"] >= start) & (hist["date"] < alert_date)]

hist["date"] = pd.to_datetime(hist["date"])
hist = hist[(hist["store_id"].astype(str) == selected_store) & (hist["sku_id"].astype(str) == sku_id)].sort_values("date")
if hist.empty:
    st.warning("No historical rows were found for this store-SKU.")
    st.stop()
hist["expected_sales_28d"] = hist["units_sold"].shift(1).rolling(28, min_periods=7).mean()
chart_start = alert_date - pd.Timedelta(days=56)
plot_hist = hist[hist["date"] >= chart_start].copy()

st.subheader("Sales vs expected demand (information available before the alert date)")
sales_long = plot_hist.melt(
    id_vars=["date"], value_vars=["units_sold", "expected_sales_28d"],
    var_name="series", value_name="units"
)
fig_sales = px.line(sales_long, x="date", y="units", color="series",
                    labels={"units": "Units", "date": "Date", "series": "Series"})
fig_sales.add_vline(x=alert_date, line_dash="dash", annotation_text="Alert date (no same-day data used)")
st.plotly_chart(fig_sales, use_container_width=True)

st.subheader("System-recorded inventory")
fig_stock = px.line(plot_hist, x="date", y="on_hand_eod", markers=True,
                    labels={"on_hand_eod": "Recorded units on hand", "date": "Date"})
fig_stock.add_vline(x=alert_date, line_dash="dash", annotation_text="Alert date")
st.plotly_chart(fig_stock, use_container_width=True)

st.caption("Manager action: verify shelf, backroom and system record. Record the physical result separately; the model cannot prove theft or shelf absence from sales data alone.")
