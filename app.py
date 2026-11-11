import os
import glob
import joblib
import sys
import sklearn._loss
sys.modules['_loss'] = sklearn._loss
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Block 1: Page Setup & Auto-Detecting Models
st.set_page_config(page_title="Forex Currency Predictor", page_icon="💱", layout="wide")

st.title("💱 Forex Currency Exchange Rate Predictor")
st.markdown("Forecast future exchange rates (vs. US$) using models trained on 20 years of historical currency data.")

# Look for .pkl files in models/ (also checks nested models/models/ just in case)
model_files = sorted(glob.glob("models/*_model.pkl") + glob.glob("models/models/*_model.pkl"))
available_currencies = sorted(list(set([os.path.basename(f).replace("_model.pkl", "") for f in model_files])))

if not available_currencies:
    st.error("⚠️ No saved models found in `models/`. Make sure your `.pkl` files are extracted inside the `models/` folder.")
    st.stop()

# Block 2: Sidebar User Controls
st.sidebar.header("⚙️ Forecast Controls")
selected_currency = st.sidebar.selectbox("Select Currency", available_currencies)
horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=5, max_value=90, value=30, step=5)


# Block 3: Model Caching & Recursive Forecast Engine
@st.cache_resource
def load_currency_model(currency: str):
    path1 = f"models/{currency}_model.pkl"
    path2 = f"models/models/{currency}_model.pkl"
    return joblib.load(path1 if os.path.exists(path1) else path2)


def make_forecast(currency: str, steps: int):
    artifact = load_currency_model(currency)
    model = artifact["model"]
    model_type = artifact["model_type"]
    n_lags = artifact.get("n_lags", 10)
    last_date = pd.to_datetime(artifact["last_date"])
    history_values = artifact["history_values"]

    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)

    preds = []
    window = list(history_values[-n_lags:])
    for _ in range(steps):
        x_input = np.array(window[-n_lags:]).reshape(1, -1)
        next_val = float(model.predict(x_input)[0])
        preds.append(next_val)
        window.append(next_val)

    preds = np.array(preds)
    forecast_df = pd.DataFrame({
        "Date": future_dates.strftime("%Y-%m-%d"),
        f"Predicted {currency}/US$": np.round(preds, 4)
    })
    return artifact, future_dates, preds, forecast_df


# Block 4: Dashboard KPIs, Chart & CSV Export
artifact, future_dates, preds, forecast_df = make_forecast(selected_currency, horizon)

col1, col2, col3, col4 = st.columns(4)
last_actual = artifact["history_values"][-1]
delta = preds[-1] - last_actual

col1.metric("Currency Pair", f"{selected_currency} / US$")
col2.metric("Winning Model", artifact["model_type"])
col3.metric("Last Recorded Rate", f"{last_actual:.4f}")
col4.metric(f"Day +{horizon} Forecast", f"{preds[-1]:.4f}", f"{delta:+.4f}")

if "metrics" in artifact:
    m = artifact["metrics"]
    st.caption(f"**Test Set Performance ({selected_currency}):** MAE: `{m['MAE']}` | RMSE: `{m['RMSE']}` | MAPE: `{m['MAPE']}%`")

st.subheader(f"📈 {selected_currency}/US$ Historical Trend & {horizon}-Day Forecast")
fig, ax = plt.subplots(figsize=(12, 4.5))
hist_dates = pd.to_datetime(artifact["history_dates"])

ax.plot(hist_dates, artifact["history_values"], label="Historical Rate (Last 180 Days)", color="#1f77b4", linewidth=2)
ax.plot(future_dates, preds, label=f"{horizon}-Day Forecast ({artifact['model_type']})", color="#ff7f0e", linestyle="--", linewidth=2.2)
ax.set_xlabel("Date")
ax.set_ylabel(f"{selected_currency} per 1 US$")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.subheader("📋 Forecast Data Table")
st.dataframe(forecast_df, use_container_width=True)

st.download_button(
    label="⬇️ Download Forecast as CSV",
    data=forecast_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_currency}_forecast_{horizon}days.csv",
    mime="text/csv"
)