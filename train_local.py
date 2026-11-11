import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

file_path = "data/Foreign_Exchange_Rates.xls"
if not os.path.exists(file_path):
    url = "https://raw.githubusercontent.com/furkhan67/forex-currency-predictor/main/data/Foreign_Exchange_Rates.xls"
    urllib.request.urlretrieve(url, file_path)

df = pd.read_csv(file_path).drop(columns=['Unnamed: 0', 'Unnamed: 24'], errors='ignore')
df['Time Serie'] = pd.to_datetime(df['Time Serie'], format='%d-%m-%Y')
df = df.rename(columns={'Time Serie': 'Date'}).sort_values('Date').reset_index(drop=True)

currency_cols = [c for c in df.columns if c != 'Date']
df[currency_cols] = df[currency_cols].replace('ND', np.nan)
for col in currency_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[currency_cols] = df[currency_cols].ffill().bfill()

col_mapping = {
    'AUSTRALIA - AUSTRALIAN DOLLAR/US$': 'AUD',
    'EURO AREA - EURO/US$': 'EUR',
    'UNITED KINGDOM - UNITED KINGDOM POUND/US$': 'GBP',
    'CANADA - CANADIAN DOLLAR/US$': 'CAD',
    'CHINA - YUAN/US$': 'CNY',
    'INDIA - INDIAN RUPEE/US$': 'INR',
    'JAPAN - YEN/US$': 'JPY',
    'SWITZERLAND - FRANC/US$': 'CHF'
}
df = df.rename(columns=col_mapping)

def create_lag_features(series, n_lags=10):
    X, y = [], []
    values = series.values
    for i in range(n_lags, len(values)):
        X.append(values[i-n_lags:i])
        y.append(values[i])
    return np.array(X), np.array(y)

n_lags = 10
test_days = 60

for curr in col_mapping.values():
    series = df[['Date', curr]].dropna().reset_index(drop=True)
    X_all, y_all = create_lag_features(series[curr], n_lags=n_lags)
    
    # Evaluate on last 60 days
    gb_test = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    gb_test.fit(X_all[:-test_days], y_all[:-test_days])
    preds = gb_test.predict(X_all[-test_days:])
    y_true = y_all[-test_days:]
    
    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mape = np.mean(np.abs((y_true - preds) / y_true)) * 100

    # Fit on full series and save locally
    final_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    final_gb.fit(X_all, y_all)

    artifact = {
        'currency': curr,
        'model_type': 'GradientBoosting',
        'model': final_gb,
        'n_lags': n_lags,
        'last_date': str(series['Date'].iloc[-1].date()),
        'history_dates': [str(d.date()) for d in series['Date'].tail(180)],
        'history_values': series[curr].tail(180).tolist(),
        'metrics': {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'MAPE': round(mape, 2)}
    }
    joblib.dump(artifact, f"models/{curr}_model.pkl")
    print(f"✅ Saved models/{curr}_model.pkl")