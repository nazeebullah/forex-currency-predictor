# 💱 End-to-End Forex Currency Exchange Rate Predictor

An end-to-end Machine Learning time-series forecasting web application built with **Python, Scikit-Learn, Streamlit, and Docker**. Trained on 20 years of daily Federal Reserve foreign exchange rate data (2000–2019), this app forecasts future exchange rates (up to 90 business days ahead) across 8 major global currencies against the US Dollar (`AUD`, `CAD`, `CHF`, `CNY`, `EUR`, `GBP`, `INR`, `JPY`).

## 🚀 Key Features
- **Time-Series Data Pipeline:** Automated cleaning of holiday/weekend market gaps (`ND`) using chronological forward/backward filling without data leakage.
- **Autoregressive Lag Engineering:** Transforms univariate currency series into supervised 10-day lag feature matrices (`GradientBoostingRegressor`).
- **Interactive Streamlit Dashboard:** Dynamic currency selection, customizable forecast horizon (5–90 business days), out-of-sample test metrics (`MAE`, `RMSE`, `MAPE`), historical vs. forecast visualization, and one-click CSV export.
- **Dockerized Deployment:** Fully containerized with `python:3.11-slim` for reproducible cross-platform execution.

## 📂 Project Structure
```text
forex-predictor/
├── data/
│   └── Foreign_Exchange_Rates.xls   # 20-year historical FX dataset (2000-2019)
├── models/                          # Serialized Gradient Boosting models (.pkl)
├── analysis.ipynb                   # EDA, ARIMA/Prophet/ML benchmarking notebook
├── train_local.py                   # Automated pipeline to train & serialize models
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container image build instructions
└── .dockerignore                    # Excludes local venv & raw artifacts from image