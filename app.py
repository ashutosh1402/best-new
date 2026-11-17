import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from xgboost import XGBRegressor
from scipy.stats import norm

st.set_page_config(page_title="AI Market Risk & Return Engine", layout="wide")
st.title("📉 AI/ML Market Risk & Next-Day Return System")

# --- SIDEBAR: INTERACTIVE USER INPUTS ---
st.sidebar.header("1. Market & Portfolio Inputs")
market = st.sidebar.selectbox("Select Exchange", ["India (NSE)", "India (BSE)", "US / Global"])
raw_ticker = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE, TCS, AAPL)", "RELIANCE").strip().upper()

# Automatically attach .NS or .BO for Indian markets
if market == "India (NSE)" and not raw_ticker.endswith(".NS") and not raw_ticker.startswith("^"):
    ticker = f"{raw_ticker}.NS"
elif market == "India (BSE)" and not raw_ticker.endswith(".BO") and not raw_ticker.startswith("^"):
    ticker = f"{raw_ticker}.BO"
else:
    ticker = raw_ticker

currency = "₹" if "India" in market else "$"
default_val = 1000000 if "India" in market else 100000

portfolio_val = st.sidebar.number_input(f"Portfolio Value ({currency})", min_value=1000, value=default_val, step=10000)
confidence = st.sidebar.slider("VaR Confidence Level (%)", 90, 99, 95)

st.sidebar.header("2. Stress Test Scenario")
market_shock = st.sidebar.slider("Simulate Market Crash (%)", -50, 0, -15)
vol_multiplier = st.sidebar.slider("Volatility Spike Multiplier", 1.0, 4.0, 2.0)

# --- LOAD & ENGINEER FEATURES ---
@st.cache_data(ttl=3600)
def load_data(symbol):
    df = yf.download(symbol, period="5y", progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Return'] = df['Close'].pct_change()
    df['Momentum_5d'] = df['Close'].pct_change(5)
    df['Volatility_21d'] = df['Return'].rolling(21).std()
    df['Vol_Change'] = df['Volume'].replace(0, np.nan).pct_change().fillna(0)
    return df.dropna(subset=['Return', 'Momentum_5d', 'Volatility_21d'])

df = load_data(ticker)

if df.empty:
    st.error(f"Could not fetch data for '{ticker}'. Check the exact NSE/BSE symbol (e.g., HDFCBANK, INFY, TATAMOTORS).")
    st.stop()

# Prepare training data for Next-Day Volatility & Next-Day Return
df['Target_Vol'] = df['Volatility_21d'].shift(-1)
df['Target_Return'] = df['Return'].shift(-1)
train_df = df.dropna(subset=['Target_Vol', 'Target_Return'])

features = ['Return', 'Momentum_5d', 'Volatility_21d', 'Vol_Change']
X = train_df[features].replace([np.inf, -np.inf], 0)
latest_features = df[features].iloc[[-1]].replace([np.inf, -np.inf], 0)

# --- 1. VOLATILITY FORECASTING (XGBoost vs Historical Std Dev) ---
xgb_vol_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
xgb_vol_model.fit(X, train_df['Target_Vol'])

xgb_daily_vol = float(xgb_vol_model.predict(latest_features)[0])
xgb_annual_vol = xgb_daily_vol * np.sqrt(252)

# Traditional 21-day Historical Standard Deviation
std_daily_vol = float(df['Volatility_21d'].iloc[-1])
std_annual_vol = std_daily_vol * np.sqrt(252)

# --- 2. 1-DAY VALUE AT RISK (VaR): STANDARD DEVIATION vs XGBOOST ---
z_score = abs(norm.ppf(1 - (confidence / 100)))

# Traditional Std Dev VaR
var_std_pct = z_score * std_daily_vol
var_std_amount = portfolio_val * var_std_pct

# XGBoost ML VaR
var_xgb_pct = z_score * xgb_daily_vol
var_xgb_amount = portfolio_val * var_xgb_pct

# --- 3. NEXT-DAY EXPECTED RETURN & RISK-RETURN RATIO (Gradient Boosting) ---
gb_ret_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
gb_ret_model.fit(X, train_df['Target_Return'])

expected_next_return = float(gb_ret_model.predict(latest_features)[0])
expected_next_pnl = portfolio_val * expected_next_return

# Risk-to-Return Ratio: Expected Reward vs. XGBoost 1-Day Max Risk (VaR)
# (Higher ratio = better reward relative to downside risk; negative if expected return is negative)
reward_to_risk_ratio = expected_next_pnl / var_xgb_amount if var_xgb_amount > 0 else 0

# --- 4. ANOMALY DETECTION (Isolation Forest) ---
iso = IsolationForest(contamination=0.02, random_state=42)
df['Anomaly'] = iso.fit_predict(df[['Return', 'Volatility_21d']])
anomalies = df[df['Anomaly'] == -1]

# --- 5. STRESS TEST CALCULATION (Based on XGBoost VaR) ---
stressed_var = var_xgb_amount * vol_multiplier
stressed_loss = (portfolio_val * abs(market_shock / 100)) + stressed_var

# --- DISPLAY METRICS ---
st.caption(f"Live Analytics for **{ticker}** | Latest Close: **{currency}{df['Close'].iloc[-1]:,.2f}**")

st.subheader("1️⃣ 1-Day Downside Risk: Standard Deviation VaR vs. XGBoost VaR")
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    f"Std Dev {confidence}% 1-Day VaR",
    f"{currency}{var_std_amount:,.2f}",
    f"-{var_std_pct:.2%} (Annual Vol: {std_annual_vol:.1%})",
    delta_color="inverse"
)
col2.metric(
    f"XGBoost {confidence}% 1-Day VaR",
    f"{currency}{var_xgb_amount:,.2f}",
    f"-{var_xgb_pct:.2%} (Annual Vol: {xgb_annual_vol:.1%})",
    delta_color="inverse"
)
col3.metric(
    "Stress Test Projected Loss",
    f"{currency}{stressed_loss:,.2f}",
    f"{market_shock}% Shock + {vol_multiplier}x Vol",
    delta_color="inverse"
)
col4.metric(
    "Abnormal Days Detected (5Y)",
    f"{len(anomalies)} days",
    "Top 2% Extreme Swings"
)

st.subheader("2️⃣ Next-Day Outlook & Risk-Return Tradeoff")
r_col1, r_col2, r_col3 = st.columns(3)
r_col1.metric(
    "Expected Next-Day Return (%)",
    f"{expected_next_return:+.2%}",
    "Gradient Boosting Forecast"
)
r_col2.metric(
    "Expected Next-Day Profit / Loss",
    f"{currency}{expected_next_pnl:+,.2f}",
    f"On {currency}{portfolio_val:,.0f} Capital"
)
r_col3.metric(
    "Next-Day Return-to-Risk Ratio",
    f"{reward_to_risk_ratio:+.2f}x",
    "Expected Gain ÷ XGBoost VaR"
)

# --- INTERACTIVE CHART ---
st.subheader(f"Abnormal Market Movements Detected: {ticker}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(
    x=anomalies.index, y=anomalies['Close'], mode='markers',
    marker=dict(color='red', size=8), name='ML Anomaly Flag'
))
fig.update_layout(yaxis_title=f"Price ({currency})", xaxis_title="Date")
st.plotly_chart(fig, use_container_width=True)
