# finnhub_forex_streamlit.py
import os
import time
import math
import pandas as pd
import numpy as np
import finnhub
import streamlit as st
from datetime import datetime, timedelta, timezone
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import joblib

# ---------- Config ----------
API_KEY = os.getenv("FINNHUB_API_KEY")
if not API_KEY:
    st.sidebar.error("Set FINNHUB_API_KEY environment variable before running.")
    st.stop()

client = finnhub.Client(api_key=API_KEY)

# Helper: fetch forex candles
def fetch_forex_candles(symbol="OANDA:EUR_USD", resolution="1", days=7):
    """
    resolution: "1","5","15","30","60","D" etc.
    days: how many days back to fetch
    """
    now = int(datetime.now(tz=timezone.utc).timestamp())
    _from = int((datetime.now(tz=timezone.utc) - timedelta(days=days)).timestamp())
    res = client.forex_candles(symbol, resolution, _from, now)
    if res.get("s") != "ok":
        raise ValueError(f"API returned status {res.get('s')}: {res}")
    df = pd.DataFrame({
        "t": pd.to_datetime(res["t"], unit="s", utc=True),
        "o": res["o"],
        "h": res["h"],
        "l": res["l"],
        "c": res["c"],
        "v": res["v"]
    }).set_index("t")
    return df

# Feature engineering
def prepare_features(df, horizon_bars=10):
    # df expected with index = UTC datetime and columns o,h,l,c,v
    df = df.copy()
    df["ret"] = df["c"].pct_change()
    # lag returns
    for lag in [1,2,3,5,8]:
        df[f"ret_lag_{lag}"] = df["ret"].shift(lag)
    # rolling volatility and mean
    df["vol_5"] = df["ret"].rolling(5).std()
    df["vol_15"] = df["ret"].rolling(15).std()
    df["rsi_14"] = RSIIndicator(close=df["c"], window=14).rsi()
    # target: forward return over horizon_bars
    df["future_c"] = df["c"].shift(-horizon_bars)
    df["fwd_ret"] = df["future_c"] / df["c"] - 1
    # binary label: up (1) if positive, else 0
    df["label"] = (df["fwd_ret"] > 0).astype(int)
    df = df.dropna()
    return df

# Train simple model
def train_model(df, feature_cols):
    # time-based split - last 20% as test
    n = len(df)
    split = int(n * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]
    X_train = train[feature_cols]
    y_train = train["label"]
    X_test = test[feature_cols]
    y_test = test["label"]

    model = XGBClassifier(
        max_depth=4, n_estimators=200, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba))
    }
    return model, metrics, (X_test, y_test, proba)

# ---------- Streamlit UI ----------
st.title("Finnhub Forex: Fetch → Store in Session → Train → Predict (10-15 min)")

with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Finnhub Forex Symbol", value="OANDA:EUR_USD")
    resolution = st.selectbox("Resolution (minutes)", options=["1","5","15","30","60"], index=1)
    days = st.number_input("Days history to fetch", min_value=1, max_value=60, value=7)
    horizon_minutes = st.number_input("Prediction horizon (minutes)", min_value=5, max_value=60, value=15)
    # compute horizon bars depending on resolution
    res_int = int(resolution) if resolution.isdigit() else None
    if res_int:
        horizon_bars = max(1, horizon_minutes // res_int)
    else:
        # daily resolution
        horizon_bars = 1

# Button: fetch
if st.button("Fetch candles and store in session"):
    try:
        df = fetch_forex_candles(symbol=symbol, resolution=resolution, days=days)
        st.session_state["df"] = df
        st.success(f"Fetched {len(df)} bars and stored in session (index UTC). Latest: {df.index[-1]}")
    except Exception as e:
        st.error(f"Fetch error: {e}")

if "df" in st.session_state:
    df = st.session_state["df"]
    st.subheader("Price preview (latest rows)")
    st.dataframe(df.tail(10))

    # Prepare features + label
    st.info("Preparing features and labels...")
    prepared = prepare_features(df, horizon_bars=horizon_bars)
    st.session_state["prepared"] = prepared

    st.subheader("Prepared features (head)")
    st.dataframe(prepared.head())

    # Feature list
    feat_cols = [c for c in prepared.columns if c.startswith("ret_lag_") or c.startswith("vol_") or c.startswith("rsi_")]
    st.write("Features used:", feat_cols)

    # Train model button
    if st.button("Train model on stored data"):
        model, metrics, testinfo = train_model(prepared, feat_cols)
        st.session_state["model"] = model
        st.success(f"Trained model. Test metrics: {metrics}")

    # Predict next move
    if "model" in st.session_state:
        model = st.session_state["model"]
        latest_row = prepared.iloc[[-1]][feat_cols]  # most recent features
        prob_up = model.predict_proba(latest_row)[:,1][0]
        st.metric("Predicted prob. UP", f"{prob_up:.3f}")
        if prob_up > 0.6:
            st.success(f"Signal: BUY (prob {prob_up:.2f})")
        elif prob_up < 0.4:
            st.warning(f"Signal: SELL (prob {prob_up:.2f})")
        else:
            st.info(f"Signal: NEUTRAL (prob {prob_up:.2f})")

    # Optionally save prepared & model to disk
    if st.button("Save prepared + model to disk"):
        prepared.to_pickle("prepared.pkl")
        joblib.dump(st.session_state.get("model"), "model.joblib")
        st.success("Saved prepared.pkl and model.joblib")

else:
    st.info("No data in session. Click 'Fetch candles and store in session' to start.")
