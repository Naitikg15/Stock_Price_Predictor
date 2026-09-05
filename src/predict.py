import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta

from data_loader import fetch_data
from features import calculate_features

def predict_next_day (ticker: str = "^NSEI") -> dict:
    '''
    Ingests the latest market data, scales the most recent 30-day window, and runs LSTM inference to forecast tomorrow's live directional probability.
    '''

    clean_ticker = ticker.replace("^", "").replace(".", "_")
    # 1. Calculate date range: Fetch the last ~160 calendar days
    # (Why 160 days? We need ~80 trading days: 50 days to warm up EMA-50 & RSI, plus the 30-day window!)

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=160)).strftime('%Y-%m-%d')

    # 2. Ingeest latest market data and compute casual features
    raw_df = fetch_data(ticker=ticker, start_date=start_date, end_date=end_date, save_raw=False)
    df = calculate_features(raw_df)

    if len(df) < 30:
        raise ValueError(f"Not enough clean trading rows ({len(df)} rows). Need at least 30 rows.")

    # 3. Step 2: Dynamic Model & Scaler Resolution
    model_path = f"models/best_lstm_model_{clean_ticker}.keras"
    if not os.path.exists(model_path):
        model_path = "models/universal_lstm_model.keras"

    scaler_path = f"models/scaler_{clean_ticker}.joblib"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found. Please train first via src/models.py")

    # 4. Step 3: Scale features using saved scaler (or compute from recent history)

    feature_cols = ['Log_Return', 'HL_Spread', 'OC_Spread', 'Volume_ROC', 'SMA_20_Ratio', 'RSI_14_Scaled', 'MACD']

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        scaled_features = scaler.transform(df[feature_cols])
    else:
        # If dedicated scaler not on disk, fit on recent history
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_features = scaler.fit_transform(df[feature_cols])

    # 5. Extract the MOST RECENT 30-day window and shape into 3D Tensor (1, 30, 7)

    latest_30_days = scaled_features[-30:]
    input_tensor = latest_30_days.reshape(1, 30, len(feature_cols))

    # 6. Step 4: Run LSTM Inference
    lstm_model = tf.keras.models.load_model(model_path)
    pred_prob = float(lstm_model.predict(input_tensor, verbose=0)[0][0])

    pred_direction = "UP" if pred_prob > 0.5 else "DOWN"
    confidence = pred_prob if pred_prob > 0.5 else (1.0 - pred_prob)

    # 7. Extract the latest market summary
    latest_row = df.iloc[-1]
    latest_date = df.index[-1].strftime('%Y-%m-%d')
    latest_close = float(raw_df['Close'].iloc[-1])

    # 8.  Package the response dictionary
    result = {
        "status": "success",
        "ticker": ticker,
        "model_used": os.path.basename(model_path),
        "latest_date": latest_date,
        "latest_close": latest_close,
        "prediction": pred_direction,
        "probability_up": round(pred_prob * 100, 2),
        "confidence": round(confidence * 100, 2),
        "indicators": {
            "rsi_14": round(float(latest_row['RSI_14']), 2),
            "macd": round(float(latest_row['MACD']), 5),
            "sma_20": round(float(latest_row['SMA_20']), 2),
            "ema_50": round(float(latest_row['EMA_50']), 2),
            "daily_return_pct": round(float(latest_row['Log_Return']) * 100, 2)
        }
    }

    return result

# -------------------------------------------------------------
# CLI Runner for Node.js / Terminal execution
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Live LSTM Stock Prediction Inference")
    parser.add_argument("--ticker", type=str, default="^NSEI", help="Stock ticker symbol (e.g. ^NSEI, AAPL, RELIANCE.NS)")
    args = parser.parse_args()
    
    try:
        output = predict_next_day(ticker=args.ticker)
        # Print JSON directly so Node.js can parse it easily
        print(json.dumps(output, indent=2))
    except Exception as e:
        error_output = {"status": "error", "message": str(e)}
        print(json.dumps(error_output))
        sys.exit(1)