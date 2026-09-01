import numpy as np
import pandas as pd

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()

    # Calculate daily log returns using a function
    df['Log_Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

    # Intraday Spreads & Volume Dynamics
    df['HL_Spread'] = (df['High'] - df['Low'])/df['Low']
    df['OC_Spread'] = (df['Close'] - df['Open'])/df['Open']
    df['Volume_ROC'] = df['Volume'].pct_change()

    # Trend Indicators? Average Ratios
    
    # SMA (20 Day Simple Moving Average)
    df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()

    # EMA(50 Day Exponential Moving Average)
    df['EMA_50'] = df['Adj Close'].ewm(span=50, adjust=False).mean()

    #Convert to Stationary ratios relative to current price
    df['SMA_20_Ratio'] = (df['Adj Close'] / df['SMA_20']) - 1.0
    df['SMA_50_Ratio'] = (df['Adj Close'] / df['EMA_50']) - 1.0

    #Momentum Oscillators

    # --- RSI (14 Days) ---

    delta=df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9) # 1e-9 avoids division by zero if loss is 0
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))
    df['RSI_14_Scaled'] = df['RSI_14'] / 100.0 # Normalized to [0,1] range

    # ---MACD (12, 26, 9)---
    ema_12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Adj Close'].ewm(span=26, adjust=False).mean()

    df['MACD'] = (ema_12 - ema_26) / df['Adj Close']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Prediction Targets (Looking forward: t+1)

    # Continuous Target: Tomorro's Log Return (for regression)
    df['Target_Return'] = df['Log_Return'].shift(-1)

    #Binry Target: 1 if tomorro is UP (> 0), 0 if DOWN (<= 0)
    df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)

    # Drop warm-up rows (first ~50 days) and the final row without a target
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()


    return df

if __name__ == "__main__":
    from data_loader import fetch_data

    raw_data = fetch_data(ticker="^NSEI", start_date="2015-01-01", end_date="2026-08-01", save_raw=False)

    featured_data = calculate_features(raw_data)

    print("\n--- Feature Engineering Pipeline Succeeded! ---")
    print(f"Original Raw Rows : {len(raw_data)}")
    print(f"Clean Feature Rows: {len(featured_data)} (after dropping warm-up NaNs)")

    print("\nSample of Engineered Features:")
    display_cols = ['Log_Return', 'HL_Spread', 'SMA_20_Ratio', 'RSI_14_Scaled', 'MACD', 'Target_Direction']
    print(featured_data[display_cols].head())

    print("\nMarket Class Balance (% of Days):")
    up_down_pct = featured_data['Target_Direction'].value_counts(normalize=True) * 100
    print(f"Down Days (0): {up_down_pct.get(0, 0):.2f}%")
    print(f"  Up Days (1): {up_down_pct.get(1, 0):.2f}%")