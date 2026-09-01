import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd. DataFrame]:

    '''Splits time- series data chronologically into Train, Validation, and Test sets.
    Preserves strict temporal order without shuffling.
    '''

    n=len(df)
    train_end = int(n * train_ratio)
    val_end = int(n *(train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df

def fit_and_transform_scalers( train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    '''Fits MinMaxScaler strictly on the training set to prevent data leakage, then transforms the training, validation, and test feature matrices.'''

    scaler = MinMaxScaler(feature_range=(0,1))

    # Fit Scaler only on training features
    train_scaled = scaler.fit_transform(train_df[feature_cols])

    # Transform validation and test sets using the trainig parameters
    val_scaled = scaler.transform(val_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])

    return train_scaled, val_scaled, test_scaled, scaler

def create_sliding_windows( data: np.ndarray, targets: pd.Series, window_size: int=30) -> Tuple[np.ndarray, np.ndarray]:
    '''Transforms 2D scaled feature arrays into 3D sequential tensors [samples, window_size, features] and extracts the corresponding target labels.'''

    X, y=[], []
    target_values = targets.values
    # Slide the window from day 0 to the end of the dataset
    for i in range(len(data) - window_size):
        #Extract a 30-day historical window of the features
        window = data[i : i+ window_size]
        #Extract the target of the last day in that window
        label = target_values[i + window_size - 1]

        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    from data_loader import fetch_data
    from features import calculate_features

    # Load raw data & engineer features
    raw_df = fetch_data(ticker="^NSEI", start_date="2015-01-01", end_date="2026-08-01", save_raw= False)
    df=calculate_features(raw_df)

    # Select the 7 feature columns we want ot feed into the LSTM
    feature_cols = [
        'Log_Return', 'HL_Spread', 'OC_Spread', 'Volume_ROC', 'SMA_20_Ratio', 'RSI_14_Scaled', 'MACD'
    ]

    # Chronological Split (80% Train, 10% Val, 10% Test)
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.8, val_ratio=0.1)

    # Fit Scaler on Train, Transform Val & Test
    train_scaled, val_scaled, test_scaled, scaler = fit_and_transform_scalers(train_df, val_df, test_df, feature_cols)

    # Generate 30-day 3D Sliding Windows
    X_train, y_train = create_sliding_windows(train_scaled, train_df['Target_Direction'], window_size=30)
    X_val, y_val = create_sliding_windows(val_scaled, val_df['Target_Direction'], window_size=30)
    X_test, y_test = create_sliding_windows(test_scaled, test_df['Target_Direction'], window_size=30)

    print("\n--- Preprocessing Pipeline Completed Successfully! ---")
    print(f"X_train Shape : {X_train.shape} | y_train Shape : {y_train.shape}")
    print(f"X_test Shape : {X_test.shape} | y_test Shape : {y_test.shape}")
    print("\nSample check: First sequence shape (Days, Features):", X_train[0].shape)