import os
import joblib
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from typing import Tuple, List

from data_loader import fetch_data
from features import calculate_features
from preprocess import chronological_split, fit_and_transform_scalers, create_sliding_windows

def build_lstm_model(
        input_shape: Tuple[int, int] = (30, 7),
        lstm_units: list = [64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 5e-4
) -> tf.keras.Model:
    ''' Constructs and compiles a 2-layer Stacked LSTM Binary Classifier.'''

    model = Sequential([
        # Layer 1: LSTM with 64 units returning full sequences
        LSTM(
            lstm_units[0],
            input_shape=input_shape,
            return_sequences=True,
            kernel_regularizer=l2(1e-4) #L2 regularization penalizes oversized weights
        ),
        Dropout(dropout_rate),

        LSTM(
            lstm_units[1],
            return_sequences=False,
            kernel_regularizer=l2(1e-4)
        ),
        Dropout(dropout_rate),

        # Output Layer: 1 sigmoid neuron for binary probability P(UP)
        Dense(1, activation='sigmoid')
    ])

    # Compile with Binary Cross-Entropy and Adam Optimizer
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        model_save_path: str = "models/best_lstm_model.keras"
) -> tf.keras.callbacks.History:

    ''' Trains the LSTM model with EarlyStopping and ModelCheckpoint to prevent overfitting.'''

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    callbacks = [
        # 1. Stop training when validation loss stops improving
        EarlyStopping( monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        # 2. Save the best model weights to disk
        ModelCheckpoint(
            filepath=model_save_path, monitor='val_loss', save_best_only=True, verbose=0)
    ]

    print(f"\n--> Starting Model Training (Max Epochs: {epochs}, Batch Size: {batch_size})...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    print(f"--> Training completed! Best model checkpoint saved to '{model_save_path}'")
    return history

def train_single_stock(
        ticker: str = "^NSEI",
        start_date: str = "2015-01-01",
        end_date: str = "2026-08-01",
        epochs: int = 50,
        batch_size: int = 32
):
    ''' Trains a dedicated model and scaler for ONE specific stock'''

    clean_ticker = ticker.replace("^", ""). replace(".", "_")
    model_save_path = f"models/best_lstm_model_{clean_ticker}.keras"
    scaler_save_path = f"models/scaler_{clean_ticker}.joblib"

    print(f"\n=======================================================")
    print(f"🚀 Training Dedicated Model for Asset: '{ticker}'")
    print(f"=======================================================")

    raw_df = fetch_data(ticker=ticker, start_date=start_date, end_date=end_date)
    df = calculate_features(raw_df)
    feature_cols = ['Log_Return', 'HL_Spread', 'OC_Spread', 'Volume_ROC', 'SMA_20_Ratio', 'RSI_14_Scaled', 'MACD']

    train_df, val_df, test_df = chronological_split(df, train_ratio=0.8, val_ratio=0.1)
    train_scaled, val_scaled, test_scaled, scaler = fit_and_transform_scalers(train_df, val_df, test_df, feature_cols)
    
    joblib.dump(scaler, scaler_save_path)
    
    X_train, y_train = create_sliding_windows(train_scaled, train_df['Target_Direction'], window_size=30)
    X_val, y_val = create_sliding_windows(val_scaled, val_df['Target_Direction'], window_size=30)
    X_test, y_test = create_sliding_windows(test_scaled, test_df['Target_Direction'], window_size=30)
    
    lstm_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    history = train_model(lstm_model, X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, model_save_path=model_save_path)
    
    test_loss, test_acc = lstm_model.evaluate(X_test, y_test, verbose=0)
    print(f"🎯 [{ticker}] Out-of-Sample Accuracy: {test_acc * 100:.2f}% | Saved: {model_save_path}")
    return lstm_model
def train_universal_model(
    tickers: List[str] = ["^NSEI", "^GSPC", "AAPL", "MSFT", "RELIANCE.NS"],
    start_date: str = "2015-01-01",
    end_date: str = "2026-08-01",
    epochs: int = 50,
    batch_size: int = 64
):
    """
    Trains ONE Universal Foundation Model across a pooled basket of global stocks.
    """
    feature_cols = ['Log_Return', 'HL_Spread', 'OC_Spread', 'Volume_ROC', 'SMA_20_Ratio', 'RSI_14_Scaled', 'MACD']
    X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list = [], [], [], [], [], []
    
    print(f"\n=======================================================")
    print(f"🌍 Training Universal Cross-Asset Model across {len(tickers)} assets:")
    print(f"   {tickers}")
    print(f"=======================================================")
    
    for ticker in tickers:
        try:
            print(f"--> Ingesting & processing: '{ticker}'...")
            raw_df = fetch_data(ticker=ticker, start_date=start_date, end_date=end_date)
            df = calculate_features(raw_df)
            
            train_df, val_df, test_df = chronological_split(df, train_ratio=0.8, val_ratio=0.1)
            tr_s, v_s, te_s, _ = fit_and_transform_scalers(train_df, val_df, test_df, feature_cols)
            
            X_tr, y_tr = create_sliding_windows(tr_s, train_df['Target_Direction'], window_size=30)
            X_v, y_v = create_sliding_windows(v_s, val_df['Target_Direction'], window_size=30)
            X_te, y_te = create_sliding_windows(te_s, test_df['Target_Direction'], window_size=30)
            
            X_train_list.append(X_tr)
            y_train_list.append(y_tr)
            X_val_list.append(X_v)
            y_val_list.append(y_v)
            X_test_list.append(X_te)
            y_test_list.append(y_te)
        except Exception as e:
            print(f"⚠️ Warning: Could not process {ticker}: {e}")
            
    X_train_all = np.concatenate(X_train_list, axis=0)
    y_train_all = np.concatenate(y_train_list, axis=0)
    X_val_all = np.concatenate(X_val_list, axis=0)
    y_val_all = np.concatenate(y_val_list, axis=0)
    X_test_all = np.concatenate(X_test_list, axis=0)
    y_test_all = np.concatenate(y_test_list, axis=0)
    
    print(f"\n📊 Total Pooled Dataset: {X_train_all.shape[0]} Training Sequences")
    
    universal_model = build_lstm_model(input_shape=(X_train_all.shape[1], X_train_all.shape[2]))
    model_save_path = "models/universal_lstm_model.keras"
    
    history = train_model(universal_model, X_train_all, y_train_all, X_val_all, y_val_all, epochs=epochs, batch_size=batch_size, model_save_path=model_save_path)
    
    test_loss, test_acc = universal_model.evaluate(X_test_all, y_test_all, verbose=0)
    print(f"\n🏆 Universal Model Out-of-Sample Accuracy: {test_acc * 100:.2f}%")
    print(f"💾 Saved to: '{model_save_path}'")
    return universal_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single", choices=["single", "universal"], help="Training mode")
    parser.add_argument("--ticker", type=str, default="^NSEI", help="Stock ticker symbol for single mode")
    args = parser.parse_args()
    
    if args.mode == "single":
        train_single_stock(ticker=args.ticker)
    elif args.mode == "universal":
        train_universal_model(tickers=["^NSEI", "^GSPC", "AAPL", "MSFT", "RELIANCE.NS"])
