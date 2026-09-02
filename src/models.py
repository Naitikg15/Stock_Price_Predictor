import os
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from typing import Tuple

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
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # 2. Save the best model weights to disk
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
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

if __name__ == "__main__":
    from data_loader import fetch_data
    from features import calculate_features
    from preprocess import chronological_split, fit_and_transform_scalers, create_sliding_windows

    # Ingest Data & Engineer Features
    raw_df = fetch_data(ticker="^NSEI", start_date="2015-01-01", end_date="2026-08-01")
    df= calculate_features(raw_df)

    # Define Features & Preprocess
    feature_cols = ['Log_Return', 'HL_Spread', 'OC_Spread', 'Volume_ROC', 'SMA_20_Ratio', 'RSI_14_Scaled', 'MACD']
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.8, val_ratio=0.1)
    
    train_scaled, val_scaled, test_scaled, scaler = fit_and_transform_scalers(train_df, val_df, test_df, feature_cols)

    X_train, y_train = create_sliding_windows(train_scaled, train_df['Target_Direction'], window_size=30)
    X_val, y_val = create_sliding_windows(val_scaled, val_df['Target_Direction'], window_size=30)
    X_test, y_test = create_sliding_windows(test_scaled, test_df['Target_Direction'], window_size=30)

    # Build the Stacked LSTM Network
    lstm_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    print("\n--- Model Architecture Summary ---")
    lstm_model.summary()

    # Train the Model!
    history = train_model(lstm_model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, patience=10)

    # Quick Test Evaluation on Unseen Out-of-Sample Data
    test_loss, test_acc = lstm_model.evaluate(X_test, y_test, verbose=0)
    print(f"\n==========================================")
    print(f"🎯 Out-of-Sample Test Accuracy: {test_acc * 100:.2f}%")
    print(f"📉 Out-of-Sample Test Loss    : {test_loss:.4f}")
    print(f"==========================================")