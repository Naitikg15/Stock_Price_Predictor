import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Any
def evaluate_predictions(y_true: np.ndarray, y_pred_prob: np.ndarray, model_name: str = "LSTM") -> Dict[str, Any]:
    ''' Evaluates classification performance: Accuracy, Precision, Recall, F1, and Confusion Matrix.'''

    # Convert continuous probabilities (0.0 to 1.0) into binary decisions (> 0.5 is UP)
    y_pred = (y_pred_prob.flatten() > 0.5).astype(int)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division = 0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Print report
    print(f"\n=======================================================")
    print(f"📊 {model_name} Classification Report")
    print(f"=======================================================")
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}% (Accuracy when predicting UP)")
    print(f"Recall    : {rec * 100:.2f}% (Percentage of UP days caught)")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"\nConfusion Matrix Breakdown:")
    print(f"   True Negative  (Saved from Drops) : {cm[0][0]}")
    print(f"   False Positive (Bought the Drop)  : {cm[0][1]}")
    print(f"   False Negative (Missed the Gain)  : {cm[1][0]}")
    print(f"   True Positive  (Caught the Gain)  : {cm[1][1]}")
    print(f"=======================================================")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}

from sklearn.ensemble import RandomForestClassifier

def evaluate_baselines(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
) -> Dict[str, Any]:

    '''Evaluates two standard baselines:
    1. Naive Model (Blindly guessing the trainign majority class)
    2. Random Forest (Traditional tabular machine learning benchmark)
    '''

    # 1. Baseline 1: Naive Majority Class Guesser

    # Finds whether '1' (Up) '0' (Down) was more common in training
    majority_class = int(np.round(np.mean(y_train)))
    naive_preds = np.full_like(y_test, majority_class)
    naive_acc = accuracy_score(y_test, naive_preds)

    print(f"\n--- Baseline 1: Naive Model (Blindly guessing'{'UP' if majority_class==1 else 'DOWN'}')---")
    print(f"Naive Test Accuracy: {naive_acc * 100:.2f}%")

    # 2. Baseline2: Random Forest (Flattening 3D arrays to 2D)

    # Shape changes from (samples, 30, 7) -> (samples, 210)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train_flat, y_train)
    rf_preds = rf.predict(X_test_flat)
    rf_acc = accuracy_score(y_test, rf_preds)

    print(f"\n--- Baseline 2: Random Forest (Tabular ML) ---")
    print(f"Random Forest Test Accuracy: {rf_acc * 100:.2f}%")
    print(f"=======================================================")

    return {'naive_acc': naive_acc, 'rf_acc' : rf_acc}



def simulate_trading_strategy(
    actual_returns: np.ndarray, 
    predicted_probs: np.ndarray, 
    transaction_cost: float = 0.001
) -> Dict[str, float]:
    """
    Simulates a long/cash algorithmic trading strategy with transaction costs
    and computes the annualized Sharpe Ratio.
    """
    # Generate trading signals (1 = Invested, 0 = In Cash)
    signals = (predicted_probs.flatten() > 0.5).astype(int)
    
    strategy_returns = []
    current_position = 0
    
    for t in range(len(actual_returns)):
        target_position = signals[t]
        
        # Deduct 0.1% fee if switching from Cash -> Stock or Stock -> Cash
        cost = transaction_cost if target_position != current_position else 0.0
        
        # Strategy Return = (Position * Market Return) - Fee
        daily_ret = (target_position * actual_returns[t]) - cost
        strategy_returns.append(daily_ret)
        current_position = target_position
        
    strat_ret = np.array(strategy_returns)
    
    # Cumulative compound growth over the test period
    cum_strategy = np.exp(np.cumsum(strat_ret)) - 1.0
    cum_benchmark = np.exp(np.cumsum(actual_returns)) - 1.0
    
    # Annualized Sharpe Ratio (252 trading days per year)
    mean_ret = np.mean(strat_ret) * 252
    std_ret = np.std(strat_ret) * np.sqrt(252) + 1e-9
    sharpe_ratio = mean_ret / std_ret
    
    print(f"\n=======================================================")
    print(f"💰 Out-of-Sample Trading Simulation (251 Test Days)")
    print(f"=======================================================")
    print(f"Strategy Total Return   : {cum_strategy[-1] * 100:.2f}%")
    print(f"Buy & Hold Total Return : {cum_benchmark[-1] * 100:.2f}%")
    print(f"Strategy Sharpe Ratio   : {sharpe_ratio:.2f}")
    print(f"=======================================================")
    
    return {
        'total_strategy_return': cum_strategy[-1],
        'total_benchmark_return': cum_benchmark[-1],
        'sharpe_ratio': sharpe_ratio
    }

if __name__ == "__main__":
    import os
    import argparse
    import tensorflow as tf
    from data_loader import fetch_data
    from features import calculate_features
    from preprocess import chronological_split, fit_and_transform_scalers, create_sliding_windows
    
    parser = argparse.ArgumentParser(description="Evaluate Model on any stock")
    parser.add_argument("--ticker", type=str, default="^NSEI", help="Stock ticker symbol")
    args = parser.parse_args()
    
    ticker = args.ticker
    clean_ticker = ticker.replace("^", "").replace(".", "_")
    
    # 1. Ingest Data & Prepare Out-of-Sample Test Set
    raw_df = fetch_data(ticker=ticker, start_date="2015-01-01", end_date="2026-08-01")
    df = calculate_features(raw_df)
    feature_cols = ['Log_Return', 'HL_Spread', 'OC_Spread', 'Volume_ROC', 'SMA_20_Ratio', 'RSI_14_Scaled', 'MACD']
    
    train_df, val_df, test_df = chronological_split(df, train_ratio=0.8, val_ratio=0.1)
    train_scaled, val_scaled, test_scaled, scaler = fit_and_transform_scalers(train_df, val_df, test_df, feature_cols)
    
    X_train, y_train = create_sliding_windows(train_scaled, train_df['Target_Direction'], window_size=30)
    X_test, y_test = create_sliding_windows(test_scaled, test_df['Target_Direction'], window_size=30)
    
    # 2. Load the Best Trained LSTM Model from models/
    model_path = f"models/best_lstm_model_{clean_ticker}.keras"
    if not os.path.exists(model_path):
        model_path = "models/best_lstm_model.keras"  # fallback if saved without ticker name
        
    print(f"\n--> Loading trained model checkpoint: '{model_path}'...")
    lstm_model = tf.keras.models.load_model(model_path)
    
    # 3. Evaluate LSTM Classification Metrics
    lstm_probs = lstm_model.predict(X_test)
    lstm_metrics = evaluate_predictions(y_test, lstm_probs, model_name=f"Stacked LSTM ({ticker})")
    
    # 4. Evaluate Naive & Random Forest Baselines
    baseline_metrics = evaluate_baselines(X_train, y_train, X_test, y_test)
    
    # 5. Run the Algorithmic Trading Simulator
    actual_test_returns = test_df['Target_Return'].values[29 : len(test_df) - 1]
    trading_results = simulate_trading_strategy(actual_test_returns, lstm_probs)

