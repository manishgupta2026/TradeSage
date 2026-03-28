import pandas as pd
import numpy as np
from features.technical_indicators import FeatureEngineer
from models.train_xgboost import TradingModelTrainer

def run_tests():
    print("Testing Feature Engineering...")
    # 1. Generate Dummy Data
    dates = pd.date_range('2023-01-01', periods=500, freq='B')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 1, 500))
    df = pd.DataFrame({
        'open': prices + np.random.normal(0, 0.5, 500),
        'high': prices + abs(np.random.normal(0, 1, 500)),
        'low': prices - abs(np.random.normal(0, 1, 500)),
        'close': prices,
        'volume': np.random.randint(10000, 100000, 500)
    }, index=dates)

    # 2. Test Feature Engineer
    fe = FeatureEngineer()
    df_feat = fe.add_technical_indicators(df)

    # Assert newly added features are present
    assert 'vpt' in df_feat.columns
    assert 'stoch_rsi' in df_feat.columns
    assert 'atr_pct' in df_feat.columns
    assert 'rolling_high_20' in df_feat.columns
    assert 'rsi_volume_conviction' in df_feat.columns

    print("Features generated successfully.")
    print(f"Total features: {len(df_feat.columns)}")

    df_target = fe.create_target_variable(df_feat)

    # We will slice to remove NaNs for training
    df_train = df_target.dropna()

    X, y, cols = fe.prepare_training_data(df_train)

    split_idx = int(len(X) * 0.8)
    X_tr, y_tr = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]

    print("\nTesting Model Training Pipeline (Ensemble + Tuning + Custom Metric)...")
    trainer = TradingModelTrainer()
    trainer.train_model(X_tr, y_tr, X_val, y_val)

    print("\nTesting Prediction...")
    metrics = trainer.evaluate_model(X_val, y_val)
    print(metrics)

if __name__ == "__main__":
    run_tests()
