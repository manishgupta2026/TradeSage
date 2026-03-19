"""
TradeSage Model Training Module
XGBoost with TimeSeriesSplit CV, tuned hyperparameters for AUC 0.75+
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


class TradingModelTrainer:
    """Trains and manages XGBoost trading prediction models."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.training_metrics = {}

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost with TimeSeriesSplit CV for robust evaluation.
        Uses conservative hyperparameters to avoid overfitting on noisy market data.
        """
        self.feature_names = list(X_train.columns)

        # Sanitize inputs
        X_train, y_train = self._sanitize(X_train, y_train)

        print(f"\n{'='*60}")
        print("TRADESAGE MODEL TRAINING")
        print(f"{'='*60}")
        print(f"Total training samples : {len(X_train):,}")
        print(f"Features               : {len(self.feature_names)}")
        pos_rate = y_train.mean() * 100
        print(f"Positive rate          : {pos_rate:.1f}%")

        # With balanced target (~40-50% positive), scale_pos_weight  1
        # Only boost if still imbalanced
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = max(0.5, min(3.0, neg / max(1, pos)))
        print(f"scale_pos_weight       : {spw:.2f}")

        # ---- TimeSeriesSplit CV to pick best n_estimators ---- #
        print("\nRunning 3-fold TimeSeriesSplit CV...")
        tscv = TimeSeriesSplit(n_splits=3)
        cv_aucs = []

        probe_model = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=0.02,
            n_estimators=1000,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=0.5,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=spw,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=40,
            verbosity=0,
        )

        best_iters = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train), 1):
            Xf_tr, Xf_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            yf_tr, yf_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            probe_model.fit(
                Xf_tr, yf_tr,
                eval_set=[(Xf_va, yf_va)],
                verbose=False
            )
            fold_auc = roc_auc_score(yf_va, probe_model.predict_proba(Xf_va)[:, 1])
            best_iters.append(probe_model.best_iteration)
            cv_aucs.append(fold_auc)
            print(f"  Fold {fold}: AUC={fold_auc:.4f}  best_iter={probe_model.best_iteration}")

        mean_auc = np.mean(cv_aucs)
        best_n   = max(200, int(np.mean(best_iters)) + 50)   # minimum 200 trees
        print(f"\nCV mean AUC: {mean_auc:.4f}  ->  using n_estimators={best_n}")

        # ---- Final model on full training data ---- #
        if X_val is not None and y_val is not None:
            X_val, y_val = self._sanitize(X_val, y_val)
            eval_set = [(X_val, y_val)]
        else:
            # Hold out last 20% as validation
            split = int(len(X_train) * 0.8)
            X_val  = X_train.iloc[split:]
            y_val  = y_train.iloc[split:]
            X_train = X_train.iloc[:split]
            y_train = y_train.iloc[:split]
            eval_set = [(X_val, y_val)]

        print(f"\nTraining final model  (train={len(X_train):,}  val={len(X_val):,})...")

        self.model = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=0.02,
            n_estimators=best_n,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=0.5,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=spw,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=40,
            verbosity=0,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        final_iter = self.model.best_iteration
        print(f" Training complete  best iteration: {final_iter}")
        return self.model

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate with AUC, precision, recall, F1 and trading win-rate.
        Precision is the most important metric for trading (signal quality).
        """
        if self.model is None:
            raise ValueError("No model trained.")

        X_test, y_test = self._sanitize(X_test, y_test)

        probs = self.model.predict_proba(X_test)[:, 1]
        preds = self.model.predict(X_test)

        auc       = roc_auc_score(y_test, probs)
        accuracy  = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall    = recall_score(y_test, preds, zero_division=0)
        f1        = f1_score(y_test, preds, zero_division=0)
        cm        = confusion_matrix(y_test, preds)

        predicted_buys = int(preds.sum())
        y_arr = y_test.values if hasattr(y_test, 'values') else y_test
        actual_wins = int(y_arr[preds == 1].sum()) if predicted_buys > 0 else 0
        win_rate = actual_wins / predicted_buys if predicted_buys > 0 else 0

        metrics = {
            'auc_score':          round(float(auc), 4),
            'accuracy':           round(float(accuracy), 4),
            'precision':          round(float(precision), 4),
            'recall':             round(float(recall), 4),
            'f1':                 round(float(f1), 4),
            'predicted_win_rate': round(float(win_rate), 4),
            'buy_signals':        predicted_buys,
            'actual_profitable':  actual_wins,
            'confusion_matrix':   cm,
        }
        self.training_metrics = metrics

        print(f"\n{'='*60}")
        print("MODEL EVALUATION")
        print(f"{'='*60}")
        print(f"ROC AUC    : {auc:.4f}")
        print(f"Accuracy   : {accuracy:.4f}")
        print(f"Precision  : {precision:.4f}   signal quality (key for trading)")
        print(f"Recall     : {recall:.4f}")
        print(f"F1         : {f1:.4f}")
        print(f"Win Rate   : {win_rate*100:.1f}%  ({actual_wins}/{predicted_buys} signals)")
        print(f"\nConfusion Matrix:")
        print(f"               Predicted 0   Predicted 1")
        print(f"Actual 0       {cm[0][0]:8d}      {cm[0][1]:8d}")
        print(f"Actual 1       {cm[1][0]:8d}      {cm[1][1]:8d}")

        if auc >= 0.75:
            print("\n Excellent  strong predictive power")
        elif auc >= 0.65:
            print("\n Good  useful predictive power")
        elif auc >= 0.60:
            print("\n- Moderate  some predictive power")
        else:
            print("\n Weak  limited predictive power")

        return metrics

    def predict(self, X):
        if self.model is None:
            raise ValueError("No model loaded.")
        if self.feature_names:
            for col in set(self.feature_names) - set(X.columns):
                X[col] = 0
            X = X[self.feature_names]
        X, _ = self._sanitize(X)
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, top_n=15):
        if self.model is None:
            raise ValueError("No model trained.")
        imp = pd.DataFrame({
            'feature':    self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n{'='*60}")
        print(f"TOP {top_n} FEATURES")
        print(f"{'='*60}")
        for i, (_, row) in enumerate(imp.head(top_n).iterrows(), 1):
            bar = '-' * int(row['importance'] * 300)
            print(f"{i:2d}. {row['feature']:<30s} {row['importance']:.4f} {bar}")
        return imp

    def save_model(self, path='tradesage_model.pkl'):
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        joblib.dump({
            'model':         self.model,
            'feature_names': self.feature_names,
            'metrics':       self.training_metrics,
        }, path)
        print(f" Model saved  {path}")

    def load_model(self, path='tradesage_model.pkl'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        data = joblib.load(path)
        self.model         = data['model']
        self.feature_names = data.get('feature_names')
        self.training_metrics = data.get('metrics', {})
        print(f" Model loaded  {path}")
        if self.feature_names:
            print(f"  Features : {len(self.feature_names)}")
        auc = self.training_metrics.get('auc_score', 'N/A')
        print(f"  Saved AUC: {auc}")

    # ------------------------------------------------------------------ #
    #  INTERNAL                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sanitize(X, y=None):
        X = X.copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        if y is not None:
            return X, y
        return X, None

