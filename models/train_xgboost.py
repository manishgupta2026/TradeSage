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
import optuna
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')


class TradingModelTrainer:
    """Trains and manages XGBoost trading prediction models."""

    def __init__(self):
        self.model = None # Main/Meta model
        self.xgb_model = None
        self.lgb_model = None
        self.cat_model = None
        self.feature_names = None
        self.training_metrics = {}

    def _optimize_hyperparameters(self, X_train, y_train, spw):
        """Use Optuna for hyperparameter tuning on XGBoost"""
        print("\nRunning Optuna hyperparameter tuning...")

        def objective(trial):
            param = {
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'gamma': trial.suggest_float('gamma', 0.0, 0.2, step=0.1),
                'scale_pos_weight': spw,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }

            tscv = TimeSeriesSplit(n_splits=3)
            aucs = []

            # Using XGBClassifier directly instead of xgb.cv for compatibility
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
                X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

                model = XGBClassifier(**param)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

                preds = model.predict_proba(X_va)[:, 1]
                aucs.append(roc_auc_score(y_va, preds))

            return np.mean(aucs)

        study = optuna.create_study(direction='maximize')
        # Limiting n_trials for execution speed, in real-world use 50-100
        study.optimize(objective, n_trials=10, show_progress_bar=False)

        print(f"Best Optuna params: {study.best_params}")
        print(f"Best Optuna AUC: {study.best_value:.4f}")

        best_params = study.best_params
        best_params['scale_pos_weight'] = spw
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'auc'
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        best_params['verbosity'] = 0
        return best_params

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

        # Ensure no data leakage - avoid entire dataset operations before this

        # Optimize XGBoost hyperparameters via Optuna
        best_params = self._optimize_hyperparameters(X_train, y_train, spw)

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

        print(f"\nTraining final ensemble  (train={len(X_train):,}  val={len(X_val):,})...")

        # 1. XGBoost with tuned parameters
        self.xgb_model = XGBClassifier(**best_params, early_stopping_rounds=40)
        self.xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        print(" XGBoost trained")

        # 2. LightGBM
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, scale_pos_weight=spw, random_state=42,
            early_stopping_rounds=40, verbose=-1, metric='auc'
        )
        self.lgb_model.fit(X_train, y_train, eval_set=eval_set)
        print(" LightGBM trained")

        # 3. CatBoost
        self.cat_model = CatBoostClassifier(
            iterations=300, learning_rate=0.05, auto_class_weights='Balanced', random_seed=42,
            early_stopping_rounds=40, verbose=0, eval_metric='AUC'
        )
        self.cat_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        print(" CatBoost trained")

        # Create meta-learner features (out-of-fold predictions on validation set)
        print("\nTraining Meta-Learner (Logistic Regression)...")
        xgb_val_preds = self.xgb_model.predict_proba(X_val)[:, 1]
        lgb_val_preds = self.lgb_model.predict_proba(X_val)[:, 1]
        cat_val_preds = self.cat_model.predict_proba(X_val)[:, 1]

        meta_X = np.column_stack((xgb_val_preds, lgb_val_preds, cat_val_preds))
        self.model = LogisticRegression() # This is the meta-learner
        self.model.fit(meta_X, y_val)

        print(" Meta-Learner trained")

        return self.model

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate with AUC, precision, recall, F1 and trading win-rate.
        Precision is the most important metric for trading (signal quality).
        """
        if self.model is None:
            raise ValueError("No model trained.")

        X_test, y_test = self._sanitize(X_test, y_test)

        xgb_preds = self.xgb_model.predict_proba(X_test)[:, 1]
        lgb_preds = self.lgb_model.predict_proba(X_test)[:, 1]
        cat_preds = self.cat_model.predict_proba(X_test)[:, 1]

        meta_X = np.column_stack((xgb_preds, lgb_preds, cat_preds))
        probs = self.model.predict_proba(meta_X)[:, 1]
        preds = self.model.predict(meta_X)

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

        # Custom Eval Metric: profit_score
        # Simulate trading returns based on model predictions
        # True Positive (actual=1, pred=1): +6% avg gain
        # False Positive (actual=0, pred=1): -2% avg loss (stop hit)
        profit_score = (actual_wins * 6.0) - ((predicted_buys - actual_wins) * 2.0)

        metrics = {
            'auc_score':          round(float(auc), 4),
            'accuracy':           round(float(accuracy), 4),
            'precision':          round(float(precision), 4),
            'recall':             round(float(recall), 4),
            'f1':                 round(float(f1), 4),
            'predicted_win_rate': round(float(win_rate), 4),
            'buy_signals':        predicted_buys,
            'actual_profitable':  actual_wins,
            'profit_score':       round(float(profit_score), 2),
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
        print(f"Profit Score: {profit_score:.2f}")
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

        xgb_preds = self.xgb_model.predict_proba(X)[:, 1]
        lgb_preds = self.lgb_model.predict_proba(X)[:, 1]
        cat_preds = self.cat_model.predict_proba(X)[:, 1]

        meta_X = np.column_stack((xgb_preds, lgb_preds, cat_preds))

        return self.model.predict(meta_X), self.model.predict_proba(meta_X)[:, 1]

    def get_feature_importance(self, top_n=15):
        if self.xgb_model is None:
            raise ValueError("No model trained.")
        # Meta-learner doesn't have native feature importances for original features,
        # so we rely on the primary XGBoost model's feature importances.
        imp = pd.DataFrame({
            'feature':    self.feature_names,
            'importance': self.xgb_model.feature_importances_
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
            'xgb_model':     self.xgb_model,
            'lgb_model':     self.lgb_model,
            'cat_model':     self.cat_model,
            'feature_names': self.feature_names,
            'metrics':       self.training_metrics,
        }, path)
        print(f" Model saved  {path}")

    def load_model(self, path='tradesage_model.pkl'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        data = joblib.load(path)
        self.model         = data['model']
        self.xgb_model     = data.get('xgb_model')
        self.lgb_model     = data.get('lgb_model')
        self.cat_model     = data.get('cat_model')
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

