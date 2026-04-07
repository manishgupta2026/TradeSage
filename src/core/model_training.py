"""
TradeSage Model Training Module v2
XGBoost + LightGBM + CatBoost ensemble with Optuna tuning
PurgedTimeSeriesSplit CV + probability calibration
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.isotonic import IsotonicRegression
import joblib
import os
import warnings

warnings.filterwarnings('ignore')


class PurgedTimeSeriesSplit:
    """
    TimeSeriesSplit with a gap between train and validation to prevent leakage.
    Gap of `forward_days` eliminates label overlap between folds.
    """
    def __init__(self, n_splits=3, gap=10):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X):
        n = len(X)
        fold = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end  = (i + 1) * fold
            val_start  = train_end + self.gap
            val_end    = val_start + fold
            yield (
                np.arange(0, train_end),
                np.arange(val_start, min(val_end, n))
            )


def profit_score(y_true, y_pred_proba, threshold=0.6):
    """Custom profit-focused metric for model evaluation."""
    preds = (y_pred_proba >= threshold).astype(int)
    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    total_signals = tp + fp
    if total_signals == 0:
        return 0.0
    # Simulated trading: avg 5% gain on TP, avg 2.5% loss on FP
    profit = tp * 0.05 - fp * 0.025
    return profit / total_signals


class TradingModelTrainer:
    """Trains and manages XGBoost (+ optional ensemble) trading prediction models."""

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.training_metrics = {}
        self.calibrator = None
        self.ensemble_models = None  # For ensemble mode

    def train_model(self, X_train, y_train, X_val=None, y_val=None, use_ensemble=False):
        """
        Train XGBoost with PurgedTimeSeriesSplit CV + Optuna tuning.
        Optionally train an ensemble with LightGBM and CatBoost.
        """
        self.feature_names = list(X_train.columns)

        # Sanitize inputs
        X_train, y_train = self._sanitize(X_train, y_train)

        print(f"\n{'='*60}")
        print("TRADESAGE MODEL TRAINING v2")
        print(f"{'='*60}")
        print(f"Total training samples : {len(X_train):,}")
        print(f"Features               : {len(self.feature_names)}")
        pos_rate = y_train.mean() * 100
        print(f"Positive rate          : {pos_rate:.1f}%")
        print(f"Ensemble mode          : {'ON' if use_ensemble else 'OFF'}")

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = max(0.5, min(3.0, neg / max(1, pos)))
        print(f"scale_pos_weight       : {spw:.2f}")

        # ---- Optuna hyperparameter tuning ---- #
        best_params = self._optuna_tune(X_train, y_train, spw)

        # ---- Prepare val set ---- #
        if X_val is not None and y_val is not None:
            X_val, y_val = self._sanitize(X_val, y_val)
        else:
            split = int(len(X_train) * 0.8)
            X_val  = X_train.iloc[split:]
            y_val  = y_train.iloc[split:]
            X_train = X_train.iloc[:split]
            y_train = y_train.iloc[:split]

        print(f"\nTraining final model  (train={len(X_train):,}  val={len(X_val):,})...")

        # ---- Train primary XGBoost ---- #
        self.model = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=spw,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1,
            device='cuda',
            tree_method='hist',
            verbosity=0,
        )
        self.model.fit(X_train, y_train, verbose=False)
        print(f"  XGBoost training complete  n_estimators={best_params['n_estimators']}")

        # ---- Ensemble (optional) ---- #
        if use_ensemble:
            self._train_ensemble(X_train, y_train, X_val, y_val, spw, best_params)

        # ---- Isotonic calibration ---- #
        print("  Calibrating probabilities (isotonic)...")
        raw_val_probs = self._raw_predict(X_val)
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(raw_val_probs, y_val)
        print("  Calibration complete")

        return self.model

    def _optuna_tune(self, X_train, y_train, spw):
        """Hyperparameter tuning with Optuna."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("  Optuna not installed, using default hyperparameters")
            return self._default_params()

        # Subsample for faster tuning
        CV_MAX_ROWS = 400_000
        if len(X_train) > CV_MAX_ROWS:
            X_cv = X_train.iloc[-CV_MAX_ROWS:]
            y_cv = y_train.iloc[-CV_MAX_ROWS:]
            print(f"\nRunning Optuna tuning (subsample: {CV_MAX_ROWS:,} most-recent rows)...")
        else:
            X_cv, y_cv = X_train, y_train
            print("\nRunning Optuna tuning...")

        tscv = PurgedTimeSeriesSplit(n_splits=3, gap=10)

        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            }

            cv_scores = []
            for tr_idx, va_idx in tscv.split(X_cv):
                Xf_tr, Xf_va = X_cv.iloc[tr_idx], X_cv.iloc[va_idx]
                yf_tr, yf_va = y_cv.iloc[tr_idx], y_cv.iloc[va_idx]

                model = xgb.XGBClassifier(
                    **params,
                    scale_pos_weight=spw,
                    objective='binary:logistic',
                    eval_metric='auc',
                    random_state=42,
                    n_jobs=-1,
                    device='cuda',
                    tree_method='hist',
                    verbosity=0,
                )
                model.fit(Xf_tr, yf_tr, verbose=False)
                probs = model.predict_proba(Xf_va)[:, 1]
                preds = (probs >= 0.5).astype(int)

                auc = roc_auc_score(yf_va, probs)
                prec = precision_score(yf_va, preds, zero_division=0)
                ps = profit_score(yf_va, probs)

                # Combined: 50% AUC + 25% precision + 25% profit_score
                score = 0.5 * auc + 0.25 * prec + 0.25 * (ps + 1) / 2
                cv_scores.append(score)

            return np.mean(cv_scores)

        N_TRIALS = 60  # Reduced from 50 for faster turnaround; can increase
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        best = study.best_params
        print(f"\n  Optuna best trial ({N_TRIALS} trials):")
        print(f"    Score: {study.best_value:.4f}")
        print(f"    max_depth={best['max_depth']}, lr={best['learning_rate']:.4f}, "
              f"n_est={best['n_estimators']}, subsample={best['subsample']:.2f}")

        return best

    def _default_params(self):
        """Fallback hyperparameters if Optuna is unavailable."""
        return {
            'max_depth': 4,
            'learning_rate': 0.02,
            'n_estimators': 400,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,
            'gamma': 0.5,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
        }

    def _train_ensemble(self, X_train, y_train, X_val, y_val, spw, xgb_params):
        """Train LightGBM + CatBoost + meta-learner."""
        self.ensemble_models = {}

        # LightGBM
        try:
            import lightgbm as lgb
            print("  Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                max_depth=xgb_params.get('max_depth', 4),
                learning_rate=xgb_params.get('learning_rate', 0.02),
                n_estimators=xgb_params.get('n_estimators', 400),
                subsample=xgb_params.get('subsample', 0.7),
                colsample_bytree=xgb_params.get('colsample_bytree', 0.7),
                min_child_weight=xgb_params.get('min_child_weight', 5),
                scale_pos_weight=spw,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
            lgb_model.fit(X_train, y_train)
            self.ensemble_models['lgbm'] = lgb_model
            lgb_auc = roc_auc_score(y_val, lgb_model.predict_proba(X_val)[:, 1])
            print(f"    LightGBM val AUC: {lgb_auc:.4f}")
        except ImportError:
            print("  LightGBM not available, skipping")

        # CatBoost
        try:
            from catboost import CatBoostClassifier
            print("  Training CatBoost...")
            cb_model = CatBoostClassifier(
                depth=min(xgb_params.get('max_depth', 4), 6),
                learning_rate=xgb_params.get('learning_rate', 0.02),
                iterations=xgb_params.get('n_estimators', 400),
                auto_class_weights='Balanced',
                random_seed=42,
                verbose=0,
            )
            cb_model.fit(X_train, y_train)
            self.ensemble_models['catboost'] = cb_model
            cb_auc = roc_auc_score(y_val, cb_model.predict_proba(X_val)[:, 1])
            print(f"    CatBoost val AUC: {cb_auc:.4f}")
        except ImportError:
            print("  CatBoost not available, skipping")

        # Meta-learner (LogisticRegression on base model outputs)
        if self.ensemble_models:
            from sklearn.linear_model import LogisticRegression
            print("  Training meta-learner...")
            meta_features = self._get_ensemble_features(X_val)
            self.meta_learner = LogisticRegression(max_iter=1000, random_state=42)
            self.meta_learner.fit(meta_features, y_val)
            meta_probs = self.meta_learner.predict_proba(meta_features)[:, 1]
            meta_auc = roc_auc_score(y_val, meta_probs)
            print(f"    Meta-learner val AUC: {meta_auc:.4f}")
        else:
            self.meta_learner = None

    def _get_ensemble_features(self, X):
        """Get stacked predictions from all base models."""
        features = [self.model.predict_proba(X)[:, 1]]
        for name, model in (self.ensemble_models or {}).items():
            features.append(model.predict_proba(X)[:, 1])
        return np.column_stack(features)

    def _raw_predict(self, X):
        """Get raw (uncalibrated) predictions, optionally using ensemble."""
        if self.ensemble_models and hasattr(self, 'meta_learner') and self.meta_learner is not None:
            meta_features = self._get_ensemble_features(X)
            return self.meta_learner.predict_proba(meta_features)[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def predict_proba_calibrated(self, X):
        """Predict probabilities and calibrate them."""
        X, _ = self._sanitize(X)
        if self.feature_names:
            for col in set(self.feature_names) - set(X.columns):
                X[col] = 0
            X = X[self.feature_names]

        raw_probs = self._raw_predict(X)
        if self.calibrator is not None:
            return self.calibrator.predict(raw_probs)
        return raw_probs

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate with AUC, precision, recall, F1, win-rate, and profit_score.
        """
        if self.model is None:
            raise ValueError("No model trained.")

        X_test, y_test = self._sanitize(X_test, y_test)

        probs = self.predict_proba_calibrated(X_test)
        preds = (probs >= 0.5).astype(int)

        auc       = roc_auc_score(y_test, probs)
        accuracy  = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall    = recall_score(y_test, preds, zero_division=0)
        f1        = f1_score(y_test, preds, zero_division=0)
        cm        = confusion_matrix(y_test, preds)
        ps        = profit_score(y_test, probs)

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
            'profit_score':       round(float(ps), 4),
            'predicted_win_rate': round(float(win_rate), 4),
            'buy_signals':        predicted_buys,
            'actual_profitable':  actual_wins,
            'confusion_matrix':   cm,
        }
        self.training_metrics = metrics

        print(f"\n{'='*60}")
        print("MODEL EVALUATION")
        print(f"{'='*60}")
        print(f"ROC AUC      : {auc:.4f}")
        print(f"Accuracy     : {accuracy:.4f}")
        print(f"Precision    : {precision:.4f}   signal quality (key for trading)")
        print(f"Recall       : {recall:.4f}")
        print(f"F1           : {f1:.4f}")
        print(f"Profit Score : {ps:.4f}")
        print(f"Win Rate     : {win_rate*100:.1f}%  ({actual_wins}/{predicted_buys} signals)")
        print(f"\nConfusion Matrix:")
        print(f"               Predicted 0   Predicted 1")
        print(f"Actual 0       {cm[0][0]:8d}      {cm[0][1]:8d}")
        print(f"Actual 1       {cm[1][0]:8d}      {cm[1][1]:8d}")

        if auc >= 0.75:
            print(f"\n★ Excellent — strong predictive power")
        elif auc >= 0.65:
            print(f"\n● Good — useful predictive power")
        elif auc >= 0.60:
            print(f"\n○ Moderate — some predictive power")
        else:
            print(f"\n✗ Weak — limited predictive power")

        return metrics

    def predict(self, X):
        if self.model is None:
            raise ValueError("No model loaded.")
        probs = self.predict_proba_calibrated(X)
        preds = (probs >= 0.5).astype(int)
        return preds, probs

    def get_feature_importance(self, top_n=15):
        if self.model is None:
            raise ValueError("No model trained.")
        base = getattr(self.model, 'estimator', self.model)
        imp = pd.DataFrame({
            'feature':    self.feature_names,
            'importance': base.feature_importances_
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
        save_data = {
            'model':           self.model,
            'feature_names':   self.feature_names,
            'metrics':         self.training_metrics,
            'calibrator':      self.calibrator,
        }
        # Save ensemble if present
        if self.ensemble_models:
            save_data['ensemble_models'] = self.ensemble_models
        if hasattr(self, 'meta_learner') and self.meta_learner is not None:
            save_data['meta_learner'] = self.meta_learner

        joblib.dump(save_data, path)
        print(f"  Model saved → {path}")

    def load_model(self, path='tradesage_model.pkl'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        data = joblib.load(path)
        self.model            = data['model']
        self.feature_names    = data.get('feature_names')
        self.training_metrics = data.get('metrics', {})
        self.calibrator       = data.get('calibrator')
        self.ensemble_models  = data.get('ensemble_models')
        if hasattr(data, 'get'):
            meta = data.get('meta_learner')
            if meta is not None:
                self.meta_learner = meta

        print(f"  Model loaded → {path}")
        if self.feature_names:
            print(f"  Features : {len(self.feature_names)}")
        auc = self.training_metrics.get('auc_score', 'N/A')
        print(f"  Saved AUC: {auc}")
        if self.ensemble_models:
            print(f"  Ensemble : {list(self.ensemble_models.keys())}")

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
