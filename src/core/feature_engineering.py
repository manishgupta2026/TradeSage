"""
TradeSage - Feature Engineering v2
Normalized, derived features + regime flags for XGBoost AUC 0.75+
"""

import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Generate technical indicators and ML-ready features."""

    def __init__(self):
        pass

    def add_technical_indicators(self, df):
        """
        Full feature pipeline: raw indicators → normalized derived features → regime flags.
        Returns DataFrame with 70+ ML-ready features.
        """
        df = self._clean_df(df)
        df = self._add_moving_averages(df)
        df = self._add_momentum(df)
        df = self._add_volatility(df)
        df = self._add_volume_features(df)
        df = self._add_derived_features(df)
        df = self._add_regime_features(df)
        df = self._add_candle_features(df)
        return df

    # ------------------------------------------------------------------ #
    #  PRIVATE HELPERS                                                     #
    # ------------------------------------------------------------------ #

    def _clean_df(self, df):
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        df.columns = [str(c).lower().strip() for c in df.columns]
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df

    def _add_moving_averages(self, df):
        for w in [5, 10, 20, 50, 100]:   # 100 instead of 200 — saves 100 warmup rows
            df[f'sma_{w}'] = df['close'].rolling(w).mean()
        # sma_200 alias → use sma_100 so derived features still work
        df['sma_200'] = df['close'].rolling(100).mean()
        for s in [9, 21, 50]:
            df[f'ema_{s}'] = df['close'].ewm(span=s, adjust=False).mean()
        return df

    def _add_momentum(self, df):
        df['rsi_14'] = momentum.rsi(df['close'], window=14)
        df['rsi_7']  = momentum.rsi(df['close'], window=7)
        df['stoch_k'] = momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = momentum.stoch_signal(df['high'], df['low'], df['close'])
        df['williams_r'] = momentum.williams_r(df['high'], df['low'], df['close'])

        # MACD components (not just diff)
        macd_ind = trend.MACD(df['close'])
        df['macd_line']   = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        df['macd_hist']   = macd_ind.macd_diff()

        # ADX + directional indicators
        adx_ind = trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx']    = adx_ind.adx()
        df['adx_pos'] = adx_ind.adx_pos()
        df['adx_neg'] = adx_ind.adx_neg()

        # Price momentum
        for d in [1, 3, 5, 10, 20]:
            df[f'price_change_{d}d'] = df['close'].pct_change(d)

        return df

    def _add_volatility(self, df):
        bb = volatility.BollingerBands(df['close'])
        df['bb_high']  = bb.bollinger_hband()
        df['bb_mid']   = bb.bollinger_mavg()
        df['bb_low']   = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_pct']   = bb.bollinger_pband()   # 0=lower, 1=upper

        kc = volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_high'] = kc.keltner_channel_hband()
        df['kc_low']  = kc.keltner_channel_lband()
        df['squeeze_on'] = (
            (df['bb_low'] > df['kc_low']) & (df['bb_high'] < df['kc_high'])
        ).astype(int)

        df['atr'] = volatility.average_true_range(df['high'], df['low'], df['close'])
        df['atr_pct'] = df['atr'] / df['close']   # normalized ATR

        df['volatility_5d']  = df['close'].pct_change().rolling(5).std()
        df['volatility_20d'] = df['close'].pct_change().rolling(20).std()

        return df

    def _add_volume_features(self, df):
        df['obv'] = volume.on_balance_volume(df['close'], df['volume'])
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio']  = df['volume'] / (df['volume_sma_20'] + 1e-10)
        df['volume_change'] = df['volume'].pct_change()
        return df

    def _add_derived_features(self, df):
        """
        Normalized features — the core improvement.
        Raw prices (ema_9=245) are meaningless; (close-ema_9)/ema_9 is not.
        """
        eps = 1e-10

        # Distance from MAs (normalized)
        df['dist_ema9']   = (df['close'] - df['ema_9'])   / (df['ema_9']   + eps)
        df['dist_ema21']  = (df['close'] - df['ema_21'])  / (df['ema_21']  + eps)
        df['dist_ema50']  = (df['close'] - df['ema_50'])  / (df['ema_50']  + eps)
        df['dist_sma20']  = (df['close'] - df['sma_20'])  / (df['sma_20']  + eps)
        df['dist_sma50']  = (df['close'] - df['sma_50'])  / (df['sma_50']  + eps)
        df['dist_sma200'] = (df['close'] - df['sma_200']) / (df['sma_200'] + eps)

        # MA slopes (momentum of the MA itself)
        df['ema9_slope']  = df['ema_9'].pct_change(3)
        df['ema21_slope'] = df['ema_21'].pct_change(5)
        df['sma50_slope'] = df['sma_50'].pct_change(10)

        # MA alignment score: how many MAs is price above?
        df['ma_alignment'] = (
            (df['close'] > df['ema_9']).astype(int) +
            (df['close'] > df['ema_21']).astype(int) +
            (df['close'] > df['sma_50']).astype(int) +
            (df['close'] > df['sma_200']).astype(int)
        )

        # Candle geometry
        df['hl_range']       = (df['high'] - df['low']) / (df['close'] + eps)
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + eps)
        df['body_size']      = abs(df['close'] - df['open']) / (df['close'] + eps)
        df['upper_wick']     = (df['high'] - df[['close','open']].max(axis=1)) / (df['close'] + eps)
        df['lower_wick']     = (df[['close','open']].min(axis=1) - df['low']) / (df['close'] + eps)

        # RSI normalized slope
        df['rsi_slope'] = df['rsi_14'].diff(3)
        df['rsi_dist_50'] = df['rsi_14'] - 50   # positive = bullish side

        # MACD normalized by price
        df['macd_norm']    = df['macd_line']   / (df['close'] + eps)
        df['macd_sig_norm'] = df['macd_signal'] / (df['close'] + eps)
        df['macd_hist_norm'] = df['macd_hist']  / (df['close'] + eps)

        # ADX directional balance
        df['di_diff'] = df['adx_pos'] - df['adx_neg']   # positive = bullish

        # Stoch position
        df['stoch_diff'] = df['stoch_k'] - df['stoch_d']

        # OBV slope (normalized)
        df['obv_slope'] = df['obv'].diff(5) / (df['obv'].abs().rolling(5).mean() + eps)

        # Volume-price relationship
        df['vp_trend'] = df['price_change_1d'] * df['volume_ratio']

        # 52-week high/low distance (use 200 bars to match SMA warmup)
        df['dist_52w_high'] = (df['close'] - df['high'].rolling(200).max()) / (df['high'].rolling(200).max() + eps)
        df['dist_52w_low']  = (df['close'] - df['low'].rolling(200).min())  / (df['low'].rolling(200).min()  + eps)

        # Consecutive up/down days
        up_days = (df['close'] > df['close'].shift(1)).astype(int)
        df['consec_up'] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumcount() + 1
        df['consec_up'] = df['consec_up'] * up_days  # zero on down days

        return df

    def _add_regime_features(self, df):
        """Binary flags capturing market regime — high signal-to-noise."""
        df['above_ema9']   = (df['close'] > df['ema_9']).astype(int)
        df['above_ema21']  = (df['close'] > df['ema_21']).astype(int)
        df['above_sma50']  = (df['close'] > df['sma_50']).astype(int)
        df['above_sma200'] = (df['close'] > df['sma_200']).astype(int)

        # Full uptrend: price > ema9 > ema21 > sma50
        df['full_uptrend'] = (
            (df['close']  > df['ema_9']) &
            (df['ema_9']  > df['ema_21']) &
            (df['ema_21'] > df['sma_50'])
        ).astype(int)

        # Golden / death cross
        df['golden_cross'] = (df['sma_50'] > df['sma_200']).astype(int)

        # RSI zones
        df['rsi_oversold']   = (df['rsi_14'] < 35).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 65).astype(int)
        df['rsi_mid_bull']   = ((df['rsi_14'] >= 50) & (df['rsi_14'] < 65)).astype(int)

        # Stoch bullish cross (K crosses above D from below 30)
        df['stoch_bullish_cross'] = (
            (df['stoch_k'] > df['stoch_d']) &
            (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)) &
            (df['stoch_k'] < 50)
        ).astype(int)

        # MACD bullish cross
        df['macd_bullish_cross'] = (
            (df['macd_hist'] > 0) &
            (df['macd_hist'].shift(1) <= 0)
        ).astype(int)

        # ADX trend strength
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        df['weak_trend']   = (df['adx'] < 20).astype(int)

        # Volume breakout
        df['vol_breakout'] = (df['volume_ratio'] > 2.0).astype(int)

        # Squeeze release (was squeezed, now expanding)
        df['squeeze_release'] = (
            (df['squeeze_on'].shift(1) == 1) & (df['squeeze_on'] == 0)
        ).astype(int)

        # BB position zones
        df['near_bb_lower'] = (df['bb_pct'] < 0.2).astype(int)
        df['near_bb_upper'] = (df['bb_pct'] > 0.8).astype(int)

        return df

    def _add_candle_features(self, df):
        """ATR-normalized candle patterns."""
        atr = df['atr'].replace(0, np.nan).fillna(df['close'] * 0.02)

        df['body_atr']        = abs(df['close'] - df['open']) / atr
        df['upper_wick_atr']  = (df['high'] - df[['close','open']].max(axis=1)) / atr
        df['lower_wick_atr']  = (df[['close','open']].min(axis=1) - df['low']) / atr

        # Hammer: small body, long lower wick, little upper wick
        df['hammer'] = (
            (df['lower_wick_atr'] > 1.5) &
            (df['body_atr'] < 0.5) &
            (df['upper_wick_atr'] < 0.3)
        ).astype(int)

        # Bullish engulfing
        df['bullish_engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &   # prev red
            (df['close'] > df['open']) &                      # curr green
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)

        return df

    # ------------------------------------------------------------------ #
    #  TARGET + TRAINING DATA                                              #
    # ------------------------------------------------------------------ #

    def create_target_variable(self, df, forward_days=10, tp_mult=2.0, sl_mult=2.0,
                               gain_threshold=0.04):
        """
        Simple forward-return target: 1 if max high in next `forward_days` bars
        is >= `gain_threshold` above today's close.

        Why this beats path-dependent TP/SL:
        - No ATR dependency (ATR is noisy on short histories)
        - Directly learnable: model just needs to predict "will price go up 4%?"
        - Typical positive rate: 30-45% depending on threshold + market regime
        - Removes the SL-hit-first ambiguity that adds noise to the label

        Parameters
        ----------
        forward_days    : int   — look-ahead window (default 10)
        gain_threshold  : float — minimum gain required, e.g. 0.04 = 4%
        tp_mult, sl_mult: kept for API compatibility but unused
        """
        df = df.copy()

        # Max high over next forward_days (shift(-forward_days) aligns future window)
        future_max_high = df['high'].rolling(forward_days).max().shift(-forward_days)
        pct_gain = (future_max_high - df['close']) / (df['close'] + 1e-10)

        df['target'] = (pct_gain >= gain_threshold).astype(float)

        # Last forward_days rows have no complete future window → set NaN → dropped later
        df.loc[df.index[-forward_days:], 'target'] = np.nan

        pos_rate = df['target'].mean() * 100
        print(f"  Target: max_high_{forward_days}d >= {gain_threshold*100:.0f}%  |  "
              f"positive rate = {pos_rate:.1f}%")
        return df

    def prepare_training_data(self, df, target_col='target'):
        """
        Returns X, y, feature_names — with strict leakage prevention.
        """
        df = df.copy()
        df = df.dropna()

        # Strict exclusion list — no raw prices, no future-derived columns
        exclude_cols = {
            'open', 'high', 'low', 'close', 'volume',
            'target', 'forward_return', 'target_return', 'symbol',
            # raw MA values (use normalized dist_ versions instead)
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_9', 'ema_21', 'ema_50',
            # raw BB/KC bands (use bb_pct, bb_width instead)
            'bb_high', 'bb_mid', 'bb_low',
            'kc_high', 'kc_low',
            # raw OBV (use obv_slope instead)
            'obv',
            # raw MACD line/signal (use normalized versions)
            'macd_line', 'macd_signal',
            # raw ATR (use atr_pct instead)
            'atr',
            # raw volume (use volume_ratio instead)
            'volume_sma_20',
        }

        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

        pos_rate = y.mean() * 100
        print(f"Training data: {X.shape[0]:,} rows × {X.shape[1]} features")
        print(f"Positive rate: {pos_rate:.1f}%  (target ~40-50%)")

        return X, y, feature_cols
