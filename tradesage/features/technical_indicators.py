"""
TradeSage - Enhanced Technical Indicators (features/technical_indicators.py)

Comprehensive feature engineering pipeline targeting AUC 0.75+.
Implements 80+ ML-ready features across 5 categories:
  1. Volume-based (VPT, OBV, MFI, A/D Line)
  2. Momentum (MACD, StochRSI, ROC, Williams %R)
  3. Volatility (BB %B, Keltner, ATR%, Historical Vol)
  4. Market Structure (HH/LL, S/R levels, Pivot Points, Fibonacci)
  5. Advanced (Relative Strength, Market Regime, Feature Interactions)
"""

import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume
import warnings

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """
    Generate technical indicators and ML-ready features for swing trading.

    All features are either:
    - Normalized (e.g., distance from MA as % of price) so scale is
      consistent across ₹50 and ₹5000 stocks.
    - Binary flags (0/1) for regime / pattern detection.

    No raw prices are returned to prevent data leakage in the model.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature pipeline. Returns DataFrame with 80+ ML-ready features.

        Call order matters — later stages depend on earlier ones.
        """
        df = self._clean_df(df)
        df = self._add_moving_averages(df)
        df = self._add_momentum(df)
        df = self._add_volatility(df)
        df = self._add_volume_features(df)
        df = self._add_market_structure(df)
        df = self._add_derived_features(df)
        df = self._add_regime_features(df)
        df = self._add_candle_features(df)
        df = self._add_feature_interactions(df)
        return df

    def create_target_variable(
        self,
        df: pd.DataFrame,
        forward_days: int = 10,
        gain_threshold: float = 0.04,
        # legacy params kept for API compatibility
        tp_mult: float = 2.0,
        sl_mult: float = 2.0,
    ) -> pd.DataFrame:
        """
        Forward-return binary target: 1 if max high in next *forward_days*
        bars is >= *gain_threshold* above today's close.

        Why this approach:
        - No ATR dependency (ATR is noisy on short histories)
        - Directly learnable by the model ("will price rise 4%?")
        - Avoids SL-hit-first ambiguity that adds noise to path-dependent labels
        - Typical positive rate: 30-45% depending on market regime

        Parameters
        ----------
        forward_days    : look-ahead window (default 10 trading days)
        gain_threshold  : minimum gain required (0.04 = 4%)
        """
        df = df.copy()

        future_max_high = df["high"].rolling(forward_days).max().shift(-forward_days)
        pct_gain = (future_max_high - df["close"]) / (df["close"] + 1e-10)
        df["target"] = (pct_gain >= gain_threshold).astype(float)

        # Last forward_days rows have no complete future window → NaN → dropped
        df.loc[df.index[-forward_days:], "target"] = np.nan

        pos_rate = df["target"].mean() * 100
        print(
            f"  Target: max_high_{forward_days}d >= {gain_threshold * 100:.0f}%  |  "
            f"positive rate = {pos_rate:.1f}%"
        )
        return df

    def prepare_training_data(
        self, df: pd.DataFrame, target_col: str = "target"
    ):
        """
        Returns (X, y, feature_names) with strict leakage prevention.

        Excluded columns fall into these buckets:
        - Raw OHLCV (scale differs stock-to-stock)
        - Raw indicator values that normalised versions replace
        - The target itself and any forward-looking columns
        """
        df = df.copy().dropna()

        exclude_cols = {
            # Raw OHLCV
            "open", "high", "low", "close", "volume",
            # Target / future-derived
            "target", "forward_return", "target_return",
            # Metadata
            "symbol",
            # Raw MAs (use normalised dist_ versions)
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
            "ema_9", "ema_21", "ema_50",
            # Raw BB/KC bands (use bb_pct, bb_width)
            "bb_high", "bb_mid", "bb_low",
            "kc_high", "kc_low",
            # Raw OBV (use obv_slope)
            "obv",
            # Raw MACD (use normalised versions)
            "macd_line", "macd_signal",
            # Raw ATR (use atr_pct)
            "atr",
            # Raw volume base (use volume_ratio)
            "volume_sma_20",
            # Pivot / fib raw levels (use distances)
            "pivot", "r1", "r2", "s1", "s2",
            "fib_236", "fib_382", "fib_500", "fib_618",
        }

        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

        pos_rate = y.mean() * 100
        print(f"Training data: {X.shape[0]:,} rows × {X.shape[1]} features")
        print(f"Positive rate: {pos_rate:.1f}%  (target ~30-45%)")
        return X, y, feature_cols

    # ------------------------------------------------------------------ #
    #  PRIVATE HELPERS                                                     #
    # ------------------------------------------------------------------ #

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        df.columns = [str(c).lower().strip() for c in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in [5, 10, 20, 50, 100]:
            df[f"sma_{w}"] = df["close"].rolling(w).mean()
        # sma_200 alias (100 bars to reduce warmup rows)
        df["sma_200"] = df["close"].rolling(100).mean()
        for s in [9, 21, 50]:
            df[f"ema_{s}"] = df["close"].ewm(span=s, adjust=False).mean()
        return df

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        # ── Standard indicators ──────────────────────────────────────────
        df["rsi_14"] = momentum.rsi(df["close"], window=14)
        df["rsi_7"] = momentum.rsi(df["close"], window=7)
        df["stoch_k"] = momentum.stoch(df["high"], df["low"], df["close"])
        df["stoch_d"] = momentum.stoch_signal(df["high"], df["low"], df["close"])
        df["williams_r"] = momentum.williams_r(df["high"], df["low"], df["close"])

        macd_ind = trend.MACD(df["close"])
        df["macd_line"] = macd_ind.macd()
        df["macd_signal"] = macd_ind.macd_signal()
        df["macd_hist"] = macd_ind.macd_diff()

        adx_ind = trend.ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx_ind.adx()
        df["adx_pos"] = adx_ind.adx_pos()
        df["adx_neg"] = adx_ind.adx_neg()

        # ── Price momentum (Rate of Change) ──────────────────────────────
        for d in [1, 3, 5, 10, 20]:
            df[f"price_change_{d}d"] = df["close"].pct_change(d)
        # Named ROC periods (same calculation, different windows)
        df["roc_5"] = df["close"].pct_change(5) * 100
        df["roc_10"] = df["close"].pct_change(10) * 100
        df["roc_20"] = df["close"].pct_change(20) * 100

        # ── Stochastic RSI ───────────────────────────────────────────────
        # StochRSI = (RSI - min(RSI,14)) / (max(RSI,14) - min(RSI,14))
        rsi = df["rsi_14"].copy()
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        rsi_range = rsi_max - rsi_min
        df["stoch_rsi"] = np.where(
            rsi_range > 0, (rsi - rsi_min) / rsi_range, 0.5
        )
        df["stoch_rsi_k"] = df["stoch_rsi"].rolling(3).mean()
        df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(3).mean()

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        bb = volatility.BollingerBands(df["close"])
        df["bb_high"] = bb.bollinger_hband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_low"] = bb.bollinger_lband()
        df["bb_width"] = bb.bollinger_wband()
        df["bb_pct"] = bb.bollinger_pband()  # 0=lower band, 1=upper band

        kc = volatility.KeltnerChannel(df["high"], df["low"], df["close"])
        df["kc_high"] = kc.keltner_channel_hband()
        df["kc_low"] = kc.keltner_channel_lband()
        # Squeeze: BB inside KC → coiled spring
        df["squeeze_on"] = (
            (df["bb_low"] > df["kc_low"]) & (df["bb_high"] < df["kc_high"])
        ).astype(int)

        df["atr"] = volatility.average_true_range(df["high"], df["low"], df["close"])
        df["atr_pct"] = df["atr"] / (df["close"] + 1e-10)  # Normalized ATR

        df["volatility_5d"] = df["close"].pct_change().rolling(5).std()
        df["volatility_20d"] = df["close"].pct_change().rolling(20).std()
        # Historical volatility annualized (20-day)
        df["hist_vol_20"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # ── OBV ──────────────────────────────────────────────────────────
        df["obv"] = volume.on_balance_volume(df["close"], df["volume"])

        # ── Volume Price Trend (VPT) ──────────────────────────────────────
        # VPT captures cumulative volume weighted by price-change percentage
        price_change_pct = df["close"].pct_change().fillna(0)
        df["vpt"] = (price_change_pct * df["volume"]).cumsum()
        df["vpt_sma_14"] = df["vpt"].rolling(14).mean()
        df["vpt_signal"] = df["vpt"] - df["vpt_sma_14"]  # crossover signal

        # ── Accumulation/Distribution Line ───────────────────────────────
        # A/D = previous A/D + CLV * Volume
        # CLV (Close Location Value) = ((Close-Low)-(High-Close))/(High-Low)
        hl_range = df["high"] - df["low"]
        clv = np.where(
            hl_range > 0,
            ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range,
            0,
        )
        df["ad_line"] = (clv * df["volume"]).cumsum()
        df["ad_slope"] = df["ad_line"].diff(5) / (
            df["ad_line"].abs().rolling(5).mean() + 1e-10
        )

        # ── Money Flow Index (MFI) ────────────────────────────────────────
        # Volume-weighted RSI (14 period) — identifies overbought/oversold with volume
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]
        delta_tp = typical_price.diff()
        pos_flow = money_flow.where(delta_tp > 0, 0).rolling(14).sum()
        neg_flow = money_flow.where(delta_tp < 0, 0).rolling(14).sum()
        mfi_ratio = pos_flow / (neg_flow + 1e-10)
        df["mfi_14"] = 100 - (100 / (1 + mfi_ratio))

        # ── Standard volume metrics ───────────────────────────────────────
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-10)
        df["volume_change"] = df["volume"].pct_change()

        return df

    def _add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot points, Fibonacci retracement, support/resistance,
        and Higher Highs / Lower Lows detection.
        """
        # ── Classic Pivot Points (daily, using previous bar) ──────────────
        # PP = (H + L + C) / 3  |  R1 = 2*PP - L  |  S1 = 2*PP - H
        prev_h = df["high"].shift(1)
        prev_l = df["low"].shift(1)
        prev_c = df["close"].shift(1)
        pivot = (prev_h + prev_l + prev_c) / 3
        df["pivot"] = pivot
        df["r1"] = 2 * pivot - prev_l
        df["r2"] = pivot + (prev_h - prev_l)
        df["s1"] = 2 * pivot - prev_h
        df["s2"] = pivot - (prev_h - prev_l)

        # Distances from pivot levels (normalised by ATR)
        atr = df["atr"].replace(0, np.nan).fillna(df["close"] * 0.02)
        df["dist_pivot"] = (df["close"] - pivot) / atr
        df["dist_r1"] = (df["close"] - df["r1"]) / atr
        df["dist_s1"] = (df["close"] - df["s1"]) / atr

        # ── Fibonacci Retracement (20-bar swing high/low) ─────────────────
        swing_high = df["high"].rolling(20).max()
        swing_low = df["low"].rolling(20).min()
        fib_range = swing_high - swing_low

        df["fib_236"] = swing_high - 0.236 * fib_range
        df["fib_382"] = swing_high - 0.382 * fib_range
        df["fib_500"] = swing_high - 0.500 * fib_range
        df["fib_618"] = swing_high - 0.618 * fib_range

        # Distance of close from each fib level (normalised)
        for lvl, col in [
            (0.236, "fib_236"),
            (0.382, "fib_382"),
            (0.500, "fib_500"),
            (0.618, "fib_618"),
        ]:
            df[f"dist_{col}"] = (df["close"] - df[col]) / (atr + 1e-10)

        # ── 52-week high/low distances ────────────────────────────────────
        df["dist_52w_high"] = (df["close"] - df["high"].rolling(200).max()) / (
            df["high"].rolling(200).max() + 1e-10
        )
        df["dist_52w_low"] = (df["close"] - df["low"].rolling(200).min()) / (
            df["low"].rolling(200).min() + 1e-10
        )

        # ── Higher Highs / Lower Lows detection (10-bar window) ───────────
        recent_high = df["high"].rolling(10).max()
        prior_high = df["high"].rolling(10).max().shift(10)
        recent_low = df["low"].rolling(10).min()
        prior_low = df["low"].rolling(10).min().shift(10)

        df["higher_high"] = (recent_high > prior_high).astype(int)
        df["lower_low"] = (recent_low < prior_low).astype(int)
        # Both HH + no LL = strong uptrend structure
        df["trend_structure_bull"] = (
            (df["higher_high"] == 1) & (df["lower_low"] == 0)
        ).astype(int)

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalised features — key to cross-stock comparability."""
        eps = 1e-10

        # Distance from MAs (normalised by MA value)
        for name, ma_col in [
            ("ema9", "ema_9"),
            ("ema21", "ema_21"),
            ("ema50", "ema_50"),
            ("sma20", "sma_20"),
            ("sma50", "sma_50"),
            ("sma200", "sma_200"),
        ]:
            df[f"dist_{name}"] = (df["close"] - df[ma_col]) / (df[ma_col] + eps)

        # MA slopes (momentum of the MA itself)
        df["ema9_slope"] = df["ema_9"].pct_change(3)
        df["ema21_slope"] = df["ema_21"].pct_change(5)
        df["sma50_slope"] = df["sma_50"].pct_change(10)

        # MA alignment score: how many MAs is price above? (0-4)
        df["ma_alignment"] = (
            (df["close"] > df["ema_9"]).astype(int)
            + (df["close"] > df["ema_21"]).astype(int)
            + (df["close"] > df["sma_50"]).astype(int)
            + (df["close"] > df["sma_200"]).astype(int)
        )

        # Candle geometry
        df["hl_range"] = (df["high"] - df["low"]) / (df["close"] + eps)
        df["close_position"] = (df["close"] - df["low"]) / (
            df["high"] - df["low"] + eps
        )
        df["body_size"] = abs(df["close"] - df["open"]) / (df["close"] + eps)
        df["upper_wick"] = (
            df["high"] - df[["close", "open"]].max(axis=1)
        ) / (df["close"] + eps)
        df["lower_wick"] = (
            df[["close", "open"]].min(axis=1) - df["low"]
        ) / (df["close"] + eps)

        # RSI derived
        df["rsi_slope"] = df["rsi_14"].diff(3)
        df["rsi_dist_50"] = df["rsi_14"] - 50

        # MACD normalised by price
        df["macd_norm"] = df["macd_line"] / (df["close"] + eps)
        df["macd_sig_norm"] = df["macd_signal"] / (df["close"] + eps)
        df["macd_hist_norm"] = df["macd_hist"] / (df["close"] + eps)

        # MACD histogram divergence (histogram expanding or contracting)
        df["macd_hist_slope"] = df["macd_hist"].diff(3)

        # ADX directional balance
        df["di_diff"] = df["adx_pos"] - df["adx_neg"]

        # Stoch derived
        df["stoch_diff"] = df["stoch_k"] - df["stoch_d"]

        # OBV slope (normalised)
        df["obv_slope"] = df["obv"].diff(5) / (
            df["obv"].abs().rolling(5).mean() + eps
        )

        # Volume-price relationship (VP trend)
        df["vp_trend"] = df["price_change_1d"] * df["volume_ratio"]

        # Consecutive up/down days
        up_days = (df["close"] > df["close"].shift(1)).astype(int)
        df["consec_up"] = (
            up_days.groupby((up_days != up_days.shift()).cumsum()).cumcount() + 1
        ) * up_days  # zero on down days

        # (Close - 20 SMA) / ATR  → normalised deviation
        df["close_sma20_atr"] = (df["close"] - df["sma_20"]) / (df["atr"] + eps)

        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary flags capturing market regime — high signal-to-noise."""

        # Price vs MAs
        df["above_ema9"] = (df["close"] > df["ema_9"]).astype(int)
        df["above_ema21"] = (df["close"] > df["ema_21"]).astype(int)
        df["above_sma50"] = (df["close"] > df["sma_50"]).astype(int)
        df["above_sma200"] = (df["close"] > df["sma_200"]).astype(int)

        # Full uptrend alignment
        df["full_uptrend"] = (
            (df["close"] > df["ema_9"])
            & (df["ema_9"] > df["ema_21"])
            & (df["ema_21"] > df["sma_50"])
        ).astype(int)

        # Golden / death cross (50 SMA vs 200 SMA)
        df["golden_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)

        # Market regime: bull / bear / neutral
        # Bull  = price > sma_50 > sma_200
        # Bear  = price < sma_50 < sma_200
        df["regime_bull"] = (
            (df["close"] > df["sma_50"]) & (df["sma_50"] > df["sma_200"])
        ).astype(int)
        df["regime_bear"] = (
            (df["close"] < df["sma_50"]) & (df["sma_50"] < df["sma_200"])
        ).astype(int)

        # RSI zones
        df["rsi_oversold"] = (df["rsi_14"] < 35).astype(int)
        df["rsi_overbought"] = (df["rsi_14"] > 65).astype(int)
        df["rsi_mid_bull"] = (
            (df["rsi_14"] >= 50) & (df["rsi_14"] < 65)
        ).astype(int)

        # MFI zones
        df["mfi_oversold"] = (df["mfi_14"] < 20).astype(int)
        df["mfi_overbought"] = (df["mfi_14"] > 80).astype(int)

        # Stoch bullish cross (K crosses D from below 30)
        df["stoch_bullish_cross"] = (
            (df["stoch_k"] > df["stoch_d"])
            & (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
            & (df["stoch_k"] < 50)
        ).astype(int)

        # MACD bullish cross
        df["macd_bullish_cross"] = (
            (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
        ).astype(int)

        # ADX trend strength
        df["strong_trend"] = (df["adx"] > 25).astype(int)
        df["weak_trend"] = (df["adx"] < 20).astype(int)

        # Volume breakout
        df["vol_breakout"] = (df["volume_ratio"] > 2.0).astype(int)

        # Dark pool proxy: volume spike + narrow price range
        df["dark_pool_proxy"] = (
            (df["volume_ratio"] > 1.5) & (df["hl_range"] < df["atr_pct"])
        ).astype(int)

        # Squeeze release
        df["squeeze_release"] = (
            (df["squeeze_on"].shift(1) == 1) & (df["squeeze_on"] == 0)
        ).astype(int)

        # BB position zones
        df["near_bb_lower"] = (df["bb_pct"] < 0.2).astype(int)
        df["near_bb_upper"] = (df["bb_pct"] > 0.8).astype(int)

        return df

    def _add_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR-normalised candlestick patterns."""
        atr = df["atr"].replace(0, np.nan).fillna(df["close"] * 0.02)

        df["body_atr"] = abs(df["close"] - df["open"]) / atr
        df["upper_wick_atr"] = (
            df["high"] - df[["close", "open"]].max(axis=1)
        ) / atr
        df["lower_wick_atr"] = (
            df[["close", "open"]].min(axis=1) - df["low"]
        ) / atr

        # Hammer: small body, long lower wick, small upper wick
        df["hammer"] = (
            (df["lower_wick_atr"] > 1.5)
            & (df["body_atr"] < 0.5)
            & (df["upper_wick_atr"] < 0.3)
        ).astype(int)

        # Bullish engulfing
        df["bullish_engulfing"] = (
            (df["close"].shift(1) < df["open"].shift(1))  # prev red
            & (df["close"] > df["open"])  # curr green
            & (df["close"] > df["open"].shift(1))
            & (df["open"] < df["close"].shift(1))
        ).astype(int)

        return df

    def _add_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineered interaction features combining two indicators into one.
        These capture compound signals that are stronger than either alone.
        """
        eps = 1e-10

        # RSI × Volume ratio — momentum with volume conviction
        df["rsi_x_vol"] = (df["rsi_14"] / 100.0) * df["volume_ratio"]

        # MACD histogram × Stoch %K — dual momentum confirmation
        df["macd_x_stoch"] = df["macd_hist_norm"] * (df["stoch_k"] / 100.0)

        # ADX × di_diff — trend strength × direction
        df["adx_x_di"] = (df["adx"] / 100.0) * np.sign(df["di_diff"])

        # BB %B × RSI — momentum within BB band position
        df["bb_x_rsi"] = df["bb_pct"] * (df["rsi_14"] / 100.0)

        # Volume ratio × ROC — volume-weighted price acceleration
        df["vol_x_roc10"] = df["volume_ratio"] * df["roc_10"] / 100.0

        # StochRSI × MFI — dual overbought/oversold signal
        df["stochrsi_x_mfi"] = df["stoch_rsi"] * (df["mfi_14"] / 100.0)

        # ADX × close_sma20_atr — trend confirmation of deviation
        df["adx_x_dev"] = (df["adx"] / 100.0) * df["close_sma20_atr"]

        return df
