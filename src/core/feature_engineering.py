"""
TradeSage - Feature Engineering v5
~105+ ML-ready features: raw indicators → normalized derivations → regime flags → market context → fundamentals
Fixes SMA_200 mismatch (now true 200-bar), 52-week window (now 252 bars).
v5: Adds fundamental features (P/E, ROE, ROCE, Debt/Equity, Shareholding) via Obscura+Screener.
"""

import pandas as pd
import numpy as np
import logging
from ta import trend, momentum, volatility, volume
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generate technical indicators and ML-ready features."""

    def __init__(self):
        self._scraper = None  # Lazy-loaded ScreenerScraper
        self._fund_cache = {}  # {symbol: {metric: value}}

    def _get_scraper(self):
        """Lazy-load ScreenerScraper to avoid import errors when Obscura isn't installed."""
        if self._scraper is None:
            try:
                from src.core.screener_scraper import ScreenerScraper
                self._scraper = ScreenerScraper()
            except Exception as e:
                logger.warning(f"Could not load ScreenerScraper: {e}")
        return self._scraper

    def set_fundamentals_cache(self, cache: dict):
        """Pre-load fundamentals cache from batch fetch (used by training pipeline)."""
        self._fund_cache = cache

    def add_technical_indicators(self, df, index_df=None, symbol=None):
        """
        Full feature pipeline: raw indicators -> normalized derived features -> regime flags -> fundamentals.
        Returns DataFrame with ~105+ ML-ready features.
        Optionally accepts index_df (e.g. Nifty50) for market context features.
        If symbol is provided, injects fundamental features from Screener.in.
        """
        df = self._clean_df(df)
        df = self._add_moving_averages(df)
        df = self._add_momentum(df)
        df = self._add_volatility(df)
        df = self._add_volume_features(df)
        df = self._add_derived_features(df)
        df = self._add_regime_features(df)
        df = self._add_candle_features(df)
        df = self._add_time_features(df)
        df = self._add_market_structure(df)
        df = self._add_feature_interactions(df)
        if index_df is not None:
            df = self._add_market_context(df, index_df)
        if symbol is not None:
            df = self._add_fundamental_features(df, symbol)
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
        for w in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{w}'] = df['close'].rolling(w).mean()
        for s in [9, 21, 50]:
            df[f'ema_{s}'] = df['close'].ewm(span=s, adjust=False).mean()
        return df

    def _add_momentum(self, df):
        df['rsi_14'] = momentum.rsi(df['close'], window=14)
        df['rsi_7']  = momentum.rsi(df['close'], window=7)
        df['stoch_k'] = momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = momentum.stoch_signal(df['high'], df['low'], df['close'])
        df['williams_r'] = momentum.williams_r(df['high'], df['low'], df['close'])

        # Stochastic RSI (new)
        rsi_series = df['rsi_14'].copy()
        stoch_rsi_k = ((rsi_series - rsi_series.rolling(14).min()) /
                       (rsi_series.rolling(14).max() - rsi_series.rolling(14).min() + 1e-10))
        df['stoch_rsi'] = stoch_rsi_k

        # Rate of Change (new)
        df['roc_10'] = df['close'].pct_change(10) * 100

        # MACD components
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
        df['bb_pct']   = bb.bollinger_pband()

        kc = volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['kc_high'] = kc.keltner_channel_hband()
        df['kc_low']  = kc.keltner_channel_lband()
        df['squeeze_on'] = (
            (df['bb_low'] > df['kc_low']) & (df['bb_high'] < df['kc_high'])
        ).astype(int)

        df['atr'] = volatility.average_true_range(df['high'], df['low'], df['close'])
        df['atr_pct'] = df['atr'] / df['close']

        df['volatility_5d']  = df['close'].pct_change().rolling(5).std()
        df['volatility_20d'] = df['close'].pct_change().rolling(20).std()

        # Historical volatility ratio (new) — low vs high vol regime
        vol_60d_median = df['volatility_20d'].rolling(60).median()
        df['vol_regime'] = (df['volatility_20d'] / (vol_60d_median + 1e-10)).clip(0, 5)

        return df

    def _add_volume_features(self, df):
        df['obv'] = volume.on_balance_volume(df['close'], df['volume'])
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio']  = df['volume'] / (df['volume_sma_20'] + 1e-10)
        df['volume_change'] = df['volume'].pct_change()

        # Money Flow Index (new)
        df['mfi'] = volume.money_flow_index(
            df['high'], df['low'], df['close'], df['volume'], window=14
        )

        # Volume Price Trend (new)
        df['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()

        # Accumulation/Distribution Line (new)
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        df['ad_line'] = (clv * df['volume']).cumsum()

        # Chaikin Money Flow (new)
        mfv = clv * df['volume']
        df['cmf'] = mfv.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-10)

        # Volume spike + narrow range (institutional proxy) (new)
        narrow_range = (df['high'] - df['low']) / (df['close'] + 1e-10)
        nr_threshold = narrow_range.rolling(20).quantile(0.25)
        df['vol_spike_narrow'] = (
            (df['volume_ratio'] > 1.5) & (narrow_range < nr_threshold)
        ).astype(int)

        return df

    def _add_derived_features(self, df):
        """Normalized features — raw prices are meaningless; ratios are not."""
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

        # MA alignment score
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
        df['rsi_dist_50'] = df['rsi_14'] - 50

        # MACD normalized by price
        df['macd_norm']      = df['macd_line']   / (df['close'] + eps)
        df['macd_sig_norm']  = df['macd_signal'] / (df['close'] + eps)
        df['macd_hist_norm'] = df['macd_hist']   / (df['close'] + eps)

        # ADX directional balance
        df['di_diff'] = df['adx_pos'] - df['adx_neg']
        df['trend_strength'] = df['adx'] * df['di_diff']

        # Stoch position
        df['stoch_diff'] = df['stoch_k'] - df['stoch_d']

        # OBV slope (normalized)
        df['obv_slope'] = df['obv'].diff(5) / (df['obv'].abs().rolling(5).mean() + eps)

        # Volume-price relationship
        df['vp_trend'] = df['price_change_1d'] * df['volume_ratio']

        # Momentum / volatility ratio
        df['ret_5d']       = df['close'].pct_change(5)
        df['vol_5d']       = df['close'].pct_change().rolling(5).std()
        df['momentum_vol'] = (df['ret_5d'] / (df['vol_5d'] + eps)).clip(-5, 5)

        # 52-week high/low distance (FIXED: now uses 252 bars)
        rolling_high = df['high'].rolling(252, min_periods=200).max()
        rolling_low  = df['low'].rolling(252, min_periods=200).min()
        df['dist_52w_high'] = (df['close'] - rolling_high) / (rolling_high + eps)
        df['dist_52w_low']  = (df['close'] - rolling_low)  / (rolling_low  + eps)

        # Consecutive up/down days
        up_days = (df['close'] > df['close'].shift(1)).astype(int)
        df['consec_up'] = up_days.groupby((up_days != up_days.shift()).cumsum()).cumcount() + 1
        df['consec_up'] = df['consec_up'] * up_days

        # VPT slope (new — normalized)
        df['vpt_slope'] = df['vpt'].diff(5) / (df['vpt'].abs().rolling(5).mean() + eps)

        # A/D slope (new — normalized)
        df['ad_slope'] = df['ad_line'].diff(5) / (df['ad_line'].abs().rolling(5).mean() + eps)

        # Gap detection — overnight gap percentage
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + eps)

        # Momentum acceleration — rate of change of rate of change
        df['momentum_accel'] = df['roc_10'].diff(5).clip(-50, 50)

        # Drawdown from recent high — mean-reversion signal
        rolling_20h = df['high'].rolling(20).max()
        df['drawdown_20d'] = (df['close'] - rolling_20h) / (df['close'] + eps)

        return df

    def _add_regime_features(self, df):
        """Binary flags capturing market regime."""
        df['above_ema9']   = (df['close'] > df['ema_9']).astype(int)
        df['above_ema21']  = (df['close'] > df['ema_21']).astype(int)
        df['above_sma50']  = (df['close'] > df['sma_50']).astype(int)
        df['above_sma200'] = (df['close'] > df['sma_200']).astype(int)

        # Full uptrend
        df['full_uptrend'] = (
            (df['close']  > df['ema_9']) &
            (df['ema_9']  > df['ema_21']) &
            (df['ema_21'] > df['sma_50'])
        ).astype(int)

        # Golden / death cross
        df['golden_cross'] = (df['sma_50'] > df['sma_200']).astype(int)

        # Bull/Bear/Neutral regime (new)
        # 2 = bull (50>200, price>50), 1 = neutral, 0 = bear
        df['market_regime'] = 1  # neutral default
        df.loc[(df['sma_50'] > df['sma_200']) & (df['close'] > df['sma_50']), 'market_regime'] = 2
        df.loc[(df['sma_50'] < df['sma_200']) & (df['close'] < df['sma_50']), 'market_regime'] = 0

        # RSI zones
        df['rsi_oversold']   = (df['rsi_14'] < 35).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 65).astype(int)
        df['rsi_mid_bull']   = ((df['rsi_14'] >= 50) & (df['rsi_14'] < 65)).astype(int)

        # Stoch bullish cross
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

        # MACD histogram divergence (new) — price new high but MACD hist lower
        price_20h = df['close'].rolling(20).max()
        macd_20h  = df['macd_hist'].rolling(20).max()
        df['macd_divergence'] = (
            (df['close'] >= price_20h * 0.99) &
            (df['macd_hist'] < macd_20h * 0.7) &
            (df['macd_hist'] > 0)
        ).astype(int)

        # ADX trend strength
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        df['weak_trend']   = (df['adx'] < 20).astype(int)

        # Volume breakout
        df['vol_breakout'] = (df['volume_ratio'] > 2.0).astype(int)

        # Squeeze release
        df['squeeze_release'] = (
            (df['squeeze_on'].shift(1) == 1) & (df['squeeze_on'] == 0)
        ).astype(int)

        # BB position zones
        df['near_bb_lower'] = (df['bb_pct'] < 0.2).astype(int)
        df['near_bb_upper'] = (df['bb_pct'] > 0.8).astype(int)

        # High volatility regime
        vol_60d_med = df['volatility_20d'].rolling(60).median()
        df['high_vol_regime'] = (df['volatility_20d'] > vol_60d_med * 1.3).astype(int)

        return df

    def _add_candle_features(self, df):
        """ATR-normalized candle geometry (pattern flags removed — low signal on daily bars)."""
        atr = df['atr'].replace(0, np.nan).fillna(df['close'] * 0.02)

        df['body_atr']        = abs(df['close'] - df['open']) / atr
        df['upper_wick_atr']  = (df['high'] - df[['close','open']].max(axis=1)) / atr
        df['lower_wick_atr']  = (df[['close','open']].min(axis=1) - df['low']) / atr

        return df

    def _add_time_features(self, df):
        """Derived time features (raw calendar features removed — low signal)."""
        # Relative volume by day-of-week (detects unusual activity vs same weekday)
        dow = df.index.dayofweek
        vol_by_dow = df['volume'].copy()
        dow_mean = vol_by_dow.groupby(dow).transform(
            lambda x: x.rolling(8, min_periods=4).mean()
        )
        df['rel_vol_dow'] = (df['volume'] / (dow_mean + 1e-10)).clip(0, 10)

        return df

    def _add_market_structure(self, df):
        """Higher Highs/Lower Lows, Support/Resistance, Fibonacci, Pivots."""
        # Higher Highs / Lower Lows (20-bar rolling)
        high_20 = df['high'].rolling(20).max()
        low_20  = df['low'].rolling(20).min()
        prev_high_20 = high_20.shift(20)
        prev_low_20  = low_20.shift(20)

        df['higher_highs'] = (high_20 > prev_high_20).astype(int)
        df['lower_lows']   = (low_20 < prev_low_20).astype(int)
        # Structure score: +1 for HH, -1 for LL
        df['structure_score'] = df['higher_highs'] - df['lower_lows']

        # Support / Resistance proximity (distance to rolling extremes as % of price)
        eps = 1e-10
        support_20  = df['low'].rolling(20).min()
        resist_20   = df['high'].rolling(20).max()
        df['dist_support']    = (df['close'] - support_20) / (df['close'] + eps)
        df['dist_resistance'] = (resist_20 - df['close']) / (df['close'] + eps)

        # Fibonacci retracement position (within 50-bar swing)
        swing_high = df['high'].rolling(50).max()
        swing_low  = df['low'].rolling(50).min()
        swing_range = swing_high - swing_low + eps
        df['fib_position'] = (df['close'] - swing_low) / swing_range  # 0=at low, 1=at high

        # Classic Pivot Points (daily)
        df['pivot']   = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['pivot_dist'] = (df['close'] - df['pivot']) / (df['close'] + eps)

        return df

    def _add_feature_interactions(self, df):
        """Engineered interaction features — combining signals for conviction."""
        eps = 1e-10

        # RSI × Volume (momentum with conviction)
        df['rsi_volume'] = ((df['rsi_14'] - 50) * df['volume_ratio']).clip(-5, 5)

        # Normalized deviation: (Close - 20 SMA) / ATR
        df['norm_deviation'] = (df['close'] - df['sma_20']) / (df['atr'] + eps)
        df['norm_deviation'] = df['norm_deviation'].clip(-5, 5)

        # MACD signal × Stochastic %K (dual momentum)
        df['macd_stoch'] = (df['macd_hist_norm'] * (df['stoch_k'] / 100)).clip(-5, 5)

        # ADX × BB_width (trend strength in volatility context)
        df['adx_bb'] = (df['adx'] * df['bb_width']).clip(0, 5)

        return df

    def _add_market_context(self, df, index_df):
        """
        Relative strength vs market index (e.g. Nifty50).
        """
        try:
            index_df = index_df.copy()
            index_df.columns = [str(c).lower() for c in index_df.columns]

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if index_df.index.tz is not None:
                index_df.index = index_df.index.tz_localize(None)

            index_close = index_df['close'].reindex(df.index, method='ffill')

            if index_close.isna().sum() / len(index_close) > 0.5:
                return df

            # 5d relative strength (existing)
            index_ret_5d = index_close.pct_change(5)
            df['rel_strength']  = df['close'].pct_change(5) - index_ret_5d

            # 20d relative strength (new)
            index_ret_20d = index_close.pct_change(20)
            df['rel_strength_20d'] = df['close'].pct_change(20) - index_ret_20d

            # Market trend
            index_sma50 = index_close.rolling(50).mean()
            df['market_trend']  = (index_close > index_sma50).astype(int)

            # Clip outliers
            df['rel_strength']     = df['rel_strength'].clip(-0.2, 0.2)
            df['rel_strength_20d'] = df['rel_strength_20d'].clip(-0.4, 0.4)
            return df
        except Exception:
            return df

    def _add_fundamental_features(self, df, symbol):
        """
        Inject fundamental metrics as constant features across all rows for a stock.
        These are slow-changing values (updated quarterly) so a constant column is correct.
        Features: fund_pe, fund_roe, fund_roce, fund_debt_equity, fund_promoter,
                  fund_fii, fund_dii, fund_div_yield
        """
        # Default all to 0 (neutral)
        fund_features = {
            'fund_pe': 0.0,
            'fund_roe': 0.0,
            'fund_roce': 0.0,
            'fund_debt_equity': 0.0,
            'fund_promoter': 0.0,
            'fund_fii': 0.0,
            'fund_dii': 0.0,
            'fund_div_yield': 0.0,
        }

        # Try to get fundamentals from pre-loaded cache first
        data = self._fund_cache.get(symbol)

        # Only fall back to live scraper if NO bulk cache was pre-loaded
        # (i.e., during live scanning, not during training)
        if data is None and not self._fund_cache:
            scraper = self._get_scraper()
            if scraper:
                try:
                    data = scraper.fetch_fundamentals(symbol, use_cache=True)
                except Exception as e:
                    logger.debug(f"Failed to fetch fundamentals for {symbol}: {e}")
                    data = {}

        if data:
            fund_features['fund_pe'] = float(data.get('pe_ratio', 0) or 0)
            fund_features['fund_roe'] = float(data.get('roe', 0) or 0)
            fund_features['fund_roce'] = float(data.get('roce', 0) or 0)
            fund_features['fund_debt_equity'] = float(data.get('debt_to_equity', 0) or 0)
            fund_features['fund_promoter'] = float(data.get('promoter_holding', 0) or 0)
            fund_features['fund_fii'] = float(data.get('fii_holding', 0) or 0)
            fund_features['fund_dii'] = float(data.get('dii_holding', 0) or 0)
            fund_features['fund_div_yield'] = float(data.get('dividend_yield', 0) or 0)

        for col, val in fund_features.items():
            df[col] = val

        return df

    # ------------------------------------------------------------------ #
    #  TARGET + TRAINING DATA                                              #
    # ------------------------------------------------------------------ #

    def create_target_variable(self, df, forward_days=10, tp_mult=2.0, sl_mult=2.0,
                               gain_threshold=0.04, threshold=None, max_drawdown=None):
        """
        Forward-return target with optional drawdown filter.
        """
        if threshold is not None:
            gain_threshold = threshold

        df = df.copy()

        future_max_high = df['high'].rolling(forward_days).max().shift(-forward_days)
        pct_gain = (future_max_high - df['close']) / (df['close'] + 1e-10)

        gain_condition = pct_gain >= gain_threshold

        if max_drawdown is not None:
            future_min_low = df['low'].rolling(forward_days).min().shift(-forward_days)
            pct_dd = (future_min_low - df['close']) / (df['close'] + 1e-10)
            dd_condition = pct_dd >= max_drawdown
            df['target'] = (gain_condition & dd_condition).astype(float)
        else:
            df['target'] = gain_condition.astype(float)

        df.loc[df.index[-forward_days:], 'target'] = np.nan

        return df

    def prepare_training_data(self, df, target_col='target'):
        """Returns X, y, feature_names — with strict leakage prevention."""
        df = df.dropna()

        # Strict exclusion — no raw prices, no future-derived, no raw indicators
        exclude_cols = {
            'open', 'high', 'low', 'close', 'volume',
            'target', 'forward_return', 'target_return', 'symbol',
            # raw MA values (use normalized dist_ versions)
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_9', 'ema_21', 'ema_50',
            # raw BB/KC bands
            'bb_high', 'bb_mid', 'bb_low', 'bb_width',
            'kc_high', 'kc_low',
            # raw OBV, VPT, A/D (use slope/normalized versions)
            'obv', 'vpt', 'ad_line',
            # raw MACD line/signal (use normalized versions)
            'macd_line', 'macd_signal',
            # raw ATR (use atr_pct)
            'atr',
            # raw volume SMA
            'volume_sma_20',
            # intermediate
            'ret_5d', 'vol_5d',
            # raw pivot (use pivot_dist)
            'pivot',
            # removed low-value features (safety exclusion for backwards compat)
            'earnings_season', 'day_of_week', 'month',
            'hammer', 'bullish_engulfing',
        }

        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Downcast to float32 to halve memory usage and prevent OOM
        X = df[feature_cols].astype(np.float32)
        y = df[target_col].astype(np.float32)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

        pos_rate = y.mean() * 100
        print(f"Training data: {X.shape[0]:,} rows × {X.shape[1]} features")
        print(f"Positive rate: {pos_rate:.1f}%  (target ~25-45%)")

        return X, y, feature_cols
