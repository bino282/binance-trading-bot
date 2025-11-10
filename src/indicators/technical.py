"""
Technical Indicators
Implements all technical indicators used in the trading strategy.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class TechnicalIndicators:
    """Calculate technical indicators for trading strategy."""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Price series
            period: RSI period
            
        Returns:
            RSI series
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data: Price series
            period: EMA period
            
        Returns:
            EMA series
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = TechnicalIndicators.calculate_ema(data, fast)
        ema_slow = TechnicalIndicators.calculate_ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ADX period
            
        Returns:
            ADX series
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index).rolling(window=period).mean()
        minus_dm = pd.Series(minus_dm, index=low.index).rolling(window=period).mean()
        
        # Calculate Directional Indicators
        plus_di = 100 * (plus_dm / atr)
        minus_di = 100 * (minus_dm / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period
            
        Returns:
            ATR series
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_cmf(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate Chaikin Money Flow (CMF).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            period: CMF period
            
        Returns:
            CMF series
        """
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        mfv = mfm * volume
        
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf
    
    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            
        Returns:
            VWAP series
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def calculate_tsi(
        data: pd.Series,
        r: int = 25,
        s: int = 13,
        signal_period: int = 7
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate True Strength Index (TSI).
        
        Args:
            data: Price series
            r: First smoothing period
            s: Second smoothing period
            signal_period: Signal line period
            
        Returns:
            Tuple of (TSI line, Signal line)
        """
        momentum = data.diff()
        
        # Double smoothed momentum
        smoothed_momentum = momentum.ewm(span=r, adjust=False).mean()
        double_smoothed_momentum = smoothed_momentum.ewm(span=s, adjust=False).mean()
        
        # Double smoothed absolute momentum
        abs_momentum = abs(momentum)
        smoothed_abs_momentum = abs_momentum.ewm(span=r, adjust=False).mean()
        double_smoothed_abs_momentum = smoothed_abs_momentum.ewm(span=s, adjust=False).mean()
        
        # TSI
        tsi = 100 * (double_smoothed_momentum / double_smoothed_abs_momentum)
        signal = tsi.ewm(span=signal_period, adjust=False).mean()
        
        return tsi, signal
    
    @staticmethod
    def calculate_stochastic_rsi(
        data: pd.Series,
        length: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic RSI.
        
        Args:
            data: Price series
            length: RSI period
            smooth_k: %K smoothing period
            smooth_d: %D smoothing period
            
        Returns:
            Tuple of (%K, %D)
        """
        rsi = TechnicalIndicators.calculate_rsi(data, length)
        
        # Stochastic of RSI
        rsi_min = rsi.rolling(window=length).min()
        rsi_max = rsi.rolling(window=length).max()
        
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
        stoch_rsi = stoch_rsi.fillna(0)
        
        # Smooth %K and %D
        k = stoch_rsi.rolling(window=smooth_k).mean() * 100
        d = k.rolling(window=smooth_d).mean()
        
        return k, d
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV series
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def calculate_adi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate Accumulation/Distribution Index (ADI).
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            
        Returns:
            ADI series
        """
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        adi = (clv * volume).cumsum()
        return adi
    
    @staticmethod
    def calculate_z_score(data: pd.Series, window: int = 50) -> pd.Series:
        """
        Calculate rolling Z-score.
        
        Args:
            data: Data series
            window: Rolling window period
            
        Returns:
            Z-score series
        """
        mean = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        z_score = (data - mean) / std
        return z_score.fillna(0)
    
    @staticmethod
    def calculate_bb_width_percentile(
        data: pd.Series,
        bb_period: int = 20,
        bb_std: float = 2.0,
        percentile_window: int = 200
    ) -> pd.Series:
        """
        Calculate Bollinger Bands width percentile.
        
        Args:
            data: Price series
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation
            percentile_window: Percentile calculation window
            
        Returns:
            BB width percentile series
        """
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(data, bb_period, bb_std)
        bb_width = (upper - lower) / middle
        
        # Calculate percentile rank
        percentile = bb_width.rolling(window=percentile_window).apply(
            lambda x: (x < x.iloc[-1]).sum() / len(x) * 100,
            raw=False
        )
        
        return percentile


def add_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add all technical indicators to dataframe based on configuration.
    
    Args:
        df: OHLCV dataframe with columns: open, high, low, close, volume
        config: Indicators configuration dictionary
        
    Returns:
        Dataframe with all indicators added
    """
    indicators = TechnicalIndicators()
    
    # RSI
    rsi_length = config.get('rsi_length', 14)
    df['rsi'] = indicators.calculate_rsi(df['close'], rsi_length)
    
    # EMAs
    ema_fast = config.get('ema_fast', 34)
    ema_mid = config.get('ema_mid', 89)
    ema_slow = config.get('ema_slow', 200)
    df['ema_fast'] = indicators.calculate_ema(df['close'], ema_fast)
    df['ema_mid'] = indicators.calculate_ema(df['close'], ema_mid)
    df['ema_slow'] = indicators.calculate_ema(df['close'], ema_slow)
    
    # MACD
    macd_fast = config.get('macd_fast', 12)
    macd_slow = config.get('macd_slow', 26)
    macd_signal = config.get('macd_signal', 9)
    df['macd'], df['macd_signal'], df['macd_hist'] = indicators.calculate_macd(
        df['close'], macd_fast, macd_slow, macd_signal
    )
    
    # ADX
    adx_length = config.get('adx_length', 14)
    df['adx'] = indicators.calculate_adx(df['high'], df['low'], df['close'], adx_length)
    
    # ATR
    atr_length = config.get('atr_length', 14)
    df['atr'] = indicators.calculate_atr(df['high'], df['low'], df['close'], atr_length)
    df['atr_pct'] = df['atr'] / df['close']
    
    # Bollinger Bands
    bb_length = config.get('bb_length', 20)
    bb_std = config.get('bb_std', 2.0)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = indicators.calculate_bollinger_bands(
        df['close'], bb_length, bb_std
    )
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # CMF
    cmf_length = config.get('cmf_length', 20)
    df['cmf'] = indicators.calculate_cmf(
        df['high'], df['low'], df['close'], df['volume'], cmf_length
    )
    
    # VWAP
    if config.get('use_vwap', True):
        df['vwap'] = indicators.calculate_vwap(
            df['high'], df['low'], df['close'], df['volume']
        )
        df['vwap_distance_pct'] = (df['close'] - df['vwap']) / df['vwap']
    
    # TSI
    tsi_config = config.get('tsi', {})
    tsi_r = tsi_config.get('r', 25)
    tsi_s = tsi_config.get('s', 13)
    tsi_signal = tsi_config.get('signal', 7)
    df['tsi'], df['tsi_signal'] = indicators.calculate_tsi(
        df['close'], tsi_r, tsi_s, tsi_signal
    )
    
    # Stochastic RSI
    stochrsi_config = config.get('stochrsi', {})
    stochrsi_length = stochrsi_config.get('length', 14)
    stochrsi_k = stochrsi_config.get('smooth_k', 3)
    stochrsi_d = stochrsi_config.get('smooth_d', 3)
    df['stochrsi_k'], df['stochrsi_d'] = indicators.calculate_stochastic_rsi(
        df['close'], stochrsi_length, stochrsi_k, stochrsi_d
    )
    
    # OBV
    df['obv'] = indicators.calculate_obv(df['close'], df['volume'])
    
    # ADI
    df['adi'] = indicators.calculate_adi(
        df['high'], df['low'], df['close'], df['volume']
    )
    
    # Flow indicators with slopes
    flow_windows = config.get('flow_windows', {})
    adi_slope_window = flow_windows.get('adi_slope', 34)
    obv_slope_window = flow_windows.get('obv_slope', 34)
    
    df['adi_slope'] = df['adi'].diff(adi_slope_window) / adi_slope_window
    df['obv_slope'] = df['obv'].diff(obv_slope_window) / obv_slope_window
    
    # Z-scores
    z_window = flow_windows.get('z_window', 50)
    df['adi_z'] = indicators.calculate_z_score(df['adi'], z_window)
    df['obv_z'] = indicators.calculate_z_score(df['obv'], z_window)
    
    macd_hist_z_window = config.get('macd_hist_z_window', 50)
    df['macd_hist_z'] = indicators.calculate_z_score(df['macd_hist'], macd_hist_z_window)
    
    # BB width percentile
    bb_width_pctile_window = config.get('bb_width_pctile_window', 200)
    df['bb_width_pctile'] = indicators.calculate_bb_width_percentile(
        df['close'], bb_length, bb_std, bb_width_pctile_window
    )
    
    return df
