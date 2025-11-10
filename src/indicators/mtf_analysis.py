"""
MTF Analysis
Provides multi-timeframe trend and alignment analysis.
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional

from ..utils.config_loader import ConfigLoader
from .technical import add_all_indicators


class MTFAnalysis:
    """
    Analyzes trend and alignment across multiple timeframes.
    """
    
    def __init__(self, config: dict):
        """
        Initialize MTF Analysis.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        self.mtf_cfg = config.get('mtf', {})
        self.enabled = self.mtf_cfg.get('enabled', True)
        self.higher_timeframes = self.mtf_cfg.get('higher_timeframes', [])
        self.indicators_config = config.get('policy_cfg', {}).get('indicators', {})
    
    def get_htf_trend(self, htf_df: pd.DataFrame) -> str:
        """
        Determine the trend of a higher timeframe using a simplified EMA cross.
        
        Args:
            htf_df: Higher timeframe OHLCV data with indicators
            
        Returns:
            'up', 'down', or 'sideways'
        """
        if htf_df.empty:
            return 'sideways'
        
        # Use EMA fast/mid cross for trend
        ema_fast = self.indicators_config.get('ema_fast', 34)
        ema_mid = self.indicators_config.get('ema_mid', 89)
        
        # Ensure indicators are calculated
        if f'ema_fast' not in htf_df.columns or f'ema_mid' not in htf_df.columns:
            htf_df = add_all_indicators(htf_df, self.indicators_config)
        
        last_row = htf_df.iloc[-1]
        
        if last_row['ema_fast'] > last_row['ema_mid']:
            return 'up'
        elif last_row['ema_fast'] < last_row['ema_mid']:
            return 'down'
        else:
            return 'sideways'
    
    def analyze_htf_alignment(
        self,
        htf_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, str]]:
        """
        Analyze the trend for all configured higher timeframes.
        
        Args:
            htf_data: Dictionary mapping interval to higher timeframe DataFrame
            
        Returns:
            Dictionary mapping interval to trend analysis
        """
        if not self.enabled:
            return {}
        
        alignment_analysis = {}
        
        for interval in self.higher_timeframes:
            if interval in htf_data:
                df = htf_data[interval]
                
                # Only use the last completed bar for analysis
                if len(df) > 1:
                    # Calculate indicators on the HTF data
                    df = add_all_indicators(df, self.indicators_config)
                    
                    # Determine trend
                    trend = self.get_htf_trend(df)
                    
                    alignment_analysis[interval] = {
                        'trend': trend,
                        'last_close': df.iloc[-1]['close']
                    }
        
        return alignment_analysis
    
    def get_htf_data_for_score_engine(
        self,
        htf_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, str]]:
        """
        Prepare HTF data structure for the Score Engine.
        
        Args:
            htf_data: Dictionary mapping interval to higher timeframe DataFrame
            
        Returns:
            Simplified dictionary for Score Engine
        """
        analysis = self.analyze_htf_alignment(htf_data)
        
        # Simplify for Score Engine (e.g., only need trend)
        simplified = {}
        for interval, data in analysis.items():
            simplified[interval] = {'trend': data['trend']}
        
        return simplified

    def prepare_all_htf_data(self, base_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Resample base data to all configured higher timeframes and calculate indicators.
        
        Args:
            base_data: Base timeframe OHLCV data
            
        Returns:
            Dictionary mapping interval to higher timeframe DataFrame with indicators
        """
        if not self.enabled:
            return {}
            
        htf_data = {}
        
        for interval in self.higher_timeframes:
            # Resample to higher timeframe
            htf_df = base_data.resample(interval).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate indicators on the HTF data
            htf_df = add_all_indicators(htf_df, self.indicators_config)
            
            # Forward fill the HTF data to align with the base timeframe index
            # This is crucial for the backtest engine to use the last completed HTF bar
            htf_df = htf_df.reindex(base_data.index, method='ffill')
            
            htf_data[interval] = htf_df
            
        return htf_data
