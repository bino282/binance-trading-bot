"""
Grid Engine
Implements dynamic grid trading and spread calculation based on volatility.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..utils.config_loader import ConfigLoader


@dataclass
class GridConfig:
    """Configuration container for Grid and Spread Engine."""
    spread_mode: str
    fixed_spread_pct: float
    bands: Dict[str, Dict]
    rsi_adjust: Dict[str, float]
    grid_enable: bool
    spacing_mode: str
    fixed_spacing_pct: float
    levels_per_side: int
    degraded_levels_per_side: int
    kill_replace_enable: bool
    kill_replace_deviation_pct: float
    kill_replace_min_seconds: int


class GridEngine:
    """
    Manages dynamic spread calculation and grid level generation.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize Grid Engine.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
        # Load Spread Engine config
        spread_cfg = config.get_spread_engine_config()
        self.spread_mode = spread_cfg.get('mode', 'dynamic')
        self.fixed_spread_pct = spread_cfg.get('fixed_spread_pct', 0.0034)
        self.bands = spread_cfg.get('bands', {})
        self.rsi_adjust = spread_cfg.get('rsi_adjust', {})
        
        # Load Grid config
        grid_cfg = config.get_grid_config()
        self.grid_enable = grid_cfg.get('enable', True)
        self.spacing_mode = grid_cfg.get('spacing_mode', 'use_band_median')
        self.fixed_spacing_pct = grid_cfg.get('fixed_spacing_pct', 0.0034)
        self.levels_per_side = grid_cfg.get('levels_per_side', 7)
        self.degraded_levels_per_side = grid_cfg.get('degraded_levels_per_side', 3)
        
        kill_replace_cfg = grid_cfg.get('kill_replace', {})
        self.kill_replace_enable = kill_replace_cfg.get('enable', True)
        self.kill_replace_deviation_pct = kill_replace_cfg.get('price_deviation_pct', 0.011)
        self.kill_replace_min_seconds = kill_replace_cfg.get('min_seconds_between', 120)
    
    def get_volatility_band(self, atr_pct: float, rsi: float) -> Tuple[str, Dict]:
        """
        Determine the current volatility band based on ATR% and RSI.
        
        Args:
            atr_pct: Current ATR as a percentage of price
            rsi: Current RSI value
            
        Returns:
            Tuple of (band_name, band_config)
        """
        if self.spread_mode != 'dynamic':
            return 'fixed', {}
        
        # Check 'far' band first (highest volatility)
        far_cfg = self.bands.get('far', {})
        if far_cfg:
            atr_min = far_cfg.get('atr_pct_min', 0.0048)
            rsi_min = far_cfg.get('rsi_soft_min', 25)
            rsi_max = far_cfg.get('rsi_soft_max', 75)
            
            if atr_pct >= atr_min or (rsi <= rsi_min or rsi >= rsi_max):
                return 'far', far_cfg
        
        # Check 'mid' band
        mid_cfg = self.bands.get('mid', {})
        if mid_cfg:
            atr_min = mid_cfg.get('atr_pct_min', 0.0031)
            atr_max = mid_cfg.get('atr_pct_max', 0.0048)
            rsi_min = mid_cfg.get('rsi_soft_min', 30)
            rsi_max = mid_cfg.get('rsi_soft_max', 70)
            
            if (atr_min <= atr_pct < atr_max) or (rsi_min <= rsi <= rsi_max):
                return 'mid', mid_cfg
        
        # Default to 'near' band (lowest volatility)
        near_cfg = self.bands.get('near', {})
        if near_cfg:
            return 'near', near_cfg
        
        # Fallback to fixed if bands are not configured
        return 'fixed', {}
    
    def calculate_spread(self, atr_pct: float, rsi: float) -> float:
        """
        Calculate the dynamic spread percentage.
        
        Args:
            atr_pct: Current ATR as a percentage of price
            rsi: Current RSI value
            
        Returns:
            Spread percentage (decimal)
        """
        if self.spread_mode == 'fixed':
            return self.fixed_spread_pct
        
        band_name, band_cfg = self.get_volatility_band(atr_pct, rsi)
        
        if band_name == 'fixed':
            return self.fixed_spread_pct
        
        spread_range = band_cfg.get('spread_pct', [self.fixed_spread_pct, self.fixed_spread_pct])
        
        # Simple implementation: use the median of the range
        spread = (spread_range[0] + spread_range[1]) / 2
        
        # Apply RSI adjustment
        if self.rsi_adjust:
            oversold_boost = self.rsi_adjust.get('oversold_boost', 0.0)
            overbought_relax = self.rsi_adjust.get('overbought_relax', 0.0)
            
            if rsi < 30:
                spread += oversold_boost
            elif rsi > 70:
                spread += overbought_relax
        
        return spread
    
    def calculate_grid_spacing(self, atr_pct: float, rsi: float) -> float:
        """
        Calculate the grid spacing percentage.
        
        Args:
            atr_pct: Current ATR as a percentage of price
            rsi: Current RSI value
            
        Returns:
            Grid spacing percentage (decimal)
        """
        if self.spacing_mode == 'fixed_spacing_pct':
            return self.fixed_spacing_pct
        
        if self.spacing_mode == 'use_band_median':
            band_name, band_cfg = self.get_volatility_band(atr_pct, rsi)
            
            if band_name == 'fixed':
                return self.fixed_spacing_pct
            
            spread_range = band_cfg.get('spread_pct', [self.fixed_spacing_pct, self.fixed_spacing_pct])
            
            # Use the median of the spread range as spacing
            return (spread_range[0] + spread_range[1]) / 2
        
        return self.fixed_spacing_pct
    
    def generate_grid_levels(
        self,
        current_price: float,
        atr_pct: float,
        rsi: float,
        pnl_gate_status: str
    ) -> List[float]:
        """
        Generate grid levels (prices) around the current price.
        
        Args:
            current_price: Current market price
            atr_pct: Current ATR as a percentage of price
            rsi: Current RSI value
            pnl_gate_status: Current PnL Gate status ('NORMAL', 'DEGRADED', 'PAUSED')
            
        Returns:
            List of grid prices (excluding current price)
        """
        if not self.grid_enable or pnl_gate_status == 'PAUSED':
            return []
        
        spacing_pct = self.calculate_grid_spacing(atr_pct, rsi)
        
        if pnl_gate_status == 'DEGRADED':
            levels = self.degraded_levels_per_side
        else:
            levels = self.levels_per_side
        
        grid_prices = []
        
        # Generate buy levels (below current price)
        for i in range(1, levels + 1):
            buy_price = current_price * (1 - i * spacing_pct)
            grid_prices.append(buy_price)
        
        # Generate sell levels (above current price)
        for i in range(1, levels + 1):
            sell_price = current_price * (1 + i * spacing_pct)
            grid_prices.append(sell_price)
        
        return sorted(grid_prices)
    
    def get_grid_levels_count(self, pnl_gate_status: str) -> int:
        """Get the number of grid levels per side."""
        if pnl_gate_status == 'DEGRADED':
            return self.degraded_levels_per_side
        return self.levels_per_side
    
    def get_kill_replace_deviation(self) -> float:
        """Get the kill-replace price deviation percentage."""
        return self.kill_replace_deviation_pct
    
    def get_kill_replace_min_seconds(self) -> int:
        """Get the kill-replace minimum seconds between replacements."""
        return self.kill_replace_min_seconds
    
    def is_kill_replace_enabled(self) -> bool:
        """Check if kill-replace is enabled."""
        return self.kill_replace_enable
