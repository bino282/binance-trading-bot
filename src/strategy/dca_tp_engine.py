"""
DCA and Trailing TP Engine
Implements Dollar Cost Averaging and Volatility-Adjusted Trailing Take Profit logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..utils.config_loader import ConfigLoader


@dataclass
class DCAState:
    """DCA state container."""
    count: int = 0
    last_fill_price: float = 0.0
    cooldown_bars: int = 0


class DCATPEngine:
    """
    Manages DCA and Trailing Take Profit logic.
    """
    
    def __init__(self, config: ConfigLoader):
        """
        Initialize DCA/TP Engine.
        
        Args:
            config: Configuration loader instance
        """
        self.config = config
        
        # DCA config
        self.dca_cfg = config.get_dca_config()
        self.dca_enable = self.dca_cfg.get('enable', True)
        self.rsi_threshold_buy = self.dca_cfg.get('rsi_threshold_buy', 38)
        self.max_steps = self.dca_cfg.get('allocation_plan', {}).get('steps_max', 3)
        self.steps_pct = self.dca_cfg.get('allocation_plan', {}).get('steps_pct', [0.18, 0.20, 0.25])
        self.cooldown_bars_cfg = self.dca_cfg.get('allocation_plan', {}).get('cooldown_bars', 6)
        self.min_distance_pct = self.dca_cfg.get('safety_guards', {}).get('min_distance_from_last_fill_pct', 0.0055)
        
        # Trailing TP config
        self.tp_cfg = config.get_tp_trailing_config()
        self.tp_enable = self.tp_cfg.get('enable', True)
        self.tp_bands = self.tp_cfg.get('by_band', {})
        self.min_hold_bars = self.tp_cfg.get('min_hold_bars', 3)
        
        # State
        self.dca_state = DCAState()
    
    def reset_dca_state(self):
        """Reset DCA state after position is closed."""
        self.dca_state = DCAState()
    
    def update_dca_fill(self, fill_price: float):
        """Update DCA state after a fill."""
        self.dca_state.count += 1
        self.dca_state.last_fill_price = fill_price
        self.dca_state.cooldown_bars = self.cooldown_bars_cfg
    
    def update_dca_cooldown(self):
        """Decrement DCA cooldown bars."""
        if self.dca_state.cooldown_bars > 0:
            self.dca_state.cooldown_bars -= 1
    
    def get_dca_order_size_pct(self) -> float:
        """Get the percentage of the order size for the next DCA step."""
        if self.dca_state.count < len(self.steps_pct):
            return self.steps_pct[self.dca_state.count]
        return 0.0
    
    def can_dca(
        self,
        current_rsi: float,
        current_price: float,
        pnl_gate_status: str
    ) -> Tuple[bool, str]:
        """
        Check if DCA is allowed based on strategy rules.
        
        Args:
            current_rsi: Current RSI value
            current_price: Current market price
            pnl_gate_status: Current PnL Gate status
            
        Returns:
            Tuple of (can_dca, reason)
        """
        if not self.dca_enable:
            return False, "DCA disabled"
        
        if pnl_gate_status == 'PAUSED':
            return False, "PnL Gate paused"
        
        if self.dca_state.count >= self.max_steps:
            return False, f"Max DCA steps ({self.max_steps}) reached"
        
        if self.dca_state.cooldown_bars > 0:
            return False, f"DCA cooldown active ({self.dca_state.cooldown_bars} bars remaining)"
        
        # Check RSI threshold
        if current_rsi >= self.rsi_threshold_buy:
            return False, f"RSI {current_rsi:.1f} >= threshold {self.rsi_threshold_buy}"
        
        # Check minimum distance from last fill
        if self.dca_state.last_fill_price > 0:
            price_distance = abs(current_price - self.dca_state.last_fill_price) / self.dca_state.last_fill_price
            
            if price_distance < self.min_distance_pct:
                return False, f"Too close to last fill: {price_distance:.2%} < {self.min_distance_pct:.2%}"
        
        # Check EMA filter (simplified - assuming price below EMA mid is preferred)
        # The full logic from the config is complex, so we'll rely on the RSI/distance for now
        
        return True, f"DCA step {self.dca_state.count + 1}/{self.max_steps}"
    
    def get_tp_band_config(self, atr_pct: float) -> Optional[Dict]:
        """
        Determine the Trailing TP band configuration based on ATR%.
        
        Args:
            atr_pct: Current ATR as a percentage of price
            
        Returns:
            Band configuration dictionary or None
        """
        if not self.tp_enable:
            return None
        
        # Check 'far' band
        far_cfg = self.tp_bands.get('far', {})
        if far_cfg and atr_pct >= self.config.get('policy_cfg.spread_engine.bands.mid.atr_pct_max', 0.0048):
            return far_cfg
        
        # Check 'mid' band
        mid_cfg = self.tp_bands.get('mid', {})
        if mid_cfg and atr_pct >= self.config.get('policy_cfg.spread_engine.bands.near.atr_pct_max', 0.0031):
            return mid_cfg
        
        # Default to 'near' band
        near_cfg = self.tp_bands.get('near', {})
        if near_cfg:
            return near_cfg
        
        return None
    
    def should_trail_tp(
        self,
        current_price: float,
        avg_entry_price: float,
        atr_pct: float,
        bars_since_entry: int
    ) -> Tuple[bool, float, float]:
        """
        Check if Trailing Take Profit should be triggered.
        
        Args:
            current_price: Current market price
            avg_entry_price: Average entry price of the position
            atr_pct: Current ATR as a percentage of price
            bars_since_entry: Number of bars since the initial entry
            
        Returns:
            Tuple of (should_trail, trigger_pct, trail_pct)
        """
        if not self.tp_enable:
            return False, 0.0, 0.0
        
        if bars_since_entry < self.min_hold_bars:
            return False, 0.0, 0.0
        
        band_cfg = self.get_tp_band_config(atr_pct)
        
        if not band_cfg:
            return False, 0.0, 0.0
        
        trigger_pct = band_cfg.get('trigger_pct', 0.0)
        trail_pct = band_cfg.get('trail_pct', 0.0)
        
        # Calculate unrealized PnL percentage
        unrealized_pnl_pct = (current_price - avg_entry_price) / avg_entry_price
        
        # Check if trigger is hit
        if unrealized_pnl_pct >= trigger_pct:
            return True, trigger_pct, trail_pct
        
        return False, 0.0, 0.0
