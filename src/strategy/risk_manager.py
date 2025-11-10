"""
Risk Manager
Implements risk management logic including PnL Gate, Stop Loss Engine, and position sizing.
"""

import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class PnLGateState:
    """PnL Gate state container."""
    status: str  # 'NORMAL', 'DEGRADED', 'PAUSED'
    daily_pnl_pct: float
    gap_pct: float
    max_drawdown_pct: float
    day_open_price: float
    day_high_price: float
    last_reset: datetime


@dataclass
class StopLossState:
    """Stop Loss Engine state container."""
    consecutive_losses: int
    daily_loss_count: int
    hard_stop_triggered: bool
    kill_switch_triggered: bool
    last_reset: datetime


class RiskManager:
    """
    Manages risk controls including PnL gates, stop losses, and position sizing.
    """
    
    def __init__(self, config: dict, starting_capital: float):
        """
        Initialize risk manager.
        
        Args:
            config: Risk management configuration
            starting_capital: Starting capital in quote currency
        """
        self.config = config
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        
        # PnL Gate config
        self.pnl_gate_config = config.get('pnl_gate', {})
        self.pnl_gate_enabled = self.pnl_gate_config.get('enable', True)
        
        # Stop Loss config
        self.sl_config = config.get('sl_engine', {})
        self.sl_enabled = self.sl_config.get('enable', True)
        
        # DCA config
        self.dca_config = config.get('dca', {})
        self.dca_enabled = self.dca_config.get('enable', True)
        
        # Initialize states
        self.pnl_gate_state = PnLGateState(
            status='NORMAL',
            daily_pnl_pct=0.0,
            gap_pct=0.0,
            max_drawdown_pct=0.0,
            day_open_price=0.0,
            day_high_price=0.0,
            last_reset=datetime.now()
        )
        
        self.sl_state = StopLossState(
            consecutive_losses=0,
            daily_loss_count=0,
            hard_stop_triggered=False,
            kill_switch_triggered=False,
            last_reset=datetime.now()
        )
        
        self.peak_capital = starting_capital
        self.day_start_capital = starting_capital
    
    def reset_daily_states(self, current_price: float):
        """
        Reset daily states (called at UTC midnight or start of new trading day).
        
        Args:
            current_price: Current market price
        """
        self.pnl_gate_state.daily_pnl_pct = 0.0
        self.pnl_gate_state.gap_pct = 0.0
        self.pnl_gate_state.day_open_price = current_price
        self.pnl_gate_state.day_high_price = current_price
        self.pnl_gate_state.status = 'NORMAL'
        self.pnl_gate_state.last_reset = datetime.now()
        
        self.sl_state.daily_loss_count = 0
        self.sl_state.hard_stop_triggered = False
        self.sl_state.last_reset = datetime.now()
        
        self.day_start_capital = self.current_capital
    
    def update_pnl_gate(
        self,
        current_price: float,
        position_value: float,
        realized_pnl: float
    ) -> str:
        """
        Update PnL Gate state based on current market conditions.
        
        Args:
            current_price: Current market price
            position_value: Current position value in quote currency
            realized_pnl: Realized PnL for the day
            
        Returns:
            Gate status: 'NORMAL', 'DEGRADED', or 'PAUSED'
        """
        if not self.pnl_gate_enabled:
            return 'NORMAL'
        
        # Update day high
        if current_price > self.pnl_gate_state.day_high_price:
            self.pnl_gate_state.day_high_price = current_price
        
        # Calculate daily PnL percentage
        total_value = self.current_capital + position_value
        daily_pnl = total_value - self.day_start_capital
        self.pnl_gate_state.daily_pnl_pct = daily_pnl / self.day_start_capital
        
        # Calculate gap from day high
        if self.pnl_gate_state.day_high_price > 0:
            self.pnl_gate_state.gap_pct = (
                self.pnl_gate_state.day_high_price - current_price
            ) / self.pnl_gate_state.day_high_price
        
        # Calculate drawdown
        if total_value > self.peak_capital:
            self.peak_capital = total_value
        
        drawdown = (self.peak_capital - total_value) / self.peak_capital
        self.pnl_gate_state.max_drawdown_pct = max(self.pnl_gate_state.max_drawdown_pct, drawdown)
        
        # Check pause conditions
        pause_daily_pnl = self.pnl_gate_config.get('pause_daily_pnl_pct', -0.0050)
        pause_gap = self.pnl_gate_config.get('pause_gap_pct', 0.0070)
        max_drawdown = self.pnl_gate_config.get('drawdown_pause', {}).get('max_drawdown_pct', 0.0065)
        
        if (self.pnl_gate_state.daily_pnl_pct <= pause_daily_pnl or
            self.pnl_gate_state.gap_pct >= pause_gap or
            self.pnl_gate_state.max_drawdown_pct >= max_drawdown):
            self.pnl_gate_state.status = 'PAUSED'
            return 'PAUSED'
        
        # Check degrade conditions
        degrade_daily_pnl = self.pnl_gate_config.get('degrade_daily_pnl_pct', -0.0040)
        degrade_gap = self.pnl_gate_config.get('degrade_gap_pct', 0.0040)
        
        if (self.pnl_gate_state.daily_pnl_pct <= degrade_daily_pnl or
            self.pnl_gate_state.gap_pct >= degrade_gap):
            self.pnl_gate_state.status = 'DEGRADED'
            return 'DEGRADED'
        
        self.pnl_gate_state.status = 'NORMAL'
        return 'NORMAL'
    
    def check_stop_loss(
        self,
        trade_pnl: float,
        is_loss: bool
    ) -> Tuple[bool, str]:
        """
        Check stop loss conditions.
        
        Args:
            trade_pnl: PnL from last trade
            is_loss: Whether last trade was a loss
            
        Returns:
            Tuple of (should_stop, reason)
        """
        if not self.sl_enabled:
            return False, ""
        
        capital_sl = self.sl_config.get('capital_sl', {})
        kill_switch = self.sl_config.get('kill_switch', {})
        
        # Update consecutive losses
        if is_loss:
            self.sl_state.consecutive_losses += 1
            self.sl_state.daily_loss_count += 1
        else:
            self.sl_state.consecutive_losses = 0
        
        # Check kill switch (consecutive losses)
        if kill_switch.get('enable', True):
            max_consecutive = kill_switch.get('max_consecutive_losses', 3)
            if self.sl_state.consecutive_losses >= max_consecutive:
                self.sl_state.kill_switch_triggered = True
                return True, f"Kill switch: {self.sl_state.consecutive_losses} consecutive losses"
        
        # Check hard stop (capital loss)
        if capital_sl.get('enable', True):
            hard_stop_pnl = capital_sl.get('hard_stop_daily_pnl_pct', -0.0070)
            
            if self.pnl_gate_state.daily_pnl_pct <= hard_stop_pnl:
                self.sl_state.hard_stop_triggered = True
                return True, f"Hard stop: Daily PnL {self.pnl_gate_state.daily_pnl_pct:.2%}"
        
        return False, ""
    
    def should_force_sell(
        self,
        current_price: float,
        position_size: float
    ) -> Tuple[bool, float, str]:
        """
        Check if force sell should be triggered.
        
        Args:
            current_price: Current market price
            position_size: Current position size in base currency
            
        Returns:
            Tuple of (should_sell, sell_quantity, reason)
        """
        # Force sell on pause
        if self.pnl_gate_state.status == 'PAUSED':
            force_sell_config = self.pnl_gate_config.get('force_sell', {})
            
            if force_sell_config.get('enable', True):
                price_drop_threshold = force_sell_config.get('price_drop_threshold_pct', 0.0030)
                
                if self.pnl_gate_state.gap_pct >= price_drop_threshold:
                    max_position_pct = force_sell_config.get('max_position_pct', 1.0)
                    sell_qty = position_size * max_position_pct
                    
                    return True, sell_qty, "Force sell on PnL Gate pause"
        
        # Force sell on hard stop
        if self.sl_state.hard_stop_triggered:
            force_sell_config = self.sl_config.get('capital_sl', {}).get('force_sell_on_stop', {})
            
            if force_sell_config.get('enable', True):
                sell_pct = force_sell_config.get('sell_percentage', 100.0) / 100.0
                sell_qty = position_size * sell_pct
                
                return True, sell_qty, "Force sell on hard stop"
        
        return False, 0.0, ""
    
    def calculate_position_size(
        self,
        current_price: float,
        order_size_quote: float,
        volatility_band: str = 'mid'
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            current_price: Current market price
            order_size_quote: Base order size in quote currency
            volatility_band: Current volatility band ('near', 'mid', 'far')
            
        Returns:
            Position size in base currency
        """
        # Adjust order size based on PnL Gate status
        if self.pnl_gate_state.status == 'DEGRADED':
            order_size_quote *= 0.5  # Reduce size by 50%
        elif self.pnl_gate_state.status == 'PAUSED':
            return 0.0  # No new positions
        
        # Calculate base quantity
        base_qty = order_size_quote / current_price
        
        return base_qty
    
    def can_open_position(self) -> Tuple[bool, str]:
        """
        Check if new position can be opened.
        
        Returns:
            Tuple of (can_open, reason)
        """
        if self.pnl_gate_state.status == 'PAUSED':
            return False, "PnL Gate paused"
        
        if self.sl_state.hard_stop_triggered:
            return False, "Hard stop triggered"
        
        if self.sl_state.kill_switch_triggered:
            return False, "Kill switch triggered"
        
        return True, ""
    
    def can_dca(self, current_rsi: float, current_price: float, last_fill_price: float) -> Tuple[bool, str]:
        """
        Check if DCA (Dollar Cost Averaging) is allowed.
        
        Args:
            current_rsi: Current RSI value
            current_price: Current market price
            last_fill_price: Last fill price
            
        Returns:
            Tuple of (can_dca, reason)
        """
        if not self.dca_enabled:
            return False, "DCA disabled"
        
        if self.pnl_gate_state.status == 'PAUSED':
            return False, "PnL Gate paused"
        
        # Check RSI threshold
        rsi_threshold = self.dca_config.get('rsi_threshold_buy', 38)
        if current_rsi >= rsi_threshold:
            return False, f"RSI {current_rsi:.1f} >= threshold {rsi_threshold}"
        
        # Check minimum distance from last fill
        if last_fill_price > 0:
            min_distance = self.dca_config.get('safety_guards', {}).get('min_distance_from_last_fill_pct', 0.0055)
            price_distance = abs(current_price - last_fill_price) / last_fill_price
            
            if price_distance < min_distance:
                return False, f"Too close to last fill: {price_distance:.2%} < {min_distance:.2%}"
        
        return True, ""
    
    def update_capital(self, realized_pnl: float):
        """
        Update current capital after trade.
        
        Args:
            realized_pnl: Realized PnL from trade
        """
        self.current_capital += realized_pnl
    
    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'day_start_capital': self.day_start_capital,
            'pnl_gate_status': self.pnl_gate_state.status,
            'daily_pnl_pct': self.pnl_gate_state.daily_pnl_pct,
            'gap_pct': self.pnl_gate_state.gap_pct,
            'max_drawdown_pct': self.pnl_gate_state.max_drawdown_pct,
            'consecutive_losses': self.sl_state.consecutive_losses,
            'hard_stop_triggered': self.sl_state.hard_stop_triggered,
            'kill_switch_triggered': self.sl_state.kill_switch_triggered,
        }

    def is_hard_stop_active(self) -> bool:
        """
        Check if a hard stop condition is active (either hard stop or kill switch).
        """
        return self.sl_state.hard_stop_triggered or self.sl_state.kill_switch_triggered

    def get_pnl_gate_status(self) -> str:
        """
        Get the current PnL Gate status.
        """
        return self.pnl_gate_state.status

    def reset_consecutive_losses(self):
        """
        Reset the consecutive loss counter. Called after a profitable trade.
        """
        self.sl_state.consecutive_losses = 0
