
"""
Backtesting Engine
Simulates trading strategy on historical data with realistic order execution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..indicators.technical import add_all_indicators
from ..indicators.score_engine import ScoreEngine, ScenarioScore
from ..indicators.mtf_analysis import MTFAnalysis
from ..strategy.risk_manager import RiskManager
from ..strategy.grid_engine import GridEngine
from ..strategy.dca_tp_engine import DCATPEngine
from ..strategy.risk_manager import RiskManager
from ..utils.logger import get_logger


@dataclass
class Trade:
    """Trade record."""
    timestamp: datetime
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    value: float
    fee: float
    pnl: float = 0.0
    scenario: str = ""
    score: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    """Backtest result container."""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: Dict = field(default_factory=dict)
    daily_pnl: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'trades': [vars(t) for t in self.trades],
            'equity_curve': self.equity_curve.to_dict(),
            'metrics': self.metrics,
            'daily_pnl': self.daily_pnl.to_dict()
        }


class BacktestEngine:
    """
    Backtesting engine for trading strategy.
    Simulates realistic order execution with fees and slippage.
    """
    
    def __init__(self, config: dict):
        """
        Initialize backtest engine.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.logger = get_logger('backtest', level='INFO')
        
        # Trading parameters
        self.symbol = config.get('pair', {}).get('symbol', 'ZECUSDT')
        self.starting_capital = config.get('starting_cash_usdt', 3000.0)
        self.order_size_quote = config.get('order_size_quote_usdt', 90.0)
        self.fee_per_leg = config.get('fee_per_leg_pct', 0.001)
        
        # Market filters
        self.tick_size = config.get('market_filters', {}).get('tick_size', 0.01)
        self.lot_size = config.get('market_filters', {}).get('lot_size', 0.001)
        self.min_notional = config.get('market_filters', {}).get('min_notional', 5.0)
        
        # Initialize components
        indicators_config = config.get('policy_cfg', {}).get('indicators', {})
        score_config = config.get('policy_cfg', {}).get('score_layer', {})
        risk_config = config.get('policy_cfg', {})
        
        self.score_engine = ScoreEngine(score_config)
        self.risk_manager = RiskManager(risk_config, self.starting_capital)
        self.grid_engine = GridEngine(self.config)
        self.dca_tp_engine = DCATPEngine(self.config)
        self.mtf_analysis = MTFAnalysis(self.config)
        
        # State variables
        self.position_size = 0.0  # Base currency
        self.position_value = 0.0  # Quote currency
        self.avg_entry_price = 0.0
        self.cash = self.starting_capital
        self.equity = self.starting_capital
        
        self.trades: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float, float, float]] = []
        
        self.last_fill_price = 0.0
        self.bars_since_entry = 0
        self.trailing_stop_price = 0.0
        self.htf_data: Dict[str, pd.DataFrame] = {}
        
        self.logger.info(f"Backtest engine initialized for {self.symbol}")
        self.logger.info(f"Starting capital: ${self.starting_capital:.2f}")
    
    def round_price(self, price: float) -> float:
        """Round price to tick size."""
        return round(price / self.tick_size) * self.tick_size
    
    def round_quantity(self, quantity: float) -> float:
        """Round quantity to lot size."""
        return round(quantity / self.lot_size) * self.lot_size
    
    def calculate_fee(self, value: float) -> float:
        """Calculate trading fee."""
        return value * self.fee_per_leg
    
    def execute_buy(
        self,
        price: float,
        timestamp: datetime,
        order_size: Optional[float] = None,
        scenario: str = "",
        score: float = 0.0,
        reason: str = ""
    ) -> Optional[Trade]:
        """
        Execute buy order.
        
        Args:
            price: Execution price
            timestamp: Order timestamp
            scenario: Trading scenario
            score: Signal score
            reason: Trade reason
            
        Returns:
            Trade object if executed, None otherwise
        """
        # Calculate quantity
        if order_size is None:
            order_size = self.order_size_quote
            
        quantity = order_size / price
        quantity = self.round_quantity(quantity)
        
        if quantity <= 0:
            return None
        
        # Calculate value and fee
        value = quantity * price
        fee = self.calculate_fee(value)
        total_cost = value + fee
        
        # Check if we have enough cash
        if total_cost > self.cash:
            self.logger.warning(f"Insufficient cash: ${self.cash:.2f} < ${total_cost:.2f}")
            return None
        
        # Check min notional
        if value < self.min_notional:
            self.logger.warning(f"Order below min notional: ${value:.2f} < ${self.min_notional:.2f}")
            return None
        
        # Update position
        total_quantity = self.position_size + quantity
        total_cost_basis = (self.position_size * self.avg_entry_price) + value
        self.avg_entry_price = total_cost_basis / total_quantity if total_quantity > 0 else 0
        
        self.position_size = total_quantity
        self.cash -= total_cost
        self.last_fill_price = price
        
        # Update DCA state
        if self.position_size > 0:
            self.dca_tp_engine.update_dca_fill(price)
            self.bars_since_entry = 0
            self.trailing_stop_price = 0.0
        
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            side='BUY',
            price=price,
            quantity=quantity,
            value=value,
            fee=fee,
            scenario=scenario,
            score=score,
            reason=reason
        )
        
        self.trades.append(trade)
        self.logger.trade('BUY', price, quantity, reason)
        
        return trade
    
    def execute_sell(
        self,
        price: float,
        timestamp: datetime,
        quantity: Optional[float] = None,
        scenario: str = "",
        score: float = 0.0,
        reason: str = ""
    ) -> Optional[Trade]:
        """
        Execute sell order.
        
        Args:
            price: Execution price
            timestamp: Order timestamp
            quantity: Quantity to sell (None = sell all)
            scenario: Trading scenario
            score: Signal score
            reason: Trade reason
            
        Returns:
            Trade object if executed, None otherwise
        """
        if self.position_size <= 0:
            return None
        
        # Default to selling entire position
        if quantity is None:
            quantity = self.position_size
        else:
            # Ensure quantity is a float before comparison
            quantity = float(quantity)
            quantity = min(quantity, self.position_size)
        
        quantity = self.round_quantity(quantity)
        
        if quantity <= 0:
            return None
        
        # Calculate value and fee
        value = quantity * price
        fee = self.calculate_fee(value)
        net_proceeds = value - fee
        
        # Calculate PnL
        cost_basis = quantity * self.avg_entry_price
        pnl = net_proceeds - cost_basis
        
        # Update position
        self.position_size -= quantity
        self.cash += net_proceeds
        
        if self.position_size <= 0:
            self.position_size = 0.0
            self.avg_entry_price = 0.0
            self.dca_tp_engine.reset_dca_state()
            self.bars_since_entry = 0
            self.trailing_stop_price = 0.0
        
        # Update risk manager
        self.risk_manager.update_capital(pnl)
        is_loss = pnl < 0
        self.risk_manager.check_stop_loss(pnl, is_loss)
        
        if not is_loss:
            self.risk_manager.reset_consecutive_losses()
        self.dca_tp_engine.reset_dca_state()
        self.trailing_stop_price = 0.0
        
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            side='SELL',
            price=price,
            quantity=quantity,
            value=value,
            fee=fee,
            pnl=pnl,
            scenario=scenario,
            score=score,
            reason=reason
        )
        
        self.trades.append(trade)
        self.logger.trade('SELL', price, quantity, f"{reason} | PnL: ${pnl:.2f}")
        
        return trade
    
    def update_equity(self, current_price: float, timestamp: datetime):
        """Update equity and position value."""
        self.position_value = self.position_size * current_price
        self.equity = self.cash + self.position_value
        
        # Record equity history
        unrealized_pnl = 0.0
        if self.position_size > 0:
            unrealized_pnl = (current_price - self.avg_entry_price) * self.position_size
        
        self.equity_history.append((timestamp, self.equity, self.cash, unrealized_pnl))
    
    def run(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: Historical data
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            BacktestResult object
        """
        self.logger.info("Starting backtest...")
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        # Prepare data
        data = self._prepare_data(data)
        
        # Prepare HTF data
        self._prepare_htf_data(data)
        
        # Warmup period
        warmup_bars = self.config.get('warmup_bars', 200)
        if len(data) <= warmup_bars:
            self.logger.warning("Not enough data for backtest after warmup period.")
            return self._calculate_results(data) # Return empty result
            
        self.logger.info(f"Trading period: {data.index[warmup_bars]} to {data.index[-1]}")
        
        # Initialize day tracking
        current_day = None
        
        # Main backtest loop
        try:
            for i in range(warmup_bars, len(data)):
                row = data.iloc[i]
                timestamp = data.index[i]
                current_price = row['close']
                
                # 1. Update equity
                self.update_equity(current_price, timestamp)
                
                # 2. Check for daily reset
                bar_day = timestamp.date()
                if current_day != bar_day:
                    self.risk_manager.reset_daily_states(current_price)
                    current_day = bar_day
                
                # 3. Check for hard stop
                if self.risk_manager.is_hard_stop_active():
                    if self.position_size > 0:
                        self.execute_sell(current_price, timestamp, quantity=None, reason="Hard Stop / Kill Switch")
                    self.logger.warning("Hard Stop Active. Trading paused for the day.")
                    continue
                
                # 4. Get PnL gate status
                pnl_gate_status = self.risk_manager.get_pnl_gate_status()
                if pnl_gate_status == 'PAUSED':
                    self.logger.warning("PnL Gate PAUSED. No new entries.")
            
                # 5. Get HTF analysis for scoring
                htf_analysis = self._get_htf_analysis(timestamp)
                
                # 6. Generate trading signal
                signal, best_scenario = self.score_engine.generate_signal(
                    row,
                    i,
                    self.position_size,
                    htf_analysis
                )
                
                # Log signal details if enabled
                if best_scenario and self.config.get('score_layer', {}).get('log_detailed_scores', False):
                    if signal in ['entry', 'exit']:
                        self.logger.signal(
                            signal,
                            best_scenario.score,
                            best_scenario.scenario_id,
                            f"Thresholds: E={best_scenario.entry_threshold} H={best_scenario.hold_threshold} X={best_scenario.exit_threshold}"
                        )
                
                # 7. Execute trading logic
                if self.position_size == 0:
                    # Entry logic
                    if signal == 'entry' and pnl_gate_status != 'PAUSED':
                        # Check if we can open position
                        can_open, reason = self.risk_manager.can_open_position()
                        
                        if can_open:
                            # Adjust order size for DEGRADED state
                            order_size = self.order_size_quote
                            reason_suffix = ""
                            if pnl_gate_status == 'DEGRADED':
                                order_size *= 0.5
                                reason_suffix = " (DEGRADED)"
                                
                            trade = self.execute_buy(
                                current_price,
                                timestamp,
                                order_size=order_size,
                                scenario=best_scenario.scenario_id if best_scenario else "",
                                score=best_scenario.score if best_scenario else 0.0,
                                reason=f"Entry signal: {best_scenario.scenario_id if best_scenario else 'N/A'}{reason_suffix}"
                            )
                            
                            if trade:
                                self.bars_since_entry = 0
                                self.risk_manager.reset_consecutive_losses()
                                self.logger.info(f"Position opened: {best_scenario.scenario_id}")
                        else:
                            self.logger.warning(f"Cannot open position: {reason}")
                    
                    # Grid trading logic (simplified for now)
                    # Grid logic is complex and usually runs in parallel or as a separate module.
                    # For backtesting, we'll focus on the main signal.
                    
                elif self.position_size > 0:
                    # 7.1. Check Hard Stop Loss
                    should_stop_loss_now, sl_reason = self.risk_manager.should_stop_loss(
                        current_price, self.avg_entry_price, self.position_size
                    )
                    if should_stop_loss_now:
                        self.execute_sell(current_price, timestamp, quantity=None, reason=sl_reason)
                        continue # Use continue instead of return to allow finalization
                        
                    # 7.2. Check Signal Exit
                    if signal == 'exit':
                        self.execute_sell(
                            current_price,
                            timestamp,
                            None,
                            best_scenario.scenario_id if best_scenario else "",
                            best_scenario.score if best_scenario else 0.0,
                            f"Exit signal: {best_scenario.scenario_id if best_scenario else 'N/A'}"
                        )
                        continue
                    
                    # 7.3. Check Trailing Take Profit
                    should_trail, trigger_pct, trail_pct = self.dca_tp_engine.should_trail_tp(
                        current_price, self.avg_entry_price, row['atr_pct'], self.bars_since_entry
                    )
                    
                    if should_trail:
                        # Initialize trailing stop
                        if self.trailing_stop_price == 0.0:
                            self.trailing_stop_price = current_price * (1 - trail_pct)
                            self.logger.info(f"Trailing TP activated. Stop: {self.trailing_stop_price:.2f}")
                        
                        # Update trailing stop
                        new_stop = current_price * (1 - trail_pct)
                        self.trailing_stop_price = max(self.trailing_stop_price, new_stop)
                        
                        # Check for stop hit
                        if current_price < self.trailing_stop_price:
                            self.execute_sell(
                                current_price, timestamp, quantity=None,
                                reason=f"Trailing TP Hit (Stop: {self.trailing_stop_price:.2f})"
                            )
                            self.trailing_stop_price = 0.0
                            continue
                    
                    # 7.4. Check DCA opportunity
                    can_dca, dca_reason = self.dca_tp_engine.can_dca(
                        row['rsi'], current_price, pnl_gate_status
                    )
                    
                    if can_dca:
                        # Execute DCA buy
                        order_size_pct = self.dca_tp_engine.get_dca_order_size_pct()
                        dca_order_size = self.order_size_quote * order_size_pct
                        
                        trade = self.execute_buy(
                            current_price,
                            timestamp,
                            order_size=dca_order_size,
                            scenario=best_scenario.scenario_id if best_scenario else "",
                            score=best_scenario.score if best_scenario else 0.0,
                            reason=f"DCA Step {self.dca_tp_engine.dca_state.count + 1} - {dca_reason}"
                        )
                        
                        if trade:
                            self.dca_tp_engine.update_dca_fill(current_price)
                            self.trailing_stop_price = 0.0 # Reset TP on DCA
                            
                    # 7.5. Update position state
                    self.bars_since_entry += 1
                    self.dca_tp_engine.update_dca_cooldown()
                    
                    # 7.6. Update PnL gate for next bar
                    unrealized_pnl = (current_price - self.avg_entry_price) * self.position_size
                    self.risk_manager.update_pnl_gate(self.equity, unrealized_pnl, current_price)
        except Exception as e:
            self.logger.error(f"Critical error during backtest loop: {e}")
            self.logger.error("Returning partial results due to error.")
            # Fall through to finalization and return partial result
            
        # Close any remaining position at end
        if self.position_size > 0:
            final_price = data.iloc[-1]['close']
            final_timestamp = data.index[-1]
            self.execute_sell(
                final_price,
                final_timestamp,
                None,
                reason="End of backtest"
            )
        
        # Calculate metrics
        result = self._calculate_results(data)
        
        self.logger.info("=" * 80)
        self.logger.info("Backtest completed!")
        
        return result
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data by adding indicators."""
        self.logger.info("Preparing data and calculating indicators...")
        indicators_config = self.config.get('policy_cfg', {}).get('indicators', {})
        data = add_all_indicators(data, indicators_config)
        return data
    
    def _prepare_htf_data(self, base_data: pd.DataFrame):
        """Prepare higher timeframe data."""
        self.logger.info("Preparing higher timeframe data...")
        self.htf_data = self.mtf_analysis.prepare_all_htf_data(base_data)
        
    def _get_htf_analysis(self, timestamp: datetime) -> Dict[str, pd.Series]:
        """Get HTF analysis for a given timestamp."""
        htf_analysis = {}
        for tf, df in self.htf_data.items():
            if timestamp in df.index:
                htf_analysis[tf] = df.loc[timestamp]
        return htf_analysis
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """
        Calculate backtest results and metrics.
        
        Args:
            data: Historical data
            
        Returns:
            BacktestResult object
        """
        if not self.equity_history:
            return BacktestResult()
            
        # Create equity curve
        equity_df = pd.DataFrame(
            self.equity_history, 
            columns=["timestamp", "equity", "cash", "unrealized_pnl"]
        )
        equity_df.set_index("timestamp", inplace=True)
        
        # Calculate daily PnL
        daily_pnl = equity_df['equity'].resample('D').last().diff().dropna()
        
        # Calculate metrics
        total_return = (self.equity - self.starting_capital) / self.starting_capital
        sharpe_ratio = self._calculate_sharpe_ratio(daily_pnl)
        max_drawdown = self._calculate_max_drawdown(equity_df['equity'])
        
        # Win/loss analysis
        trade_pnl = [t.pnl for t in self.trades if t.side == 'SELL']
        win_rate = np.mean([1 if p > 0 else 0 for p in trade_pnl]) if trade_pnl else 0
        avg_win = np.mean([p for p in trade_pnl if p > 0]) if any(p > 0 for p in trade_pnl) else 0
        avg_loss = np.mean([p for p in trade_pnl if p < 0]) if any(p < 0 for p in trade_pnl) else 0
        profit_factor = abs(sum(p for p in trade_pnl if p > 0) / sum(p for p in trade_pnl if p < 0)) if any(p < 0 for p in trade_pnl) else float('inf')
        
        metrics = {
            "Total Return": f"{total_return:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Total Trades": len([t for t in self.trades if t.side == 'SELL']),
            "Win Rate": f"{win_rate:.2%}",
            "Avg Win": f"${avg_win:.2f}",
            "Avg Loss": f"${avg_loss:.2f}",
            "Profit Factor": f"{profit_factor:.2f}",
            "Final Equity": f"${self.equity:.2f}"
        }
        
        result = BacktestResult(
            trades=self.trades,
            equity_curve=equity_df,
            metrics=metrics,
            daily_pnl=daily_pnl
        )
        
        self._log_summary(result)
        
        return result
    
    def _calculate_sharpe_ratio(self, daily_pnl: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if daily_pnl.std() == 0:
            return 0.0
        
        excess_returns = daily_pnl - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate max drawdown."""
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
    
    def _log_summary(self, result: BacktestResult):
        """Log backtest summary."""
        self.logger.info("=" * 80)
        self.logger.info("Backtest Summary")
        self.logger.info("=" * 80)
        for key, value in result.metrics.items():
            self.logger.info(f"{key:<20} {value}")
        self.logger.info("=" * 80)
