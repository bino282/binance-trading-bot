
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
class Order:
    """Order record."""
    order_id: str
    timestamp: datetime
    side: str  # 'BUY' or 'SELL'
    type: str  # 'LIMIT' or 'MARKET'
    price: float
    quantity: float
    status: str  # 'NEW', 'FILLED', 'CANCELED'
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    reason: str = ""


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
    orders: List[Order] = field(default_factory=list)
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

        # Order Management System (OMS) state
        self.order_counter = 0
        self.orders: List[Order] = []
        self.pending_orders: List[Order] = []  # Active LIMIT orders waiting for fill
        
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
    
    def _create_order(
        self,
        timestamp: datetime,
        side: str,
        order_type: str,
        price: float,
        quantity: float,
        reason: str = ""
    ) -> Order:
        """Create a new order and add it to the order list."""
        self.order_counter += 1
        order_id = f"order_{self.order_counter}"
        
        order = Order(
            order_id=order_id,
            timestamp=timestamp,
            side=side,
            type=order_type,
            price=price,
            quantity=quantity,
            status='NEW',
            reason=reason
        )
        
        self.orders.append(order)
        return order

    def round_quantity(self, quantity: float) -> float:
        """Round quantity to lot size."""
        return round(quantity / self.lot_size) * self.lot_size
    
    def calculate_fee(self, value: float) -> float:
        """Calculate trading fee."""
        return value * self.fee_per_leg
    
    def _execute_fill(
        self,
        order: Order,
        fill_price: float,
        fill_time: datetime,
        fill_quantity: float,
        scenario: str = "",
        score: float = 0.0,
    ) -> Trade:
        """Internal method to process an order fill and update backtest state."""
        
        # 1. Calculate value and fee
        value = fill_quantity * fill_price
        fee = self.calculate_fee(value)
        
        # 2. Update order status
        order.status = 'FILLED'
        order.filled_price = fill_price
        order.filled_quantity = fill_quantity
        
        # 3. Update position and cash
        if order.side == 'BUY':
            total_cost = value + fee
            
            # Check if we have enough cash (should be checked before order creation, but as a safeguard)
            if total_cost > self.cash:
                self.logger.error(f"Critical Error: Insufficient cash for fill: ${self.cash:.2f} < ${total_cost:.2f}")
                # This should not happen if initial checks are correct, but we'll proceed with negative cash for now
                # In a real backtester, this order would be rejected or partially filled.
            
            # Update position
            total_quantity = self.position_size + fill_quantity
            total_cost_basis = (self.position_size * self.avg_entry_price) + value
            self.avg_entry_price = total_cost_basis / total_quantity if total_quantity > 0 else 0
            
            self.position_size = total_quantity
            self.cash -= total_cost
            self.last_fill_price = fill_price
            
            # Update DCA state
            if self.position_size > 0:
                self.dca_tp_engine.update_dca_fill(fill_price)
                self.bars_since_entry = 0
                self.trailing_stop_price = 0.0
            
            pnl = 0.0
            net_proceeds = 0.0
            
        elif order.side == 'SELL':
            net_proceeds = value - fee
            
            # Calculate PnL
            cost_basis = fill_quantity * self.avg_entry_price
            pnl = net_proceeds - cost_basis
            
            # Update position
            self.position_size -= fill_quantity
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
            
        # 4. Create trade record
        trade = Trade(
            timestamp=fill_time,
            side=order.side,
            price=fill_price,
            quantity=fill_quantity,
            value=value,
            fee=fee,
            pnl=pnl,
            scenario=scenario,
            score=score,
            reason=order.reason
        )
        
        self.trades.append(trade)
        self.logger.trade(order.side, fill_price, fill_quantity, order.reason)
        
        return trade

    def execute_buy(
        self,
        price: float,
        timestamp: datetime,
        order_size: Optional[float] = None,
        order_type: str = 'MARKET', # New parameter
        scenario: str = "",
        score: float = 0.0,
        reason: str = ""
    ) -> Optional[Order]: # Returns Order instead of Trade
        """
        Create a buy order.
        
        Args:
            price: Limit price for LIMIT order, or current price for MARKET order
            timestamp: Order timestamp
            order_size: Quote currency amount to spend
            order_type: 'MARKET' or 'LIMIT'
            scenario: Trading scenario
            score: Signal score
            reason: Trade reason
            
        Returns:
            Order object if created, None otherwise
        """
        # Calculate quantity
        if order_size is None:
            order_size = self.order_size_quote
            
        quantity = order_size / price
        quantity = self.round_quantity(quantity)
        
        if quantity <= 0:
            return None
        
        # Check min notional
        if order_size < self.min_notional:
            self.logger.warning(f"Order below min notional: ${order_size:.2f} < ${self.min_notional:.2f}")
            return None
        
        # Check if we have enough cash for the *potential* cost
        # For simplicity, we reserve the full quote amount for the order
        if order_size > self.cash:
            self.logger.warning(f"Insufficient cash: ${self.cash:.2f} < ${order_size:.2f}")
            return None
        
        # Create the order
        order = self._create_order(
            timestamp=timestamp,
            side='BUY',
            order_type=order_type,
            price=price,
            quantity=quantity,
            reason=reason
        )
        
        if order_type == 'MARKET':
            # MARKET orders are filled immediately at the given price
            self._execute_fill(
                order=order,
                fill_price=price,
                fill_time=timestamp,
                fill_quantity=quantity,
                scenario=scenario,
                score=score
            )
            return order
        
        elif order_type == 'LIMIT':
            # LIMIT orders are added to the pending list
            self.pending_orders.append(order)
            self.logger.info(f"LIMIT BUY order created: {order.order_id} @ {order.price}")
            return order
            
        return None
    
    def execute_sell(
        self,
        price: float,
        timestamp: datetime,
        quantity: Optional[float] = None,
        order_type: str = 'MARKET', # New parameter
        scenario: str = "",
        score: float = 0.0,
        reason: str = ""
    ) -> Optional[Order]: # Returns Order instead of Trade
        """
        Create a sell order.
        
        Args:
            price: Limit price for LIMIT order, or current price for MARKET order
            timestamp: Order timestamp
            quantity: Quantity to sell (None = sell all)
            order_type: 'MARKET' or 'LIMIT'
            scenario: Trading scenario
            score: Signal score
            reason: Trade reason
            
        Returns:
            Order object if created, None otherwise
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
        
        # Check min notional (using estimated value)
        estimated_value = quantity * price
        if estimated_value < self.min_notional:
            self.logger.warning(f"Order below min notional: ${estimated_value:.2f} < ${self.min_notional:.2f}")
            return None
        
        # Create the order
        order = self._create_order(
            timestamp=timestamp,
            side='SELL',
            order_type=order_type,
            price=price,
            quantity=quantity,
            reason=reason
        )
        
        if order_type == 'MARKET':
            # MARKET orders are filled immediately at the given price
            self._execute_fill(
                order=order,
                fill_price=price,
                fill_time=timestamp,
                fill_quantity=quantity,
                scenario=scenario,
                score=score
            )
            return order
        
        elif order_type == 'LIMIT':
            # LIMIT orders are added to the pending list
            self.pending_orders.append(order)
            self.logger.info(f"LIMIT SELL order created: {order.order_id} @ {order.price}")
            return order
            
        return None
    
    def _process_pending_orders(self, bar: pd.Series, timestamp: datetime):
        """
        Process pending LIMIT orders against the current bar's high/low prices.
        
        Args:
            bar: The current OHLCV bar (pd.Series)
            timestamp: The timestamp of the current bar
        """
        filled_orders = []
        
        for order in self.pending_orders:
            if order.status != 'NEW':
                filled_orders.append(order)
                continue
                
            fill_price = None
            
            if order.side == 'BUY':
                # LIMIT BUY: Fill if bar's low <= limit_price
                if bar['low'] <= order.price:
                    # Fill at the limit price (or better, but we use limit price for simplicity)
                    fill_price = order.price
                    
            elif order.side == 'SELL':
                # LIMIT SELL: Fill if bar's high >= limit_price
                if bar['high'] >= order.price:
                    # Fill at the limit price (or better, but we use limit price for simplicity)
                    fill_price = order.price
            
            if fill_price is not None:
                # Execute the fill
                self._execute_fill(
                    order=order,
                    fill_price=fill_price,
                    fill_time=timestamp,
                    fill_quantity=order.quantity,
                    scenario="", # Order from strategy, no specific scenario/score for fill
                    score=0.0
                )
                filled_orders.append(order)
                self.logger.info(f"Order {order.order_id} FILLED at {fill_price} ({order.side} {order.type})")
        
        # Remove filled orders from pending list
        self.pending_orders = [order for order in self.pending_orders if order.status == 'NEW']
        
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
                
                # 1. Process pending orders (fills happen here)
                self._process_pending_orders(row, timestamp)

                # 2. Update equity
                self.update_equity(current_price, timestamp)
                
                # 2. Check for daily reset
                bar_day = timestamp.date()
                if current_day != bar_day:
                    self.risk_manager.reset_daily_states(current_price)
                    current_day = bar_day
                
                # 3. Check for hard stop
                if self.risk_manager.is_hard_stop_active():
                    if self.position_size > 0:
                        # Force sell is a MARKET order
                        self.execute_sell(current_price, timestamp, quantity=None, order_type='MARKET', reason="Hard Stop / Kill Switch")
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
                                
                            order = self.execute_buy(
                                current_price,
                                timestamp,
                                order_size=order_size,
                                order_type='MARKET', # Immediate fill for entry signal
                                scenario=best_scenario.scenario_id if best_scenario else "",
                                score=best_scenario.score if best_scenario else 0.0,
                                reason=f"Entry signal: {best_scenario.scenario_id if best_scenario else 'N/A'}{reason_suffix}"
                            )
                            
                            if order and order.status == 'FILLED':
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
                        # Stop loss is a MARKET order
                        self.execute_sell(current_price, timestamp, quantity=None, order_type='MARKET', reason=sl_reason)
                        continue # Use continue instead of return to allow finalization
                        
                    # 7.2. Check Signal Exit
                    if signal == 'exit':
                        self.execute_sell(
                            current_price,
                            timestamp,
                            quantity=None,
                            order_type='MARKET', # Immediate fill for exit signal
                            scenario=best_scenario.scenario_id if best_scenario else "",
                            score=best_scenario.score if best_scenario else 0.0,
                            reason=f"Exit signal: {best_scenario.scenario_id if best_scenario else 'N/A'}"
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
                                current_price, timestamp, quantity=None, order_type='MARKET',
                                reason=f"Trailing TP Hit (Stop: {self.trailing_stop_price:.2f})"
                            )
                            self.trailing_stop_price = 0.0
                            continue
                    
                    # 7.4. Check DCA opportunity
                    can_dca, dca_reason = self.dca_tp_engine.can_dca(
                        row['rsi'], current_price, pnl_gate_status
                    )
                    
                    if can_dca:
                        # DCA is a MARKET order for immediate execution
                        order = self.execute_buy(
                            current_price,
                            timestamp,
                            order_size=self.order_size_quote,
                            order_type='MARKET',
                            reason=f"DCA: {dca_reason}"
                        )
                        if order and order.status == 'FILLED':
                            self.logger.info(f"DCA executed: {dca_reason}")
                            
                    # 7.5. Check Trailing Take Profit (Update only)
                    # The trailing stop hit logic is already above, this is for updating the state
                    self.dca_tp_engine.update_trailing_tp(current_price)
                    
                    # 7.6. Check Grid Trading (Simplified for now)
                    # Grid logic is complex and usually runs in parallel or as a separate module.
                    # For backtesting, we'll focus on the main signal.
                    
                    # 7.7. Update bars since entry
                    self.bars_since_entry += 1
                    
                # 8. Finalize bar
                # Nothing to do here for now
                
            # End of loop: Finalize position
            if self.position_size > 0:
                self.logger.info("End of backtest. Closing remaining position.")
                self.execute_sell(current_price, timestamp, quantity=None, order_type='MARKET', reason="End of Backtest")
                
        except Exception as e:
            self.logger.error(f"An error occurred during backtest: {e}")
            
        # Calculate and return results
        return self._calculate_results(data)
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate final metrics and return BacktestResult."""
        
        # Ensure final equity is recorded
        if len(self.equity_history) > 0:
            final_equity = self.equity_history[-1][1]
        else:
            final_equity = self.starting_capital
            
        # 1. Equity Curve
        equity_df = pd.DataFrame(
            self.equity_history,
            columns=['timestamp', 'equity', 'cash', 'unrealized_pnl']
        ).set_index('timestamp')
        
        # 2. Daily PnL
        daily_pnl = equity_df['equity'].resample('D').last().ffill().pct_change().fillna(0)
        daily_pnl = pd.DataFrame(daily_pnl, columns=['daily_return'])
        
        # 3. Metrics
        metrics = self._calculate_metrics(equity_df, daily_pnl)
        
        # 4. Final Result
        return BacktestResult(
            trades=self.trades,
            orders=self.orders, # Include all orders
            equity_curve=equity_df,
            metrics=metrics,
            daily_pnl=daily_pnl
        )
        
    def _calculate_metrics(self, equity_df: pd.DataFrame, daily_pnl: pd.DataFrame) -> Dict:
        """Calculate performance metrics."""
        
        metrics = {}
        
        # Total Return
        initial_equity = self.starting_capital
        final_equity = equity_df['equity'].iloc[-1] if not equity_df.empty else initial_equity
        total_return = (final_equity / initial_equity) - 1
        metrics['Total Return'] = total_return
        
        # Annualized Return (assuming daily data for simplicity)
        num_days = (equity_df.index[-1] - equity_df.index[0]).days if not equity_df.empty else 0
        annualized_return = (1 + total_return) ** (365 / num_days) - 1 if num_days > 0 else 0
        metrics['Annualized Return'] = annualized_return
        
        # Max Drawdown
        equity_curve = equity_df['equity']
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min() if not drawdown.empty else 0
        metrics['Max Drawdown'] = max_drawdown
        
        # Sharpe Ratio (using daily returns)
        risk_free_rate = 0.0 # Assuming 0% risk-free rate
        sharpe_ratio = (daily_pnl['daily_return'].mean() * np.sqrt(365)) / daily_pnl['daily_return'].std() if daily_pnl['daily_return'].std() != 0 else 0
        metrics['Sharpe Ratio'] = sharpe_ratio
        
        # Win Rate
        winning_trades = [t for t in self.trades if t.side == 'SELL' and t.pnl > 0]
        total_closed_trades = len([t for t in self.trades if t.side == 'SELL'])
        win_rate = len(winning_trades) / total_closed_trades if total_closed_trades > 0 else 0
        metrics['Win Rate'] = win_rate
        
        # Total Trades
        metrics['Total Trades'] = len(self.trades)
        
        # Average PnL per Trade
        total_pnl = sum(t.pnl for t in self.trades if t.side == 'SELL')
        avg_pnl = total_pnl / total_closed_trades if total_closed_trades > 0 else 0
        metrics['Average PnL per Trade'] = avg_pnl
        
        return metrics
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data by adding indicators."""
        self.logger.info("Preparing data and adding indicators...")
        data = add_all_indicators(data, self.config)
        self.logger.info("Data preparation complete.")
        return data
    
    def _prepare_htf_data(self, data: pd.DataFrame):
        """Prepare Higher Timeframe data."""
        self.logger.info("Preparing Higher Timeframe data...")
        self.htf_data = self.mtf_analysis.prepare_all_htf_data(data)
        self.logger.info("HTF data preparation complete.")
        
    def _get_htf_analysis(self, timestamp: datetime) -> Dict:
        """Get HTF analysis for a given timestamp."""
        return self.mtf_analysis.get_htf_analysis(timestamp, self.htf_data)

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
