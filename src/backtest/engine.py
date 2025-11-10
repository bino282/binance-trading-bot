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
            data: OHLCV DataFrame with indicators
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            BacktestResult object
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting backtest...")
        self.logger.info("=" * 80)
        
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Add indicators
        indicators_config = self.config.get('policy_cfg', {}).get('indicators', {})
        data = add_all_indicators(data, indicators_config)
        
        # Prepare HTF data
        self.htf_data = self._prepare_htf_data(data)
        
        # Get warmup period
        warmup_bars = indicators_config.get('warmup_bars', 300)
        
        self.logger.info(f"Total bars: {len(data)}")
        self.logger.info(f"Warmup bars: {warmup_bars}")
        self.logger.info(f"Trading period: {data.index[warmup_bars]} to {data.index[-1]}")
        
        # Initialize day tracking
        current_day = None
        
        # Main backtest loop
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
                    self.execute_sell(current_price, timestamp, reason="Hard Stop / Kill Switch")
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
                    self.execute_sell(current_price, timestamp, sl_reason)
                    return
                    
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
                            current_price, timestamp, 
                            reason=f"Trailing TP Hit (Stop: {self.trailing_stop_price:.2f})"
                        )
                        self.trailing_stop_price = 0.0
                        continue
                
                # 7.4. Check DCA opportunity
                can_dca, dca_reason = self.dca_tp_engine.can_dca(
                    row['rsi'], current_price, pnl_gate_status
                )
                
                if can_dca:
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
        self.logger.info("=" * 80)
        self._log_summary(result)
        
        return result
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate backtest results and metrics."""
        # Create equity curve
        equity_df = pd.DataFrame(
            self.equity_history,
            columns=['timestamp', 'equity', 'cash', 'unrealized_pnl']
        )
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate daily PnL
        equity_df['daily_pnl'] = equity_df['equity'].diff()
        equity_df['daily_pnl_pct'] = equity_df['equity'].pct_change()
        
        # Calculate metrics
        final_equity = self.equity
        total_return = (final_equity - self.starting_capital) / self.starting_capital
        
        # Trade statistics
        buy_trades = [t for t in self.trades if t.side == 'BUY']
        sell_trades = [t for t in self.trades if t.side == 'SELL']
        
        winning_trades = [t for t in sell_trades if t.pnl > 0]
        losing_trades = [t for t in sell_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        profit_factor = (
            sum(t.pnl for t in winning_trades) / abs(sum(t.pnl for t in losing_trades))
            if losing_trades and sum(t.pnl for t in losing_trades) != 0
            else 0
        )
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(equity_df['daily_pnl_pct'].dropna()) > 0:
            sharpe_ratio = (
                equity_df['daily_pnl_pct'].mean() / equity_df['daily_pnl_pct'].std()
                * np.sqrt(252)
            )
        else:
            sharpe_ratio = 0
        
        # Total fees
        total_fees = sum(t.fee for t in self.trades)
        
        metrics = {
            'starting_capital': self.starting_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_fees': total_fees,
        }
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_df,
            metrics=metrics,
            daily_pnl=equity_df[['daily_pnl', 'daily_pnl_pct']]
        )
    
    def _log_summary(self, result: BacktestResult):
        """Log backtest summary."""
        m = result.metrics
        
        self.logger.info(f"Starting Capital: ${m['starting_capital']:.2f}")
        self.logger.info(f"Final Equity: ${m['final_equity']:.2f}")
        self.logger.info(f"Total Return: {m['total_return_pct']:.2f}%")
        self.logger.info(f"Max Drawdown: {m['max_drawdown']:.2%}")
        self.logger.info(f"Sharpe Ratio: {m['sharpe_ratio']:.2f}")
        self.logger.info(f"")
        self.logger.info(f"Total Trades: {m['total_trades']}")
        self.logger.info(f"Buy Trades: {m['buy_trades']}")
        self.logger.info(f"Sell Trades: {m['sell_trades']}")
        self.logger.info(f"Win Rate: {m['win_rate']:.2%}")
        self.logger.info(f"Avg Win: ${m['avg_win']:.2f}")
        self.logger.info(f"Avg Loss: ${m['avg_loss']:.2f}")
        self.logger.info(f"Profit Factor: {m['profit_factor']:.2f}")
        self.logger.info(f"Total Fees: ${m['total_fees']:.2f}")

    def _prepare_htf_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare higher timeframe data for backtesting."""
        htf_config = self.config.get('policy_cfg', {}).get('multi_timeframe', {})
        timeframes = htf_config.get('timeframes', [])
        
        htf_data = {}
        for tf in timeframes:
            self.logger.info(f"Preparing HTF data for {tf}...")
            # Resample data to higher timeframe
            resampled_data = data['close'].resample(tf).ohlc()
            resampled_data.columns = ['open', 'high', 'low', 'close']
            resampled_data.dropna(inplace=True)
            
            # Add indicators to HTF data
            indicators_config = self.config.get('policy_cfg', {}).get('indicators', {})
            resampled_data = add_all_indicators(resampled_data, indicators_config)
            
            # Add MTF analysis (e.g., trend)
            resampled_data = self.mtf_analysis.add_analysis(resampled_data)
            
            htf_data[tf] = resampled_data
            
        return htf_data

    def _get_htf_analysis(self, timestamp: datetime) -> Optional[Dict]:
        """Get the latest HTF analysis for a given timestamp."""
        analysis = {}
        for tf, df in self.htf_data.items():
            # Find the latest bar in the HTF data that is before or at the current timestamp
            latest_bar = df.index[df.index <= timestamp].max()
            if latest_bar is not pd.NaT:
                analysis[tf] = df.loc[latest_bar].to_dict()
            
        return analysis if analysis else None
