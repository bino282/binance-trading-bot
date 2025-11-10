"""
Live Trader
Orchestrates live trading with real-time data and order execution.
"""

import time
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timedelta

from .binance_client import BinanceClientWrapper
from ..indicators.technical import add_all_indicators
from ..indicators.score_engine import ScoreEngine
from ..strategy.risk_manager import RiskManager
from ..utils.logger import get_logger
from ..utils.config_loader import ConfigLoader


class LiveTrader:
    """
    Live trading orchestrator.
    Manages real-time data, signal generation, and order execution.
    """
    
    def __init__(
        self,
        config: ConfigLoader,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        dry_run: bool = False
    ):
        """
        Initialize live trader.
        
        Args:
            config: Configuration loader
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True
            dry_run: Simulate trades without execution
        """
        self.config = config
        self.dry_run = dry_run
        self.logger = get_logger('live_trader', level='INFO')
        
        # Trading parameters
        self.symbol = config.get_symbol()
        self.base_asset = config.get_base_asset()
        self.quote_asset = config.get_quote_asset()
        self.order_size_quote = config.get_order_size()
        self.kline_interval = config.get_kline_interval()
        
        # Market filters
        self.tick_size = config.get_tick_size()
        self.lot_size = config.get_lot_size()
        self.min_notional = config.get_min_notional()
        
        # Initialize Binance client
        self.client = BinanceClientWrapper(api_key, api_secret, testnet)
        
        # Initialize strategy components
        score_config = config.get_score_layer_config()
        risk_config = config.get_all().get('policy_cfg', {})
        
        self.score_engine = ScoreEngine(score_config)
        
        # Get starting capital from account or config
        starting_capital = self.client.get_account_balance(self.quote_asset)
        if starting_capital == 0:
            starting_capital = config.get_starting_capital()
        
        self.risk_manager = RiskManager(risk_config, starting_capital)
        
        # State variables
        self.position_size = 0.0
        self.avg_entry_price = 0.0
        self.last_fill_price = 0.0
        self.dca_count = 0
        self.bars_since_entry = 0
        
        # Data buffer
        self.data_buffer: Optional[pd.DataFrame] = None
        self.last_bar_time: Optional[datetime] = None
        
        # Trading state
        self.is_running = False
        self.current_day = None
        
        self.logger.info("=" * 80)
        self.logger.info(f"Live Trader Initialized")
        self.logger.info(f"Symbol: {self.symbol}")
        self.logger.info(f"Mode: {'TESTNET' if testnet else 'MAINNET'}")
        self.logger.info(f"Dry Run: {dry_run}")
        self.logger.info(f"Starting Capital: ${starting_capital:.2f}")
        self.logger.info("=" * 80)
    
    def initialize_data_buffer(self):
        """Initialize data buffer with historical klines."""
        self.logger.info("Initializing data buffer...")
        
        # Get warmup bars
        warmup_bars = self.config.get('policy_cfg.indicators.warmup_bars', 300)
        
        # Fetch historical data
        df = self.client.get_klines(
            self.symbol,
            self.kline_interval,
            limit=warmup_bars + 50  # Extra buffer
        )
        
        if df.empty:
            raise RuntimeError("Failed to fetch historical data")
        
        # Add indicators
        indicators_config = self.config.get_indicators_config()
        df = add_all_indicators(df, indicators_config)
        
        self.data_buffer = df
        self.last_bar_time = df.index[-1]
        
        self.logger.info(f"Data buffer initialized with {len(df)} bars")
        self.logger.info(f"Last bar time: {self.last_bar_time}")
    
    def update_data_buffer(self):
        """Update data buffer with latest kline."""
        # Fetch latest klines
        df = self.client.get_klines(
            self.symbol,
            self.kline_interval,
            limit=10
        )
        
        if df.empty:
            return
        
        # Check if we have a new complete bar
        latest_bar_time = df.index[-2]  # Use -2 to get completed bar
        
        if self.last_bar_time is None or latest_bar_time > self.last_bar_time:
            # New bar completed
            self.logger.info(f"New bar completed: {latest_bar_time}")
            
            # Append new bar to buffer
            new_bar = df.iloc[-2]
            self.data_buffer = pd.concat([self.data_buffer, new_bar.to_frame().T])
            
            # Keep only recent bars
            max_buffer_size = 500
            if len(self.data_buffer) > max_buffer_size:
                self.data_buffer = self.data_buffer.iloc[-max_buffer_size:]
            
            # Recalculate indicators
            indicators_config = self.config.get_indicators_config()
            self.data_buffer = add_all_indicators(self.data_buffer, indicators_config)
            
            self.last_bar_time = latest_bar_time
            
            return True  # New bar available
        
        return False  # No new bar
    
    def sync_position(self):
        """Sync position with exchange."""
        quantity, avg_price = self.client.get_position(self.symbol)
        
        self.position_size = quantity
        self.avg_entry_price = avg_price
        
        if quantity > 0:
            self.logger.info(f"Position synced: {quantity:.6f} {self.base_asset} @ ${avg_price:.4f}")
        else:
            self.logger.info("No open position")
    
    def execute_buy(
        self,
        price: float,
        reason: str = ""
    ) -> bool:
        """
        Execute buy order.
        
        Args:
            price: Current price
            reason: Trade reason
            
        Returns:
            True if successful
        """
        # Calculate quantity
        quantity = self.order_size_quote / price
        quantity = self.client.round_step_size(quantity, self.lot_size)
        
        if quantity <= 0:
            self.logger.warning("Invalid quantity")
            return False
        
        # Check min notional
        value = quantity * price
        if value < self.min_notional:
            self.logger.warning(f"Order below min notional: ${value:.2f}")
            return False
        
        self.logger.info(f"Executing BUY: {quantity:.6f} @ ${price:.4f} | {reason}")
        
        if self.dry_run:
            self.logger.info("[DRY RUN] Order not executed")
            return True
        
        # Place market order
        order = self.client.place_market_order(
            self.symbol,
            'BUY',
            quantity
        )
        
        if order:
            self.logger.order('MARKET', 'BUY', price, quantity, order.get('orderId'))
            
            # Update position
            self.sync_position()
            self.last_fill_price = price
            
            return True
        
        return False
    
    def execute_sell(
        self,
        price: float,
        quantity: Optional[float] = None,
        reason: str = ""
    ) -> bool:
        """
        Execute sell order.
        
        Args:
            price: Current price
            quantity: Quantity to sell (None = all)
            reason: Trade reason
            
        Returns:
            True if successful
        """
        if self.position_size <= 0:
            self.logger.warning("No position to sell")
            return False
        
        # Default to selling all
        if quantity is None:
            quantity = self.position_size
        else:
            quantity = min(quantity, self.position_size)
        
        quantity = self.client.round_step_size(quantity, self.lot_size)
        
        if quantity <= 0:
            self.logger.warning("Invalid quantity")
            return False
        
        # Calculate PnL
        pnl = (price - self.avg_entry_price) * quantity
        
        self.logger.info(f"Executing SELL: {quantity:.6f} @ ${price:.4f} | PnL: ${pnl:.2f} | {reason}")
        
        if self.dry_run:
            self.logger.info("[DRY RUN] Order not executed")
            return True
        
        # Place market order
        order = self.client.place_market_order(
            self.symbol,
            'SELL',
            quantity
        )
        
        if order:
            self.logger.order('MARKET', 'SELL', price, quantity, order.get('orderId'))
            
            # Update position
            self.sync_position()
            
            # Update risk manager
            self.risk_manager.update_capital(pnl)
            is_loss = pnl < 0
            self.risk_manager.check_stop_loss(pnl, is_loss)
            
            if not is_loss:
                self.risk_manager.reset_consecutive_losses()
            
            if self.position_size == 0:
                self.dca_count = 0
                self.bars_since_entry = 0
            
            return True
        
        return False
    
    def process_bar(self):
        """Process new bar and generate trading signals."""
        if self.data_buffer is None or len(self.data_buffer) == 0:
            return
        
        # Get current bar
        current_bar = self.data_buffer.iloc[-1]
        current_price = current_bar['close']
        bar_index = len(self.data_buffer) - 1
        
        # Check for new day (reset daily states)
        bar_day = current_bar.name.date()
        if self.current_day != bar_day:
            self.logger.info(f"New trading day: {bar_day}")
            self.risk_manager.reset_daily_states(current_price)
            self.current_day = bar_day
        
        # Update risk manager
        position_value = self.position_size * current_price
        realized_pnl = 0.0  # Track separately in production
        
        self.risk_manager.update_pnl_gate(
            current_price,
            position_value,
            realized_pnl
        )
        pnl_gate_status = self.risk_manager.get_pnl_gate_status()
        
        self.logger.info(f"PnL Gate Status: {pnl_gate_status}")
        
        # 4. Check force sell conditions (Hard Stop / Kill Switch)
        
        # Check for hard stop
        if self.risk_manager.is_hard_stop_active():
            if self.position_size > 0:
                self.execute_sell(current_price, None, reason="Hard Stop / Kill Switch")
            self.logger.warning("Hard Stop Active. Trading paused.")
            return
            
        should_force_sell, sell_qty, force_reason = self.risk_manager.should_force_sell(
            current_price,
            self.position_size
        )
        
        if should_force_sell and sell_qty > 0:
            self.execute_sell(current_price, sell_qty, force_reason)
            return
        
        # 5. Get HTF analysis for scoring
        htf_analysis = self._get_htf_analysis(current_bar.name)
        
        # 6. Generate trading signal
        signal, best_scenario = self.score_engine.generate_signal(
            current_bar,
            bar_index,
            self.position_size,
            htf_analysis
        )
        
        # Log signal
        if best_scenario:
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
                        
                    self.execute_buy(
                        current_price,
                        order_size=order_size,
                        reason=f"Entry signal: {best_scenario.scenario_id if best_scenario else 'N/A'}{reason_suffix}"
                    )
                    self.bars_since_entry = 0
                else:
                    self.logger.warning(f"Cannot open position: {reason}")
            
        elif self.position_size > 0:
            # 7.1. Check Hard Stop Loss
            should_stop_loss, sl_reason = self.risk_manager.should_stop_loss(
                current_price, self.avg_entry_price, self.position_size
            )
            if should_stop_loss:
                self.execute_sell(current_price, None, sl_reason)
                return
            
            # 7.2. Check Signal Exit
            if signal == 'exit':
                self.execute_sell(
                    current_price,
                    None,
                    f"Exit signal: {best_scenario.scenario_id if best_scenario else 'N/A'}"
                )
                return
            
            # 7.3. Check Trailing Take Profit
            should_trail, trigger_pct, trail_pct = self.dca_tp_engine.should_trail_tp(
                current_price, self.avg_entry_price, current_bar['atr_pct'], self.bars_since_entry
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
                        current_price, None, 
                        reason=f"Trailing TP Hit (Stop: {self.trailing_stop_price:.2f})"
                    )
                    self.trailing_stop_price = 0.0
                    return
            
            # 7.4. Check DCA opportunity
            can_dca, dca_reason = self.dca_tp_engine.can_dca(
                current_bar['rsi'], current_price, pnl_gate_status
            )
            
            if can_dca:
                order_size_pct = self.dca_tp_engine.get_dca_order_size_pct()
                dca_order_size = self.order_size_quote * order_size_pct
                
                self.execute_buy(
                    current_price,
                    order_size=dca_order_size,
                    reason=f"DCA Step {self.dca_tp_engine.dca_state.count + 1} - {dca_reason}"
                )
                self.dca_tp_engine.update_dca_fill(current_price)
                self.trailing_stop_price = 0.0 # Reset TP on DCA
                    
            # 7.5. Update position state
            self.bars_since_entry += 1
            self.dca_tp_engine.update_dca_cooldown()
        
        # Log risk metrics
        metrics = self.risk_manager.get_risk_metrics()
        self.logger.pnl(
            0.0,  # realized_pnl
            (current_price - self.avg_entry_price) * self.position_size if self.position_size > 0 else 0.0,
            metrics['current_capital'] - self.risk_manager.starting_capital
        )
    
    def run(self):
        """Run live trading loop."""
        self.logger.info("Starting live trading...")
        
        # Initialize
        self.initialize_data_buffer()
        self.sync_position()
        
        self.is_running = True
        
        # Get main loop interval
        main_loop_ms = self.config.get('runtime_cfg.main_loop_ms', 500)
        sleep_seconds = main_loop_ms / 1000.0
        
        try:
            while self.is_running:
                # Update data buffer
                new_bar = self.update_data_buffer()
                
                if new_bar:
                    # Process new bar
                    self.process_bar()
                
                # Sleep
                time.sleep(sleep_seconds)
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
            raise
        finally:
            self.stop()
    
    def stop(self):
        """Stop live trading."""
        self.logger.info("Stopping live trading...")
        self.is_running = False
        
        # Close position if configured
        if self.position_size > 0:
            self.logger.warning(f"Open position: {self.position_size:.6f} {self.base_asset}")
            # Optionally close position here
        
        self.logger.info("Live trading stopped")

    def _prepare_htf_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare higher timeframe data for live trading."""
        htf_config = self.config.get_all().get('policy_cfg', {}).get('multi_timeframe', {})
        timeframes = htf_config.get('timeframes', [])
        
        htf_data = {}
        for tf in timeframes:
            self.logger.info(f"Preparing HTF data for {tf}...")
            
            # Fetch historical data for HTF
            df = self.client.get_klines(
                self.symbol,
                tf,
                limit=300  # Enough for indicators
            )
            
            if df.empty:
                self.logger.warning(f"Failed to fetch HTF data for {tf}")
                continue
            
            # Add indicators to HTF data
            indicators_config = self.config.get_indicators_config()
            df = add_all_indicators(df, indicators_config)
            
            # Add MTF analysis (e.g., trend)
            df = self.mtf_analysis.add_analysis(df)
            
            htf_data[tf] = df
            
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
