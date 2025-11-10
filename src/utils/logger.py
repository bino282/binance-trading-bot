"""
Logger Utility
Provides logging functionality for the trading bot.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorlog


class TradingLogger:
    """Custom logger for trading bot with colored console output."""
    
    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        level: str = "INFO",
        console_output: bool = True
    ):
        """
        Initialize the trading logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Enable console output
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()
        
        # File handler
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler with colors
        if console_output:
            console_handler = colorlog.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def trade(self, side: str, price: float, quantity: float, reason: str = ""):
        """Log trade execution."""
        msg = f"TRADE: {side} {quantity:.6f} @ {price:.4f}"
        if reason:
            msg += f" | Reason: {reason}"
        self.logger.info(msg)
    
    def order(self, order_type: str, side: str, price: float, quantity: float, order_id: Optional[str] = None):
        """Log order placement."""
        msg = f"ORDER: {order_type} {side} {quantity:.6f} @ {price:.4f}"
        if order_id:
            msg += f" | ID: {order_id}"
        self.logger.info(msg)
    
    def fill(self, side: str, price: float, quantity: float, fee: float, order_id: Optional[str] = None):
        """Log order fill."""
        msg = f"FILL: {side} {quantity:.6f} @ {price:.4f} | Fee: {fee:.6f}"
        if order_id:
            msg += f" | ID: {order_id}"
        self.logger.info(msg)
    
    def pnl(self, realized_pnl: float, unrealized_pnl: float, total_pnl: float):
        """Log PnL update."""
        self.logger.info(
            f"PnL: Realized={realized_pnl:.2f} | Unrealized={unrealized_pnl:.2f} | Total={total_pnl:.2f}"
        )
    
    def risk_event(self, event_type: str, details: str):
        """Log risk management event."""
        self.logger.warning(f"RISK: {event_type} | {details}")
    
    def signal(self, signal_type: str, score: float, scenario: str, details: str = ""):
        """Log trading signal."""
        msg = f"SIGNAL: {signal_type} | Score: {score:.1f} | Scenario: {scenario}"
        if details:
            msg += f" | {details}"
        self.logger.info(msg)


def get_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "INFO",
    console_output: bool = True
) -> TradingLogger:
    """
    Get or create a trading logger.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console_output: Enable console output
        
    Returns:
        TradingLogger instance
    """
    return TradingLogger(name, log_dir, level, console_output)
