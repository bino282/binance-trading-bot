"""
Binance Client Wrapper
Handles connection to Binance API for live trading (testnet and mainnet).
"""

import time
from typing import Dict, List, Optional, Tuple
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd

from ..utils.logger import get_logger


class BinanceClientWrapper:
    """
    Wrapper for Binance API client with testnet/mainnet support.
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True
    ):
        """
        Initialize Binance client.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet if True, mainnet if False
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.logger = get_logger('binance_client')
        
        # Initialize client
        if testnet:
            self.client = Client(api_key, api_secret, testnet=True)
            self.logger.info("Connected to Binance TESTNET")
        else:
            self.client = Client(api_key, api_secret)
            self.logger.warning("Connected to Binance MAINNET - REAL MONEY!")
        
        # Test connection
        try:
            self.client.ping()
            self.logger.info("Connection successful")
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            raise
    
    def get_account_balance(self, asset: str = 'USDT') -> float:
        """
        Get account balance for specific asset.
        
        Args:
            asset: Asset symbol (e.g., 'USDT', 'BTC')
            
        Returns:
            Available balance
        """
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0.0
        except BinanceAPIException as e:
            self.logger.error(f"Error getting balance: {e}")
            return 0.0
    
    def get_position(self, symbol: str) -> Tuple[float, float]:
        """
        Get current position for symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'ZECUSDT')
            
        Returns:
            Tuple of (quantity, avg_price)
        """
        try:
            # Get base asset from symbol (e.g., 'ZEC' from 'ZECUSDT')
            base_asset = symbol.replace('USDT', '').replace('BUSD', '')
            
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == base_asset:
                    quantity = float(balance['free']) + float(balance['locked'])
                    
                    if quantity > 0:
                        # Get average price from recent trades
                        trades = self.client.get_my_trades(symbol=symbol, limit=50)
                        if trades:
                            total_cost = sum(float(t['price']) * float(t['qty']) for t in trades if t['isBuyer'])
                            total_qty = sum(float(t['qty']) for t in trades if t['isBuyer'])
                            avg_price = total_cost / total_qty if total_qty > 0 else 0
                            return quantity, avg_price
                    
                    return quantity, 0.0
            
            return 0.0, 0.0
        except BinanceAPIException as e:
            self.logger.error(f"Error getting position: {e}")
            return 0.0, 0.0
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            self.logger.error(f"Error getting price: {e}")
            return 0.0
    
    def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """
        Get order book.
        
        Args:
            symbol: Trading symbol
            limit: Number of levels
            
        Returns:
            Order book data
        """
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            return {
                'bids': [[float(p), float(q)] for p, q in depth['bids']],
                'asks': [[float(p), float(q)] for p, q in depth['asks']]
            }
        except BinanceAPIException as e:
            self.logger.error(f"Error getting order book: {e}")
            return {'bids': [], 'asks': []}
    
    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Optional[Dict]:
        """
        Place market order.
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            
        Returns:
            Order response or None
        """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            
            self.logger.info(f"Market order placed: {side} {quantity} {symbol}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        time_in_force: str = 'GTC'
    ) -> Optional[Dict]:
        """
        Place limit order.
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            price: Limit price
            quantity: Order quantity
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
            
        Returns:
            Order response or None
        """
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce=time_in_force,
                price=price,
                quantity=quantity
            )
            
            self.logger.info(f"Limit order placed: {side} {quantity} {symbol} @ {price}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Error placing limit order: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: int) -> bool:
        """
        Cancel order.
        
        Args:
            symbol: Trading symbol
            order_id: Order ID
            
        Returns:
            True if successful
        """
        try:
            self.client.cancel_order(symbol=symbol, orderId=order_id)
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        except BinanceAPIException as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_open_orders(self, symbol: str) -> List[Dict]:
        """
        Get open orders for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of open orders
        """
        try:
            orders = self.client.get_open_orders(symbol=symbol)
            return orders
        except BinanceAPIException as e:
            self.logger.error(f"Error getting open orders: {e}")
            return []
    
    def cancel_all_orders(self, symbol: str) -> bool:
        """
        Cancel all open orders for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if successful
        """
        try:
            self.client.cancel_open_orders(symbol=symbol)
            self.logger.info(f"All orders cancelled for {symbol}")
            return True
        except BinanceAPIException as e:
            self.logger.error(f"Error cancelling all orders: {e}")
            return False
    
    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Get recent trades.
        
        Args:
            symbol: Trading symbol
            limit: Number of trades
            
        Returns:
            List of trades
        """
        try:
            trades = self.client.get_my_trades(symbol=symbol, limit=limit)
            return trades
        except BinanceAPIException as e:
            self.logger.error(f"Error getting trades: {e}")
            return []
    
    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get recent klines (candlesticks).
        
        Args:
            symbol: Trading symbol
            interval: Kline interval (e.g., '5m', '1h')
            limit: Number of klines
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
        
        except BinanceAPIException as e:
            self.logger.error(f"Error getting klines: {e}")
            return pd.DataFrame()
    
    def get_exchange_info(self, symbol: str) -> Optional[Dict]:
        """
        Get exchange info for symbol (filters, precision, etc.).
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Exchange info or None
        """
        try:
            info = self.client.get_symbol_info(symbol)
            return info
        except BinanceAPIException as e:
            self.logger.error(f"Error getting exchange info: {e}")
            return None
    
    def round_step_size(self, quantity: float, step_size: float) -> float:
        """
        Round quantity to exchange step size.
        
        Args:
            quantity: Raw quantity
            step_size: Exchange step size
            
        Returns:
            Rounded quantity
        """
        precision = len(str(step_size).split('.')[-1].rstrip('0'))
        return round(quantity - (quantity % step_size), precision)
    
    def round_tick_size(self, price: float, tick_size: float) -> float:
        """
        Round price to exchange tick size.
        
        Args:
            price: Raw price
            tick_size: Exchange tick size
            
        Returns:
            Rounded price
        """
        precision = len(str(tick_size).split('.')[-1].rstrip('0'))
        return round(price - (price % tick_size), precision)
