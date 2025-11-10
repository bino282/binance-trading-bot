"""
Data Loader
Fetches and manages historical price data for backtesting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException


class DataLoader:
    """Loads historical OHLCV data for backtesting."""
    
    def __init__(self, symbol: str, interval: str = '5m'):
        """
        Initialize data loader.
        
        Args:
            symbol: Trading symbol (e.g., 'ZECUSDT')
            interval: Kline interval (e.g., '5m', '1h')
        """
        self.symbol = symbol
        self.interval = interval
        self.client = Client("", "")  # Public API, no keys needed for historical data
    
    def fetch_historical_data(
        self,
        start_date: str,
        end_date: str,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical kline data from Binance.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            save_path: Optional path to save CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching historical data for {self.symbol} from {start_date} to {end_date}...")
        
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        all_klines = []
        current_ts = start_ts
        
        # Fetch data in chunks (Binance limit is 1000 klines per request)
        while current_ts < end_ts:
            try:
                klines = self.client.get_klines(
                    symbol=self.symbol,
                    interval=self.interval,
                    startTime=current_ts,
                    endTime=end_ts,
                    limit=1000
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                current_ts = klines[-1][0] + 1  # Next timestamp
                
                print(f"Fetched {len(klines)} klines, total: {len(all_klines)}")
                time.sleep(0.1)  # Rate limiting
                
            except BinanceAPIException as e:
                print(f"API Error: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df.set_index('timestamp', inplace=True)
        
        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        print(f"Total klines fetched: {len(df)}")
        
        # Save to CSV if path provided
        if save_path:
            df.to_csv(save_path)
            print(f"Data saved to {save_path}")
        
        return df
    
    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load historical data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Ensure correct column types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        return df
    
    def fetch_multiple_timeframes(
        self,
        start_date: str,
        end_date: str,
        intervals: List[str],
        save_dir: Optional[str] = None
    ) -> dict:
        """
        Fetch data for multiple timeframes.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            intervals: List of intervals (e.g., ['5m', '15m', '1h'])
            save_dir: Optional directory to save CSV files
            
        Returns:
            Dictionary mapping interval to DataFrame
        """
        data = {}
        
        for interval in intervals:
            print(f"\nFetching {interval} data...")
            loader = DataLoader(self.symbol, interval)
            
            save_path = None
            if save_dir:
                save_path = f"{save_dir}/{self.symbol}_{interval}_{start_date}_{end_date}.csv"
            
            df = loader.fetch_historical_data(start_date, end_date, save_path)
            data[interval] = df
        
        return data
    
    @staticmethod
    def resample_to_higher_timeframe(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """
        Resample OHLCV data to a higher timeframe.
        
        Args:
            df: Source DataFrame with OHLCV data
            target_interval: Target interval (e.g., '15m', '1h', '4h')
            
        Returns:
            Resampled DataFrame
        """
        # Map interval strings to pandas offset aliases
        interval_map = {
            '1m': '1T',
            '5m': '5T',
            '10m': '10T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        freq = interval_map.get(target_interval)
        if not freq:
            raise ValueError(f"Unsupported interval: {target_interval}")
        
        # Resample OHLCV data
        resampled = pd.DataFrame()
        resampled['open'] = df['open'].resample(freq).first()
        resampled['high'] = df['high'].resample(freq).max()
        resampled['low'] = df['low'].resample(freq).min()
        resampled['close'] = df['close'].resample(freq).last()
        resampled['volume'] = df['volume'].resample(freq).sum()
        
        return resampled.dropna()
    
    def prepare_backtest_data(
        self,
        start_date: str,
        end_date: str,
        base_interval: str = '5m',
        higher_timeframes: Optional[List[str]] = None
    ) -> dict:
        """
        Prepare complete dataset for backtesting including multiple timeframes.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            base_interval: Base trading timeframe
            higher_timeframes: List of higher timeframes for MTF analysis
            
        Returns:
            Dictionary with 'base' and 'htf' dataframes
        """
        # Fetch base timeframe data
        base_df = self.fetch_historical_data(start_date, end_date)
        
        result = {'base': base_df, 'htf': {}}
        
        # Resample to higher timeframes if needed
        if higher_timeframes:
            for htf in higher_timeframes:
                print(f"Resampling to {htf}...")
                result['htf'][htf] = self.resample_to_higher_timeframe(base_df, htf)
        
        return result


def download_sample_data(
    symbol: str = 'ZECUSDT',
    days: int = 90,
    interval: str = '5m',
    save_dir: str = 'data/historical'
) -> str:
    """
    Download sample data for testing.
    
    Args:
        symbol: Trading symbol
        days: Number of days of historical data
        interval: Kline interval
        save_dir: Directory to save data
        
    Returns:
        Path to saved CSV file
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    loader = DataLoader(symbol, interval)
    
    save_path = f"{save_dir}/{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    
    df = loader.fetch_historical_data(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        save_path
    )
    
    return save_path
