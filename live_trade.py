#!/usr/bin/env python3
"""
Live Trading Entry Point
Run live trading on Binance testnet or mainnet.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import ConfigLoader, load_api_keys
from src.live.trader import LiveTrader
from src.utils.logger import get_logger


def main():
    """Main live trading function."""
    parser = argparse.ArgumentParser(description='Run live trading on Binance')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to strategy config file')
    parser.add_argument('--mode', type=str, choices=['testnet', 'mainnet'], required=True,
                       help='Trading mode: testnet or mainnet')
    parser.add_argument('--api-keys', type=str, default='config/api_keys.yaml',
                       help='Path to API keys file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate trades without execution')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = get_logger('live_trade_main', level='INFO')
    
    logger.info("=" * 80)
    logger.info("BINANCE TRADING BOT - LIVE TRADING MODE")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = ConfigLoader(args.config)
    
    symbol = config.get_symbol()
    
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Dry Run: {args.dry_run}")
    
    # Confirmation for mainnet
    if args.mode == 'mainnet' and not args.dry_run:
        logger.warning("=" * 80)
        logger.warning("WARNING: YOU ARE ABOUT TO TRADE WITH REAL MONEY!")
        logger.warning("=" * 80)
        response = input("Type 'YES' to confirm and proceed: ")
        if response != 'YES':
            logger.info("Trading cancelled by user")
            return 0
    
    # Load API keys
    try:
        logger.info(f"Loading API keys from: {args.api_keys}")
        api_keys_config = load_api_keys(args.api_keys)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("\nPlease create config/api_keys.yaml with the following structure:")
        logger.error("""
binance:
  testnet:
    api_key: "YOUR_TESTNET_API_KEY"
    api_secret: "YOUR_TESTNET_API_SECRET"
  mainnet:
    api_key: "YOUR_MAINNET_API_KEY"
    api_secret: "YOUR_MAINNET_API_SECRET"
        """)
        return 1
    
    # Get API credentials
    testnet = (args.mode == 'testnet')
    mode_key = 'testnet' if testnet else 'mainnet'
    
    if mode_key not in api_keys_config.get('binance', {}):
        logger.error(f"API keys for {mode_key} not found in config file")
        return 1
    
    api_key = api_keys_config['binance'][mode_key]['api_key']
    api_secret = api_keys_config['binance'][mode_key]['api_secret']
    
    if not api_key or not api_secret:
        logger.error(f"API key or secret is empty for {mode_key}")
        return 1
    
    # Initialize live trader
    try:
        logger.info("Initializing live trader...")
        trader = LiveTrader(
            config=config,
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            dry_run=args.dry_run
        )
    except Exception as e:
        logger.error(f"Failed to initialize trader: {e}")
        return 1
    
    # Run live trading
    try:
        logger.info("Starting live trading loop...")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 80)
        
        trader.run()
        
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received")
    except Exception as e:
        logger.error(f"Error during live trading: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        trader.stop()
    
    logger.info("=" * 80)
    logger.info("Live trading stopped")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
