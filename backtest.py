#!/usr/bin/env python3
"""
Backtest Entry Point
Run backtests on historical data with the configured strategy.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import ConfigLoader
from src.backtest.data_loader import DataLoader
from src.backtest.engine import BacktestEngine
from src.reporting.pnl_report import PnLReportGenerator, create_markdown_report
from src.reporting.visualizer import BacktestVisualizer
from src.utils.logger import get_logger


def main():
    """Main backtest function."""
    parser = argparse.ArgumentParser(description='Run backtest on historical data')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to strategy config file')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=None,
                       help='Starting capital (overrides config)')
    parser.add_argument('--output', type=str, default='data/reports',
                       help='Output directory for reports')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Path to historical data CSV (optional, will fetch if not provided)')
    parser.add_argument('--save-data', action='store_true',
                       help='Save fetched historical data to CSV')
    parser.add_argument('--no-charts', action='store_true',
                       help='Skip chart generation')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = get_logger('backtest_main', level='INFO')
    
    logger.info("=" * 80)
    logger.info("BINANCE TRADING BOT - BACKTEST MODE")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = ConfigLoader(args.config)
    
    symbol = config.get_symbol()
    interval = config.get_kline_interval()
    
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Period: {args.start_date} to {args.end_date}")
    
    # Override starting capital if provided
    if args.initial_capital:
        config.config['starting_cash_usdt'] = args.initial_capital
        logger.info(f"Starting capital overridden: ${args.initial_capital:.2f}")
    
    # Load or fetch historical data
    if args.data_file:
        logger.info(f"Loading data from: {args.data_file}")
        loader = DataLoader(symbol, interval)
        data = loader.load_from_csv(args.data_file)
    else:
        logger.info("Fetching historical data from Binance...")
        loader = DataLoader(symbol, interval)
        
        save_path = None
        if args.save_data:
            Path('data/historical').mkdir(parents=True, exist_ok=True)
            save_path = f"data/historical/{symbol}_{interval}_{args.start_date}_{args.end_date}.csv"
        
        data = loader.fetch_historical_data(args.start_date, args.end_date, save_path)
    
    if data.empty:
        logger.error("No data available for backtesting")
        return 1
    
    logger.info(f"Data loaded: {len(data)} bars")
    
    # Initialize backtest engine
    logger.info("Initializing backtest engine...")
    engine = BacktestEngine(config.get_all())
    
    # Run backtest
    logger.info("Running backtest...")
    result = engine.run(data, args.start_date, args.end_date)
    
    # Generate reports
    logger.info("Generating reports...")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if result is None:
        logger.error("Backtest failed to produce a result object. Exiting.")
        return 1
        
    # Generate and save reports
    report_gen = PnLReportGenerator(result)
    report_gen.print_summary()
    report_gen.save_report(str(output_dir), prefix=symbol)
    
    # Generate markdown report
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    markdown_file = output_dir / f"{symbol}_report_{timestamp}.md"
    create_markdown_report(result, str(markdown_file))
    
    # Generate charts
    if not args.no_charts:
        logger.info("Generating charts...")
        visualizer = BacktestVisualizer(result)
        visualizer.save_all_charts(str(output_dir), prefix=symbol)
    
    logger.info("=" * 80)
    logger.info("Backtest completed successfully!")
    logger.info(f"Reports saved to: {output_dir}")
    logger.info("=" * 80)
    
    # Print key metrics
    key_metrics = report_gen.get_key_metrics()
    logger.info("\nKEY METRICS:")
    logger.info(f"  Total Return: {key_metrics['total_return_pct']:.2f}%")
    logger.info(f"  Win Rate: {key_metrics['win_rate']:.2%}")
    logger.info(f"  Max Drawdown: {key_metrics['max_drawdown']:.2%}")
    logger.info(f"  Sharpe Ratio: {key_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Profit Factor: {key_metrics['profit_factor']:.2f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
