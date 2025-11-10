# Binance Trading Bot - ZEC/USDT Strategy

A comprehensive Python trading bot for Binance with backtesting, testnet, and mainnet live trading capabilities. This bot implements a sophisticated multi-timeframe, score-driven trading strategy with advanced risk management.

## Features

- **Multi-Timeframe Analysis (MTF)**: Analyzes 5m, 10m, 15m, 30m, 1h, and 4h timeframes
- **Score-Driven Engine**: Intelligent signal scoring system with 9 scenario-based strategies
- **Advanced Risk Management**: 
  - PnL Gate with daily reset
  - Stop Loss Engine with capital protection
  - DCA (Dollar Cost Averaging) with safety guards
  - Trailing Take Profit
- **Grid Trading**: Dynamic spread calculation based on volatility
- **Arbiter System**: Signal conflict resolution with precedence rules
- **Comprehensive Reporting**: Detailed PnL reports, trade analysis, and performance metrics
- **Multiple Trading Modes**:
  - Backtest mode with historical data
  - Testnet live trading
  - Mainnet live trading

## Project Structure

```
binance-trading-bot/
├── config/
│   ├── ZECUSDT_TEMPLATE.yaml    # Strategy configuration
│   └── api_keys.yaml             # API credentials (create this)
├── src/
│   ├── backtest/
│   │   ├── engine.py             # Backtesting engine
│   │   └── data_loader.py        # Historical data fetcher
│   ├── live/
│   │   ├── trader.py             # Live trading orchestrator
│   │   ├── binance_client.py    # Binance API wrapper
│   │   └── order_manager.py     # Order execution & management
│   ├── indicators/
│   │   ├── technical.py          # Technical indicators (RSI, EMA, MACD, etc.)
│   │   └── score_engine.py       # Score calculation engine
│   ├── strategy/
│   │   ├── signal_generator.py   # Trading signal generation
│   │   ├── risk_manager.py       # Risk management logic
│   │   └── arbiter.py            # Signal arbitration
│   ├── reporting/
│   │   ├── pnl_report.py         # PnL calculation and reporting
│   │   └── visualizer.py         # Chart generation
│   └── utils/
│       ├── config_loader.py      # YAML config parser
│       └── logger.py             # Logging utilities
├── data/
│   ├── historical/               # Historical price data
│   └── reports/                  # Generated reports
├── logs/                         # Trading logs
├── tests/                        # Unit tests
├── backtest.py                   # Backtest entry point
├── live_trade.py                 # Live trading entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. **Clone or extract the project**

2. **Install dependencies**:
```bash
pip3 install -r requirements.txt
```

3. **Configure API keys** (for live trading):
Create `config/api_keys.yaml`:
```yaml
binance:
  testnet:
    api_key: "YOUR_TESTNET_API_KEY"
    api_secret: "YOUR_TESTNET_API_SECRET"
  mainnet:
    api_key: "YOUR_MAINNET_API_KEY"
    api_secret: "YOUR_MAINNET_API_SECRET"
```

## Usage

### Backtesting

Run a backtest with historical data:

```bash
python3 backtest.py --config config/ZECUSDT_TEMPLATE.yaml --start-date 2024-01-01 --end-date 2024-12-31
```

Options:
- `--config`: Path to strategy config file
- `--start-date`: Backtest start date (YYYY-MM-DD)
- `--end-date`: Backtest end date (YYYY-MM-DD)
- `--initial-capital`: Starting capital in USDT (default: from config)
- `--output`: Output report path

### Live Trading - Testnet

Test your strategy on Binance testnet:

```bash
python3 live_trade.py --config config/ZECUSDT_TEMPLATE.yaml --mode testnet
```

### Live Trading - Mainnet

**⚠️ WARNING: Use real funds with caution!**

```bash
python3 live_trade.py --config config/ZECUSDT_TEMPLATE.yaml --mode mainnet
```

Options:
- `--config`: Path to strategy config file
- `--mode`: Trading mode (testnet or mainnet)
- `--dry-run`: Simulate trades without execution

## Strategy Overview

This bot implements a **3-Tier Hybrid Trading Strategy** with the following components:

### 1. Score-Driven Engine
- **9 Scenario-Based Strategies**:
  - #1: Strong uptrend
  - #2: Momentum breakout
  - #3: Oversold bounce
  - #4: Volume accumulation
  - #5: Pullback entry
  - #6: Volatility contraction
  - #7: Mean reversion
  - #8: Momentum divergence
  - #9: Consolidation breakout

### 2. Multi-Timeframe Analysis (MTF)
- Base timeframe: 5m
- Higher timeframes: 10m, 15m, 30m, 1h, 4h
- HTF alignment bonuses/penalties in score calculation

### 3. Risk Management
- **PnL Gate**: Degrades or pauses trading based on daily PnL
- **Stop Loss Engine**: Hard stops on excessive losses
- **DCA Engine**: Averages down in oversold conditions
- **Trailing Take Profit**: Dynamic profit taking

### 4. Grid Trading
- Dynamic spread calculation based on ATR and RSI
- 3 volatility bands: near, mid, far
- Up to 7 levels per side

### 5. Arbiter System
- Resolves conflicting signals
- Precedence: exit_reduce > tp_trailing > entry_new > dca

## Configuration

The strategy is highly configurable via `config/ZECUSDT_TEMPLATE.yaml`. Key parameters:

- **Market Filters**: tick_size, lot_size, min_notional
- **Capital Management**: starting_cash_usdt, order_size_quote_usdt
- **Indicators**: RSI, EMA, MACD, ADX, ATR, Bollinger Bands, CMF, VWAP, TSI, Stochastic RSI
- **Score Thresholds**: Entry, hold, and exit scores for each scenario
- **Risk Parameters**: Stop loss levels, DCA thresholds, trailing stops
- **Grid Settings**: Spacing, levels, kill-replace logic

## Reports

After backtesting or live trading, reports are generated in `data/reports/`:

- **PnL Summary**: Total return, win rate, max drawdown, Sharpe ratio
- **Trade Log**: Detailed list of all trades with entry/exit prices
- **Performance Charts**: Equity curve, drawdown chart, trade distribution
- **Risk Metrics**: Daily PnL, exposure, position sizing
- **Score Analysis**: Score distribution, scenario performance

## Safety Features

- **Kill Switch**: Stops trading after consecutive losses
- **Force Sell**: Emergency liquidation on hard stop trigger
- **Daily Reset**: Resets PnL gate and stop loss counters at UTC midnight
- **Position Limits**: Maximum position size and notional value
- **Price Deviation Check**: Cancels and replaces orders if price moves too much

## Logging

Logs are written to `logs/` directory with rotation every 15 minutes:
- Order logs
- Risk management logs
- Fill logs
- Summary logs

## Testing

Run unit tests:

```bash
python3 -m pytest tests/
```

## Disclaimer

**This trading bot is provided for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.**

## License

MIT License

## Support

For issues or questions, please open an issue on the project repository.
