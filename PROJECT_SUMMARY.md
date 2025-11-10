# Project Summary: Binance Trading Bot

This document provides a summary of the Binance Trading Bot project, including its features, architecture, and usage instructions.

## 1. Project Overview

The Binance Trading Bot is a comprehensive Python application designed for automated trading on the Binance exchange. It implements a sophisticated, hybrid trading strategy based on the provided `ZECUSDT_TEMPLATE.yaml` configuration file. The bot supports backtesting, live trading on testnet and mainnet, and detailed performance reporting.

### Key Features:

*   **Hybrid Trading Strategy**: Combines a score-based signal generation engine with a dynamic grid trading system.
*   **Multi-Timeframe (MTF) Analysis**: Incorporates data from higher timeframes (10m, 15m, 30m, 1h, 4h) to improve signal accuracy.
*   **Dynamic Risk Management**: Implements a **PnL Gate** with three states (NORMAL, DEGRADED, PAUSED), **Hard Stop-Loss**, and a **Kill Switch** to protect capital.
*   **Smart DCA (Dollar-Cost Averaging)**: Automatically averages down the entry price with safety guards and a configurable allocation plan, managed by `DCATPEngine`.
*   **Volatility-Adjusted Trailing Take Profit**: Dynamically adjusts the take-profit trigger and trail percentages based on market volatility (ATR), managed by `DCATPEngine`.
*   **Backtesting Engine**: Allows for thorough testing of the strategy on historical data with realistic simulation of fees and slippage.
*   **Live Trading Module**: Supports real-time trading on both Binance testnet and mainnet.
*   **Comprehensive Reporting**: Generates detailed PnL reports, trade logs, and performance charts.

## 2. Architecture

The project is structured into several modules, each responsible for a specific part of the trading logic:

*   `src/indicators`: Contains the technical indicators and the score engine for signal generation, including `MTFAnalysis`.
*   `src/strategy`: Implements the core trading strategies, including the `GridEngine`, `DCATPEngine`, and `RiskManager`.
*   `src/backtest`: Includes the backtesting engine and data loader.
*   `src/live`: Contains the Binance client wrapper and the live trader orchestrator.
*   `src/reporting`: Provides tools for generating PnL reports and visualizations.
*   `src/utils`: Includes utility functions for configuration loading and logging.

## 3. Usage

### 3.1. Installation

1.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure API keys**:

    Copy `config/api_keys.yaml.template` to `config/api_keys.yaml` and fill in your Binance API key and secret.

### 3.2. Backtesting

To run a backtest, use the `backtest.py` script:

```bash
python3 backtest.py --config config/ZECUSDT_TEMPLATE.yaml --start-date 2023-01-01 --end-date 2023-12-31
```

### 3.3. Live Trading

To start live trading, use the `live_trade.py` script:

```bash
# For testnet trading
python3 live_trade.py --config config/ZECUSDT_TEMPLATE.yaml --testnet --dry-run

# For mainnet trading (use with caution!)
python3 live_trade.py --config config/ZECUSDT_TEMPLATE.yaml
```

## 4. Disclaimer

Trading cryptocurrencies involves substantial risk and may not be suitable for all investors. The developers of this project are not responsible for any losses incurred while using this trading bot. **Use at your own risk.**
