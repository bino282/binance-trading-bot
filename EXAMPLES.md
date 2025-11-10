# Usage Examples

This document provides practical examples of using the Binance Trading Bot.

## Table of Contents

1. [Backtesting Examples](#backtesting-examples)
2. [Live Trading Examples](#live-trading-examples)
3. [Configuration Customization](#configuration-customization)
4. [Report Analysis](#report-analysis)

---

## Backtesting Examples

### Example 1: Basic Backtest

Run a simple backtest for 3 months:

```bash
python3 backtest.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --output data/reports
```

**Output:**
- Trade log CSV
- Equity curve chart
- PnL distribution histogram
- Scenario performance breakdown
- Summary report

### Example 2: Backtest with Custom Capital

Override starting capital:

```bash
python3 backtest.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-06-30 \
  --initial-capital 5000 \
  --output data/reports
```

### Example 3: Backtest with Saved Data

Save historical data for reuse:

```bash
# First run - fetch and save data
python3 backtest.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --save-data \
  --output data/reports

# Subsequent runs - use saved data
python3 backtest.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --data-file data/historical/ZECUSDT_5m_2024-01-01_2024-12-31.csv \
  --output data/reports
```

### Example 4: Quick Backtest (No Charts)

Skip chart generation for faster results:

```bash
python3 backtest.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --no-charts \
  --output data/reports
```

---

## Live Trading Examples

### Example 1: Testnet Trading (Recommended First)

Start with testnet to test without risk:

```bash
python3 live_trade.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --mode testnet
```

**What happens:**
- Connects to Binance testnet
- Fetches real-time market data
- Generates trading signals
- Executes trades with testnet funds
- Logs all activity to `logs/`

Press `Ctrl+C` to stop gracefully.

### Example 2: Dry Run Mode

Simulate trades without execution:

```bash
python3 live_trade.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --mode testnet \
  --dry-run
```

**Use case:** Test signal generation and strategy logic without placing orders.

### Example 3: Mainnet Trading (Real Money)

**⚠️ WARNING: Use real funds with extreme caution!**

```bash
python3 live_trade.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --mode mainnet
```

You'll be prompted to confirm:
```
Type 'YES' to confirm and proceed:
```

### Example 4: Custom API Keys Path

Use a different API keys file:

```bash
python3 live_trade.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --mode testnet \
  --api-keys /path/to/my_api_keys.yaml
```

---

## Configuration Customization

### Example 1: Adjust Position Size

Edit `config/ZECUSDT_TEMPLATE.yaml`:

```yaml
starting_cash_usdt: 5000.0      # Increase starting capital
order_size_quote_usdt: 150.0    # Increase order size
```

### Example 2: Modify Risk Parameters

```yaml
policy_cfg:
  pnl_gate:
    degrade_daily_pnl_pct: -0.0030    # More aggressive (was -0.0040)
    pause_daily_pnl_pct: -0.0040      # More aggressive (was -0.0050)
  
  sl_engine:
    capital_sl:
      hard_stop_daily_pnl_pct: -0.0060  # Tighter stop (was -0.0070)
```

### Example 3: Adjust Score Thresholds

Make entry signals more selective:

```yaml
policy_cfg:
  score_layer:
    thresholds:
      '#1':
        entry: 75    # Higher threshold (was 68)
        hold: 65     # Higher threshold (was 58)
        exit: 50     # Higher threshold (was 48)
```

### Example 4: Disable Features

Turn off DCA or grid trading:

```yaml
policy_cfg:
  dca:
    enable: false    # Disable DCA
  
  grid:
    enable: false    # Disable grid trading
```

---

## Report Analysis

### Understanding Backtest Reports

After running a backtest, you'll get several files:

#### 1. Summary Text (`ZECUSDT_summary_*.txt`)

```
CAPITAL METRICS
Starting Capital:        $3,000.00
Final Equity:            $3,450.00
Total Return:            $450.00 (15.00%)
Max Drawdown:            -8.50%
Sharpe Ratio:            1.85

TRADE STATISTICS
Total Trades:            45
Buy Trades:              23
Sell Trades:             22
Winning Trades:          15 (68.18%)
Losing Trades:           7
```

**Key Metrics:**
- **Total Return:** Overall profit/loss percentage
- **Max Drawdown:** Largest peak-to-trough decline
- **Sharpe Ratio:** Risk-adjusted return (>1 is good, >2 is excellent)
- **Win Rate:** Percentage of profitable trades

#### 2. Trade Log (`ZECUSDT_trades_*.csv`)

| trade_id | timestamp | side | price | quantity | pnl | scenario | reason |
|----------|-----------|------|-------|----------|-----|----------|--------|
| 1 | 2024-01-15 10:30 | BUY | 45.20 | 2.0 | 0 | #1 | Entry |
| 2 | 2024-01-15 14:45 | SELL | 46.50 | 2.0 | 2.60 | #1 | Exit |

**Analysis:**
- Review which scenarios are most profitable
- Identify optimal entry/exit times
- Check average holding period

#### 3. Scenario Performance (`ZECUSDT_scenarios_*.csv`)

| scenario | trades | wins | losses | total_pnl | win_rate |
|----------|--------|------|--------|-----------|----------|
| #1 | 12 | 9 | 3 | $125.50 | 75.00% |
| #2 | 8 | 5 | 3 | $85.20 | 62.50% |
| #7 | 5 | 3 | 2 | $45.80 | 60.00% |

**Insights:**
- Scenario #1 (Strong uptrend) performs best
- Consider increasing allocation to top scenarios
- May want to disable underperforming scenarios

#### 4. Visual Dashboard (`ZECUSDT_dashboard_*.png`)

The dashboard includes:
- **Equity Curve:** Shows capital growth over time
- **Drawdown Chart:** Visualizes risk periods
- **PnL Distribution:** Histogram of trade outcomes
- **Scenario Performance:** Bar chart of scenario profitability

### Analyzing Live Trading Logs

Live trading logs are in `logs/live_trader_YYYYMMDD.log`:

```
2024-01-15 10:30:15 - SIGNAL: entry | Score: 72.5 | Scenario: #1
2024-01-15 10:30:16 - ORDER: MARKET BUY 2.000000 @ 45.20
2024-01-15 10:30:17 - FILL: BUY 2.000000 @ 45.20 | Fee: 0.090400
2024-01-15 14:45:22 - SIGNAL: exit | Score: 42.0 | Scenario: #1
2024-01-15 14:45:23 - ORDER: MARKET SELL 2.000000 @ 46.50
2024-01-15 14:45:24 - FILL: SELL 2.000000 @ 46.50 | Fee: 0.093000
2024-01-15 14:45:24 - PnL: Realized=2.60 | Unrealized=0.00 | Total=2.60
```

**Monitor:**
- Signal generation frequency
- Order execution prices
- Slippage and fees
- Risk events (PnL gate, stop losses)

---

## Advanced Examples

### Example 1: Multi-Period Backtesting

Test strategy across different market conditions:

```bash
# Bull market period
python3 backtest.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --output data/reports/bull

# Bear market period
python3 backtest.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --start-date 2024-04-01 \
  --end-date 2024-06-30 \
  --output data/reports/bear

# Sideways market period
python3 backtest.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --start-date 2024-07-01 \
  --end-date 2024-09-30 \
  --output data/reports/sideways
```

Compare results to understand strategy performance in different conditions.

### Example 2: Parameter Optimization

Create multiple config files with different parameters:

```bash
# Test conservative settings
python3 backtest.py \
  --config config/ZECUSDT_CONSERVATIVE.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output data/reports/conservative

# Test aggressive settings
python3 backtest.py \
  --config config/ZECUSDT_AGGRESSIVE.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --output data/reports/aggressive
```

Compare Sharpe ratios and max drawdowns to find optimal settings.

### Example 3: Running Multiple Bots

Run different strategies simultaneously:

```bash
# Terminal 1: ZEC/USDT bot
python3 live_trade.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --mode testnet

# Terminal 2: BNB/USDT bot (if you create this config)
python3 live_trade.py \
  --config config/BNBUSDT_TEMPLATE.yaml \
  --mode testnet
```

**Note:** Ensure you have sufficient capital for multiple bots.

---

## Tips and Best Practices

### Backtesting Tips

1. **Use sufficient data:** At least 3-6 months for reliable results
2. **Test multiple periods:** Bull, bear, and sideways markets
3. **Watch for overfitting:** Good backtest ≠ good live performance
4. **Consider fees:** Backtest includes realistic 0.1% fees
5. **Check drawdowns:** Max drawdown is often more important than returns

### Live Trading Tips

1. **Start small:** Use minimum position sizes initially
2. **Monitor regularly:** Check logs and positions daily
3. **Set alerts:** Use external monitoring for critical events
4. **Keep cash reserves:** Don't use 100% of capital
5. **Test on testnet first:** Always validate on testnet before mainnet

### Risk Management Tips

1. **Never risk more than you can afford to lose**
2. **Use stop losses:** Hard stops are your safety net
3. **Diversify:** Don't put all capital in one bot/pair
4. **Review regularly:** Analyze performance weekly
5. **Adjust parameters:** Optimize based on market conditions

---

## Troubleshooting

### Backtest runs but shows poor results

**Solution:** Adjust strategy parameters or test different time periods.

### Live bot not placing trades

**Possible causes:**
- PnL gate is paused
- Score thresholds not met
- Insufficient balance
- API permissions issue

**Check:** Review logs in `logs/` directory

### Orders failing to execute

**Possible causes:**
- Below minimum notional value
- Invalid quantity (lot size)
- Insufficient balance
- API rate limit

**Solution:** Check exchange info and adjust order sizes

---

## Next Steps

- Experiment with different configurations
- Analyze backtest results to optimize parameters
- Test thoroughly on testnet before going live
- Monitor live trading closely and adjust as needed
- Keep learning about trading strategies and risk management

Happy Trading! 📈
