# Quick Start Guide

Get started with the Binance Trading Bot in 5 minutes!

## Prerequisites

- Python 3.11 or higher
- Binance account (for live trading)
- Basic understanding of trading concepts

## Step 1: Install Dependencies

```bash
cd binance-trading-bot
pip3 install -r requirements.txt
```

## Step 2: Run a Backtest

Test the strategy with historical data:

```bash
python3 backtest.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --output data/reports
```

This will:
- Fetch historical ZEC/USDT data from Binance
- Run the backtest with the configured strategy
- Generate reports and charts in `data/reports/`

## Step 3: Review Results

Check the generated reports:

```bash
ls -lh data/reports/
```

You'll find:
- `ZECUSDT_summary_*.txt` - Text summary of results
- `ZECUSDT_trades_*.csv` - Detailed trade log
- `ZECUSDT_equity_*.csv` - Equity curve data
- `ZECUSDT_dashboard_*.png` - Visual dashboard
- `ZECUSDT_report_*.md` - Markdown report

## Step 4: Configure API Keys (for Live Trading)

Create `config/api_keys.yaml`:

```bash
cp config/api_keys.yaml.template config/api_keys.yaml
```

Edit `config/api_keys.yaml` and add your Binance API credentials:

```yaml
binance:
  testnet:
    api_key: "YOUR_TESTNET_API_KEY"
    api_secret: "YOUR_TESTNET_API_SECRET"
  mainnet:
    api_key: "YOUR_MAINNET_API_KEY"
    api_secret: "YOUR_MAINNET_API_SECRET"
```

**Get API Keys:**
- Testnet: https://testnet.binance.vision/
- Mainnet: https://www.binance.com/en/my/settings/api-management

## Step 5: Test on Testnet

Run live trading on testnet (no real money):

```bash
python3 live_trade.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --mode testnet
```

Press `Ctrl+C` to stop.

## Step 6: Go Live (Optional)

**⚠️ WARNING: This uses real money!**

```bash
python3 live_trade.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --mode mainnet
```

You'll be asked to confirm before trading starts.

## Dry Run Mode

Test without executing trades:

```bash
python3 live_trade.py \
  --config config/ZECUSDT_TEMPLATE.yaml \
  --mode testnet \
  --dry-run
```

## Common Issues

### Issue: "Module not found"
**Solution:** Make sure you're in the project directory and dependencies are installed:
```bash
cd binance-trading-bot
pip3 install -r requirements.txt
```

### Issue: "API keys file not found"
**Solution:** Create `config/api_keys.yaml` from the template:
```bash
cp config/api_keys.yaml.template config/api_keys.yaml
```

### Issue: "Connection failed"
**Solution:** Check your internet connection and API credentials.

### Issue: "Insufficient balance"
**Solution:** Deposit funds to your Binance account (testnet or mainnet).

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize strategy parameters in `config/ZECUSDT_TEMPLATE.yaml`
- Review the code in `src/` to understand the implementation
- Create your own strategy configuration for different trading pairs

## Support

For questions or issues:
1. Check the [README.md](README.md) documentation
2. Review the code comments
3. Check Binance API documentation

## Safety Tips

✅ **DO:**
- Start with testnet
- Use small position sizes
- Monitor your bot regularly
- Set stop losses
- Test thoroughly before going live

❌ **DON'T:**
- Trade with money you can't afford to lose
- Leave the bot unattended for long periods
- Share your API keys
- Commit API keys to version control
- Skip backtesting

Happy Trading! 🚀
