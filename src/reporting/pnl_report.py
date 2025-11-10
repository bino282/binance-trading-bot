"""
PnL Report Generator
Creates comprehensive PnL reports and trade analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
from pathlib import Path

from ..backtest.engine import BacktestResult, Trade


class PnLReportGenerator:
    """Generate detailed PnL reports from backtest results."""
    
    def __init__(self, result: BacktestResult, config: dict):
        """
        Initialize report generator.
        
        Args:
            result: Backtest result object
            config: Strategy configuration dictionary
        """
        self.result = result
        self.config = config
        self.trades = result.trades
        self.equity_curve = result.equity_curve
        self.metrics = result.metrics
    
    def _calculate_trade_stats(self) -> dict:
        """Calculate derived trade statistics."""
        buy_trades = [t for t in self.trades if t.side == 'BUY']
        sell_trades = [t for t in self.trades if t.side == 'SELL']
        
        total_fees = sum(t.fee for t in self.trades)
        
        # PnL is only calculated on SELL trades
        pnl_trades = [t for t in self.trades if t.side == 'SELL']
        winning_trades = [t for t in pnl_trades if t.pnl > 0]
        losing_trades = [t for t in pnl_trades if t.pnl < 0]
        
        return {
            'total_trades': len(pnl_trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'total_fees': total_fees
        }

    def generate_trade_log(self) -> pd.DataFrame:
        """
        Generate detailed trade log.
        
        Returns:
            DataFrame with all trades
        """
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for i, trade in enumerate(self.trades):
            trade_data.append({
                'trade_id': i + 1,
                'timestamp': trade.timestamp,
                'side': trade.side,
                'price': trade.price,
                'quantity': trade.quantity,
                'value': trade.value,
                'fee': trade.fee,
                'pnl': trade.pnl if trade.side == 'SELL' else 0,
                'scenario': trade.scenario,
                'score': trade.score,
                'reason': trade.reason
            })
        
        df = pd.DataFrame(trade_data)
        return df
    
    def generate_daily_summary(self) -> pd.DataFrame:
        """
        Generate daily PnL summary.
        
        Returns:
            DataFrame with daily statistics
        """
        if self.equity_curve.empty:
            return pd.DataFrame()
        
        # Resample to daily
        daily = self.equity_curve.resample('D').agg({
            'equity': 'last',
            'cash': 'last',
            'unrealized_pnl': 'last',
        })
        
        # Calculate daily PnL and return
        daily['daily_pnl'] = daily['equity'].diff()
        daily['daily_return_pct'] = daily['equity'].pct_change() * 100
        
        # Calculate drawdown
        peak = daily['equity'].expanding(min_periods=1).max()
        daily['drawdown'] = (daily['equity'] - peak) / peak
        
        return daily.dropna(subset=['equity'])
    
    def generate_scenario_performance(self) -> pd.DataFrame:
        """
        Generate performance breakdown by scenario.
        
        Returns:
            DataFrame with scenario statistics
        """
        sell_trades = [t for t in self.trades if t.side == 'SELL' and t.scenario]
        
        if not sell_trades:
            return pd.DataFrame()
        
        # Group by scenario
        scenario_data = {}
        for trade in sell_trades:
            if trade.scenario not in scenario_data:
                scenario_data[trade.scenario] = {
                    'trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0.0,
                    'avg_pnl': 0.0,
                    'win_rate': 0.0
                }
            
            scenario_data[trade.scenario]['trades'] += 1
            scenario_data[trade.scenario]['total_pnl'] += trade.pnl
            
            if trade.pnl > 0:
                scenario_data[trade.scenario]['wins'] += 1
            else:
                scenario_data[trade.scenario]['losses'] += 1
        
        # Calculate metrics
        for scenario in scenario_data:
            data = scenario_data[scenario]
            data['avg_pnl'] = data['total_pnl'] / data['trades'] if data['trades'] > 0 else 0
            data['win_rate'] = data['wins'] / data['trades'] if data['trades'] > 0 else 0
        
        df = pd.DataFrame.from_dict(scenario_data, orient='index')
        df.index.name = 'scenario'
        df = df.sort_values('total_pnl', ascending=False)
        
        return df
    
    def generate_monthly_summary(self) -> pd.DataFrame:
        """
        Generate monthly PnL summary.
        
        Returns:
            DataFrame with monthly statistics
        """
        if self.equity_curve.empty:
            return pd.DataFrame()
        
        # Resample to monthly
        monthly = self.equity_curve.resample('M').agg({
            'equity': 'last',
            'daily_pnl': 'sum',
            'drawdown': 'min'
        })
        
        monthly.rename(columns={'daily_pnl': 'monthly_pnl'}, inplace=True)
        monthly['monthly_return_pct'] = monthly['equity'].pct_change() * 100
        
        return monthly
    
    def generate_summary_text(self) -> str:
        """
        Generate text summary of backtest results.
        
        Returns:
            Formatted text summary
        """
        m = self.metrics
        
        summary = []
        summary.append("=" * 80)
        summary.append("BACKTEST SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        
        # Capital metrics
        summary.append("CAPITAL METRICS")
        summary.append("-" * 80)
        summary.append(f"Starting Capital:        ${self.config['starting_cash_usdt']:,.2f}")
        summary.append(f"Final Equity:            {m['Final Equity']}")
        summary.append(f"Total Return:            {m['Total Return']}")
        summary.append(f"Max Drawdown:            {m['Max Drawdown']}")
        summary.append(f"Sharpe Ratio:            {m['Sharpe Ratio']}")
        summary.append("")
        
        # Trade statistics
        summary.append("TRADE STATISTICS")
        summary.append("-" * 80)
        
        # Calculate derived stats
        stats = self._calculate_trade_stats()
        
        summary.append(f"Total Trades (Closed):   {m['Total Trades']}")
        summary.append(f"Buy Trades:              {stats['buy_trades']}")
        summary.append(f"Sell Trades:             {stats['sell_trades']}")
        summary.append(f"Winning Trades:          {stats['winning_trades']} ({m['Win Rate']})")
        summary.append(f"Losing Trades:           {stats['losing_trades']}")
        summary.append("")
        
        # PnL statistics
        summary.append("PNL STATISTICS")
        summary.append("-" * 80)
        summary.append(f"Average Win:             {m['Avg Win']}")
        summary.append(f"Average Loss:            {m['Avg Loss']}")
        summary.append(f"Profit Factor:           {m['Profit Factor']}")
        summary.append(f"Total Fees:              ${stats['total_fees']:,.2f}")
        summary.append("")

        
        summary.append("=" * 80)
        
        return "\n".join(summary)
    
    def save_report(self, output_dir: str, prefix: str = "backtest"):
        """
        Save complete report to files.
        
        Args:
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary text
        summary_file = output_path / f"{prefix}_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(self.generate_summary_text())
        
        print(f"Summary saved to: {summary_file}")
        
        # Save trade log
        trade_log = self.generate_trade_log()
        if not trade_log.empty:
            trade_log_file = output_path / f"{prefix}_trades_{timestamp}.csv"
            trade_log.to_csv(trade_log_file, index=False)
            print(f"Trade log saved to: {trade_log_file}")
        
        # Save daily summary
        daily_summary = self.generate_daily_summary()
        if not daily_summary.empty:
            daily_file = output_path / f"{prefix}_daily_{timestamp}.csv"
            daily_summary.to_csv(daily_file)
            print(f"Daily summary saved to: {daily_file}")
        
        # Save scenario performance
        scenario_perf = self.generate_scenario_performance()
        if not scenario_perf.empty:
            scenario_file = output_path / f"{prefix}_scenarios_{timestamp}.csv"
            scenario_perf.to_csv(scenario_file)
            print(f"Scenario performance saved to: {scenario_file}")
        
        # Save equity curve
        if not self.equity_curve.empty:
            equity_file = output_path / f"{prefix}_equity_{timestamp}.csv"
            self.equity_curve.to_csv(equity_file)
            print(f"Equity curve saved to: {equity_file}")
        
        # Save metrics as JSON
        import json
        metrics_file = output_path / f"{prefix}_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        print(f"Metrics saved to: {metrics_file}")
    
    def print_summary(self):
        """Print summary to console."""
        print(self.generate_summary_text())
        
        # Print scenario performance
        scenario_perf = self.generate_scenario_performance()
        if not scenario_perf.empty:
            print("\nSCENARIO PERFORMANCE")
            print("-" * 80)
            print(scenario_perf.to_string())
            print("")
    
    def get_key_metrics(self) -> Dict:
        """
        Get key metrics for quick access.
        
        Returns:
            Dictionary of key metrics
        """
        return {
            'total_return_pct': self.metrics['total_return_pct'],
            'win_rate': self.metrics['win_rate'],
            'max_drawdown': self.metrics['max_drawdown'],
            'sharpe_ratio': self.metrics['sharpe_ratio'],
            'profit_factor': self.metrics['profit_factor'],
            'total_trades': self.metrics['total_trades']
        }


def create_markdown_report(result: BacktestResult, config: Dict, output_file: str):
    """
    Create a comprehensive Markdown report.
    
    Args:
        result: Backtest result
        output_file: Output file path
    """
    generator = PnLReportGenerator(result, config)
    m = result.metrics
    
    lines = []
    lines.append("# Backtest Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Total Return:** {m['total_return_pct']:.2f}%")
    lines.append(f"- **Win Rate:** {m['win_rate']:.2%}")
    lines.append(f"- **Max Drawdown:** {m['max_drawdown']:.2%}")
    lines.append(f"- **Sharpe Ratio:** {m['sharpe_ratio']:.2f}")
    lines.append(f"- **Profit Factor:** {m['profit_factor']:.2f}")
    lines.append("")
    
    # Capital Metrics
    lines.append("## Capital Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Starting Capital | ${m['starting_capital']:,.2f} |")
    lines.append(f"| Final Equity | ${m['final_equity']:,.2f} |")
    lines.append(f"| Total Return | ${m['final_equity'] - m['starting_capital']:,.2f} |")
    lines.append(f"| Return % | {m['total_return_pct']:.2f}% |")
    lines.append(f"| Max Drawdown | {m['max_drawdown']:.2%} |")
    lines.append("")
    
    # Trade Statistics
    lines.append("## Trade Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Trades | {m['total_trades']} |")
    lines.append(f"| Buy Trades | {m['buy_trades']} |")
    lines.append(f"| Sell Trades | {m['sell_trades']} |")
    lines.append(f"| Winning Trades | {m['winning_trades']} ({m['win_rate']:.2%}) |")
    lines.append(f"| Losing Trades | {m['losing_trades']} |")
    lines.append(f"| Average Win | ${m['avg_win']:.2f} |")
    lines.append(f"| Average Loss | ${m['avg_loss']:.2f} |")
    lines.append(f"| Profit Factor | {m['profit_factor']:.2f} |")
    lines.append("")
    
    # Scenario Performance
    scenario_perf = generator.generate_scenario_performance()
    if not scenario_perf.empty:
        lines.append("## Scenario Performance")
        lines.append("")
        lines.append("| Scenario | Trades | Wins | Losses | Total PnL | Avg PnL | Win Rate |")
        lines.append("|----------|--------|------|--------|-----------|---------|----------|")
        
        for scenario, row in scenario_perf.iterrows():
            lines.append(
                f"| {scenario} | {row['trades']:.0f} | {row['wins']:.0f} | {row['losses']:.0f} | "
                f"${row['total_pnl']:.2f} | ${row['avg_pnl']:.2f} | {row['win_rate']:.2%} |"
            )
        lines.append("")
    
    # Risk Metrics
    lines.append("## Risk Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Sharpe Ratio | {m['sharpe_ratio']:.2f} |")
    lines.append(f"| Max Drawdown | {m['max_drawdown']:.2%} |")
    lines.append(f"| Total Fees | ${m['total_fees']:.2f} |")
    lines.append("")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Markdown report saved to: {output_file}")
