"""
Visualizer
Creates charts and visualizations for backtest results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from ..backtest.engine import BacktestResult


class BacktestVisualizer:
    """Create visualizations for backtest results."""
    
    def __init__(self, result: BacktestResult):
        """
        Initialize visualizer.
        
        Args:
            result: Backtest result object
        """
        self.result = result
        self.equity_curve = result.equity_curve
        self.trades = result.trades
        self.metrics = result.metrics
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_equity_curve(self, save_path: str = None, show: bool = True):
        """
        Plot equity curve.
        
        Args:
            save_path: Path to save figure
            show: Show figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot equity
        ax.plot(self.equity_curve.index, self.equity_curve['equity'], 
                label='Equity', linewidth=2, color='blue')
        
        # Plot starting capital line
        ax.axhline(y=self.metrics['starting_capital'], 
                   color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
        
        # Mark buy/sell trades
        buy_trades = [t for t in self.trades if t.side == 'BUY']
        sell_trades = [t for t in self.trades if t.side == 'SELL']
        
        if buy_trades:
            buy_times = [t.timestamp for t in buy_trades]
            buy_equity = [self.equity_curve.loc[t.timestamp, 'equity'] 
                         if t.timestamp in self.equity_curve.index else None 
                         for t in buy_trades]
            ax.scatter(buy_times, buy_equity, color='green', marker='^', 
                      s=100, alpha=0.7, label='Buy', zorder=5)
        
        if sell_trades:
            sell_times = [t.timestamp for t in sell_trades]
            sell_equity = [self.equity_curve.loc[t.timestamp, 'equity'] 
                          if t.timestamp in self.equity_curve.index else None 
                          for t in sell_trades]
            ax.scatter(sell_times, sell_equity, color='red', marker='v', 
                      s=100, alpha=0.7, label='Sell', zorder=5)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Equity curve saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_drawdown(self, save_path: str = None, show: bool = True):
        """
        Plot drawdown chart.
        
        Args:
            save_path: Path to save figure
            show: Show figure
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot drawdown
        drawdown_pct = self.equity_curve['drawdown'] * 100
        ax.fill_between(self.equity_curve.index, drawdown_pct, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax.plot(self.equity_curve.index, drawdown_pct, 
               color='red', linewidth=1.5)
        
        # Mark max drawdown
        max_dd_idx = drawdown_pct.idxmin()
        max_dd_val = drawdown_pct.min()
        ax.scatter([max_dd_idx], [max_dd_val], color='darkred', 
                  s=200, marker='v', zorder=5, label=f'Max DD: {max_dd_val:.2f}%')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Drawdown Chart', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Drawdown chart saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_pnl_distribution(self, save_path: str = None, show: bool = True):
        """
        Plot PnL distribution histogram.
        
        Args:
            save_path: Path to save figure
            show: Show figure
        """
        sell_trades = [t for t in self.trades if t.side == 'SELL']
        
        if not sell_trades:
            print("No sell trades to plot")
            return
        
        pnls = [t.pnl for t in sell_trades]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram
        ax.hist(pnls, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        
        # Add mean line
        mean_pnl = np.mean(pnls)
        ax.axvline(x=mean_pnl, color='green', linestyle='--', linewidth=2, 
                  label=f'Mean: ${mean_pnl:.2f}')
        
        ax.set_xlabel('PnL ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('PnL Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PnL distribution saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_scenario_performance(self, save_path: str = None, show: bool = True):
        """
        Plot scenario performance bar chart.
        
        Args:
            save_path: Path to save figure
            show: Show figure
        """
        sell_trades = [t for t in self.trades if t.side == 'SELL' and t.scenario]
        
        if not sell_trades:
            print("No scenario data to plot")
            return
        
        # Aggregate by scenario
        scenario_pnl = {}
        for trade in sell_trades:
            if trade.scenario not in scenario_pnl:
                scenario_pnl[trade.scenario] = 0.0
            scenario_pnl[trade.scenario] += trade.pnl
        
        scenarios = list(scenario_pnl.keys())
        pnls = list(scenario_pnl.values())
        
        # Sort by PnL
        sorted_data = sorted(zip(scenarios, pnls), key=lambda x: x[1], reverse=True)
        scenarios, pnls = zip(*sorted_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color bars based on positive/negative
        colors = ['green' if p > 0 else 'red' for p in pnls]
        
        ax.bar(scenarios, pnls, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        ax.set_xlabel('Scenario', fontsize=12)
        ax.set_ylabel('Total PnL ($)', fontsize=12)
        ax.set_title('Scenario Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scenario performance saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_daily_returns(self, save_path: str = None, show: bool = True):
        """
        Plot daily returns.
        
        Args:
            save_path: Path to save figure
            show: Show figure
        """
        if 'daily_pnl_pct' not in self.equity_curve.columns:
            print("No daily returns data available")
            return
        
        daily_returns = self.equity_curve['daily_pnl_pct'].dropna() * 100
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot daily returns
        colors = ['green' if r > 0 else 'red' for r in daily_returns]
        ax.bar(daily_returns.index, daily_returns, color=colors, alpha=0.6, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Daily Return (%)', fontsize=12)
        ax.set_title('Daily Returns', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Daily returns saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_dashboard(self, save_path: str = None, show: bool = True):
        """
        Create comprehensive dashboard with multiple charts.
        
        Args:
            save_path: Path to save figure
            show: Show figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.equity_curve.index, self.equity_curve['equity'], 
                linewidth=2, color='blue')
        ax1.axhline(y=self.metrics['starting_capital'], 
                   color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Equity ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, 0])
        drawdown_pct = self.equity_curve['drawdown'] * 100
        ax2.fill_between(self.equity_curve.index, drawdown_pct, 0, 
                        color='red', alpha=0.3)
        ax2.plot(self.equity_curve.index, drawdown_pct, color='red', linewidth=1)
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. PnL Distribution
        ax3 = fig.add_subplot(gs[1, 1])
        sell_trades = [t for t in self.trades if t.side == 'SELL']
        if sell_trades:
            pnls = [t.pnl for t in sell_trades]
            ax3.hist(pnls, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('PnL Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('PnL ($)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Scenario Performance
        ax4 = fig.add_subplot(gs[2, 0])
        scenario_trades = [t for t in self.trades if t.side == 'SELL' and t.scenario]
        if scenario_trades:
            scenario_pnl = {}
            for trade in scenario_trades:
                if trade.scenario not in scenario_pnl:
                    scenario_pnl[trade.scenario] = 0.0
                scenario_pnl[trade.scenario] += trade.pnl
            
            scenarios = list(scenario_pnl.keys())
            pnls = list(scenario_pnl.values())
            colors = ['green' if p > 0 else 'red' for p in pnls]
            ax4.bar(scenarios, pnls, color=colors, alpha=0.7, edgecolor='black')
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.set_title('Scenario Performance', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Scenario')
        ax4.set_ylabel('Total PnL ($)')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. Metrics Table
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        metrics_text = [
            ['Metric', 'Value'],
            ['Total Return', f"{self.metrics['total_return_pct']:.2f}%"],
            ['Win Rate', f"{self.metrics['win_rate']:.2%}"],
            ['Max Drawdown', f"{self.metrics['max_drawdown']:.2%}"],
            ['Sharpe Ratio', f"{self.metrics['sharpe_ratio']:.2f}"],
            ['Profit Factor', f"{self.metrics['profit_factor']:.2f}"],
            ['Total Trades', f"{self.metrics['total_trades']}"],
        ]
        
        table = ax5.table(cellText=metrics_text, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax5.set_title('Key Metrics', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle('Backtest Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_all_charts(self, output_dir: str, prefix: str = "backtest"):
        """
        Save all charts to directory.
        
        Args:
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual charts
        self.plot_equity_curve(
            save_path=str(output_path / f"{prefix}_equity_{timestamp}.png"),
            show=False
        )
        
        self.plot_drawdown(
            save_path=str(output_path / f"{prefix}_drawdown_{timestamp}.png"),
            show=False
        )
        
        self.plot_pnl_distribution(
            save_path=str(output_path / f"{prefix}_pnl_dist_{timestamp}.png"),
            show=False
        )
        
        self.plot_scenario_performance(
            save_path=str(output_path / f"{prefix}_scenarios_{timestamp}.png"),
            show=False
        )
        
        self.plot_daily_returns(
            save_path=str(output_path / f"{prefix}_daily_returns_{timestamp}.png"),
            show=False
        )
        
        # Save dashboard
        self.create_dashboard(
            save_path=str(output_path / f"{prefix}_dashboard_{timestamp}.png"),
            show=False
        )
        
        print(f"\nAll charts saved to: {output_path}")
