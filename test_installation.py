#!/usr/bin/env python3
"""
Installation Test Script
Validates that all dependencies and modules are properly installed.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Standard library
        import pandas
        print("✓ pandas")
        
        import numpy
        print("✓ numpy")
        
        import yaml
        print("✓ pyyaml")
        
        import matplotlib
        print("✓ matplotlib")
        
        import binance
        print("✓ python-binance")
        
        import colorlog
        print("✓ colorlog")
        
        # Project modules
        from src.utils.config_loader import ConfigLoader
        print("✓ src.utils.config_loader")
        
        from src.utils.logger import get_logger
        print("✓ src.utils.logger")
        
        from src.indicators.technical import TechnicalIndicators
        print("✓ src.indicators.technical")
        
        from src.indicators.score_engine import ScoreEngine
        print("✓ src.indicators.score_engine")
        
        from src.strategy.risk_manager import RiskManager
        print("✓ src.strategy.risk_manager")
        
        from src.backtest.data_loader import DataLoader
        print("✓ src.backtest.data_loader")
        
        from src.backtest.engine import BacktestEngine
        print("✓ src.backtest.engine")
        
        from src.live.binance_client import BinanceClientWrapper
        print("✓ src.live.binance_client")
        
        from src.live.trader import LiveTrader
        print("✓ src.live.trader")
        
        from src.reporting.pnl_report import PnLReportGenerator
        print("✓ src.reporting.pnl_report")
        
        from src.reporting.visualizer import BacktestVisualizer
        print("✓ src.reporting.visualizer")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config_loader import ConfigLoader
        
        config_path = "config/ZECUSDT_TEMPLATE.yaml"
        config = ConfigLoader(config_path)
        
        print(f"✓ Config loaded: {config_path}")
        print(f"  Symbol: {config.get_symbol()}")
        print(f"  Starting Capital: ${config.get_starting_capital():.2f}")
        print(f"  Order Size: ${config.get_order_size():.2f}")
        
        print("\n✅ Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        return False


def test_indicators():
    """Test technical indicators calculation."""
    print("\nTesting technical indicators...")
    
    try:
        import pandas as pd
        import numpy as np
        from src.indicators.technical import TechnicalIndicators
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        data = pd.DataFrame({
            'open': np.random.uniform(40, 50, 100),
            'high': np.random.uniform(50, 60, 100),
            'low': np.random.uniform(30, 40, 100),
            'close': np.random.uniform(40, 50, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        }, index=dates)
        
        indicators = TechnicalIndicators()
        
        # Test RSI
        rsi = indicators.calculate_rsi(data['close'], 14)
        assert len(rsi) == len(data), "RSI length mismatch"
        print("✓ RSI calculation")
        
        # Test EMA
        ema = indicators.calculate_ema(data['close'], 20)
        assert len(ema) == len(data), "EMA length mismatch"
        print("✓ EMA calculation")
        
        # Test MACD
        macd, signal, hist = indicators.calculate_macd(data['close'])
        assert len(macd) == len(data), "MACD length mismatch"
        print("✓ MACD calculation")
        
        # Test ATR
        atr = indicators.calculate_atr(data['high'], data['low'], data['close'])
        assert len(atr) == len(data), "ATR length mismatch"
        print("✓ ATR calculation")
        
        print("\n✅ Indicators test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Indicators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_score_engine():
    """Test score engine."""
    print("\nTesting score engine...")
    
    try:
        import pandas as pd
        import numpy as np
        from src.indicators.score_engine import ScoreEngine
        
        config = {
            'enabled': True,
            'shadow_mode': False,
            'thresholds': {
                '#1': {'entry': 68, 'hold': 58, 'exit': 48}
            },
            'bonus_penalty': {},
            'hysteresis': {'cooldown_bars_5m': 5}
        }
        
        engine = ScoreEngine(config)
        
        # Create sample bar
        bar = pd.Series({
            'close': 45.0,
            'rsi': 55.0,
            'ema_fast': 44.0,
            'ema_mid': 43.0,
            'ema_slow': 42.0,
            'macd': 0.5,
            'macd_hist': 0.2,
            'adx': 25.0,
            'cmf': 0.15,
            'bb_upper': 46.0,
            'bb_middle': 45.0,
            'bb_lower': 44.0,
            'atr_pct': 0.02
        })
        
        # Test scenario calculation
        scenario = engine.calculate_scenario_1_score(bar)
        assert scenario.score >= 0, "Invalid score"
        print(f"✓ Scenario #1 score: {scenario.score:.1f}")
        
        print("\n✅ Score engine test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Score engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_manager():
    """Test risk manager."""
    print("\nTesting risk manager...")
    
    try:
        from src.strategy.risk_manager import RiskManager
        
        config = {
            'pnl_gate': {
                'enable': True,
                'degrade_daily_pnl_pct': -0.01,
                'pause_daily_pnl_pct': -0.02
            },
            'sl_engine': {
                'enable': True,
                'kill_switch': {'enable': True, 'max_consecutive_losses': 3}
            },
            'dca': {
                'enable': True,
                'rsi_threshold_buy': 38
            }
        }
        
        risk_manager = RiskManager(config, 1000.0)
        
        # Test PnL gate
        status = risk_manager.update_pnl_gate(100.0, 0.0, 0.0)
        assert status in ['NORMAL', 'DEGRADED', 'PAUSED'], "Invalid PnL gate status"
        print(f"✓ PnL gate status: {status}")
        
        # Test position sizing
        position_size = risk_manager.calculate_position_size(100.0, 50.0)
        assert position_size > 0, "Invalid position size"
        print(f"✓ Position size: {position_size:.6f}")
        
        # Test DCA check
        can_dca, reason = risk_manager.can_dca(35.0, 100.0, 105.0)
        print(f"✓ DCA check: {can_dca} ({reason})")
        
        print("\n✅ Risk manager test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Risk manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("INSTALLATION TEST")
    print("=" * 80)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Indicators", test_indicators()))
    results.append(("Score Engine", test_score_engine()))
    results.append(("Risk Manager", test_risk_manager()))
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 All tests passed! Installation is complete.")
        print("\nYou can now:")
        print("  1. Run a backtest: python3 backtest.py --help")
        print("  2. Start live trading: python3 live_trade.py --help")
        print("\nSee QUICKSTART.md for detailed instructions.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nTry:")
        print("  1. Reinstall dependencies: pip3 install -r requirements.txt")
        print("  2. Check Python version: python3 --version (need 3.11+)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
