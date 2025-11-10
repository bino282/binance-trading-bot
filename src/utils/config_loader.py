"""
Configuration Loader
Loads and validates YAML configuration files for the trading bot.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Loads and manages trading bot configuration."""
    
    def __init__(self, config_path: str):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate required configuration fields."""
        required_fields = ['pair', 'market_filters', 'starting_cash_usdt', 'policy_cfg']
        
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")
        
        # Validate pair config
        pair_fields = ['symbol', 'base', 'quote']
        for field in pair_fields:
            if field not in self.config['pair']:
                raise ValueError(f"Missing required pair field: {field}")
        
        # Validate market filters
        filter_fields = ['tick_size', 'lot_size', 'min_notional']
        for field in filter_fields:
            if field not in self.config['market_filters']:
                raise ValueError(f"Missing required market_filters field: {field}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'pair.symbol')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_symbol(self) -> str:
        """Get trading symbol."""
        return self.config['pair']['symbol']
    
    def get_base_asset(self) -> str:
        """Get base asset."""
        return self.config['pair']['base']
    
    def get_quote_asset(self) -> str:
        """Get quote asset."""
        return self.config['pair']['quote']
    
    def get_starting_capital(self) -> float:
        """Get starting capital in quote currency."""
        return float(self.config['starting_cash_usdt'])
    
    def get_order_size(self) -> float:
        """Get order size in quote currency."""
        return float(self.config['order_size_quote_usdt'])
    
    def get_fee_per_leg(self) -> float:
        """Get trading fee per leg as decimal."""
        return float(self.config['fee_per_leg_pct'])
    
    def get_tick_size(self) -> float:
        """Get minimum price tick size."""
        return float(self.config['market_filters']['tick_size'])
    
    def get_lot_size(self) -> float:
        """Get minimum quantity lot size."""
        return float(self.config['market_filters']['lot_size'])
    
    def get_min_notional(self) -> float:
        """Get minimum notional value."""
        return float(self.config['market_filters']['min_notional'])
    
    def get_kline_interval(self) -> str:
        """Get base kline interval."""
        return self.config.get('runtime_cfg', {}).get('kline_interval', '5m')
    
    def get_indicators_config(self) -> Dict[str, Any]:
        """Get indicators configuration."""
        return self.config.get('policy_cfg', {}).get('indicators', {})
    
    def get_score_engine_config(self) -> Dict[str, Any]:
        """Get score engine configuration."""
        return self.config.get('score_engine', {})
    
    def get_mtf_config(self) -> Dict[str, Any]:
        """Get multi-timeframe configuration."""
        return self.config.get('mtf', {})
    
    def get_arbiter_config(self) -> Dict[str, Any]:
        """Get arbiter configuration."""
        return self.config.get('arbiter', {})
    
    def get_pnl_gate_config(self) -> Dict[str, Any]:
        """Get PnL gate configuration."""
        return self.config.get('policy_cfg', {}).get('pnl_gate', {})
    
    def get_sl_engine_config(self) -> Dict[str, Any]:
        """Get stop loss engine configuration."""
        return self.config.get('policy_cfg', {}).get('sl_engine', {})
    
    def get_dca_config(self) -> Dict[str, Any]:
        """Get DCA configuration."""
        return self.config.get('policy_cfg', {}).get('dca', {})
    
    def get_tp_trailing_config(self) -> Dict[str, Any]:
        """Get trailing take profit configuration."""
        return self.config.get('policy_cfg', {}).get('tp_trailing', {})
    
    def get_grid_config(self) -> Dict[str, Any]:
        """Get grid trading configuration."""
        return self.config.get('policy_cfg', {}).get('grid', {})
    
    def get_spread_engine_config(self) -> Dict[str, Any]:
        """Get spread engine configuration."""
        return self.config.get('policy_cfg', {}).get('spread_engine', {})
    
    def get_score_layer_config(self) -> Dict[str, Any]:
        """Get score layer configuration."""
        return self.config.get('policy_cfg', {}).get('score_layer', {})
    
    def get_scenario_policy(self, scenario: str) -> Optional[Dict[str, Any]]:
        """
        Get scenario-specific policy configuration.
        
        Args:
            scenario: Scenario identifier (e.g., '#1', '#2')
            
        Returns:
            Scenario policy configuration or None
        """
        return self.config.get('policy_cfg', {}).get('scenario_policy', {}).get(scenario)
    
    def is_testnet(self) -> bool:
        """Check if testnet mode is enabled."""
        return self.config.get('pair', {}).get('testnet', True)
    
    def is_backtest_mode(self) -> bool:
        """Check if running in backtest mode."""
        runtime = self.config.get('pair', {}).get('runtime', 'backtest')
        return runtime == 'backtest'
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary."""
        return self.config


def load_api_keys(api_keys_path: str = "config/api_keys.yaml") -> Dict[str, Any]:
    """
    Load API keys from configuration file.
    
    Args:
        api_keys_path: Path to API keys YAML file
        
    Returns:
        API keys configuration
    """
    path = Path(api_keys_path)
    
    if not path.exists():
        raise FileNotFoundError(
            f"API keys file not found: {api_keys_path}\n"
            "Please create config/api_keys.yaml with your Binance API credentials."
        )
    
    with open(path, 'r') as f:
        return yaml.safe_load(f)
