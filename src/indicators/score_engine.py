"""
Score Engine
Implements the score-driven signal generation system with 9 scenario-based strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ScenarioScore:
    """Container for scenario score and details."""
    scenario_id: str
    score: float
    entry_threshold: float
    hold_threshold: float
    exit_threshold: float
    components: Dict[str, float]
    bonuses: Dict[str, float]
    penalties: Dict[str, float]


class ScoreEngine:
    """
    Score-driven signal generation engine.
    Evaluates market conditions across 9 scenarios and generates trading signals.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the score engine.
        
        Args:
            config: Score layer configuration from strategy config
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.shadow_mode = config.get('shadow_mode', False)
        self.thresholds = config.get('thresholds', {})
        self.bonus_penalty = config.get('bonus_penalty', {})
        self.hysteresis_cooldown = config.get('hysteresis', {}).get('cooldown_bars_5m', 5)
        
        self.last_signal_bar = -999
        self.current_scenario: Optional[str] = None
    
    def calculate_scenario_1_score(self, row: pd.Series, htf_data: Optional[Dict] = None) -> ScenarioScore:
        """
        Scenario #1: Strong uptrend
        - Price above EMAs in order (fast > mid > slow)
        - Strong momentum (MACD positive, RSI 50-70)
        - Moderate volume
        """
        score = 0.0
        components = {}
        bonuses = {}
        penalties = {}
        
        # EMA alignment (30 points)
        if row['close'] > row['ema_fast'] > row['ema_mid'] > row['ema_slow']:
            ema_score = 30
        elif row['close'] > row['ema_fast'] > row['ema_mid']:
            ema_score = 20
        elif row['close'] > row['ema_fast']:
            ema_score = 10
        else:
            ema_score = 0
        components['ema_alignment'] = ema_score
        score += ema_score
        
        # MACD momentum (20 points)
        if row['macd'] > 0 and row['macd_hist'] > 0:
            macd_score = 20
        elif row['macd'] > 0:
            macd_score = 10
        else:
            macd_score = 0
        components['macd_momentum'] = macd_score
        score += macd_score
        
        # RSI in uptrend zone (20 points)
        rsi = row['rsi']
        if 55 <= rsi <= 70:
            rsi_score = 20
        elif 50 <= rsi <= 75:
            rsi_score = 15
        elif 45 <= rsi < 50:
            rsi_score = 10
        else:
            rsi_score = 0
        components['rsi_zone'] = rsi_score
        score += rsi_score
        
        # ADX trend strength (15 points)
        adx = row['adx']
        if adx > 25:
            adx_score = 15
        elif adx > 20:
            adx_score = 10
        else:
            adx_score = 5
        components['adx_strength'] = adx_score
        score += adx_score
        
        # CMF volume confirmation (15 points)
        if row['cmf'] > 0.1:
            cmf_score = 15
        elif row['cmf'] > 0:
            cmf_score = 10
        else:
            cmf_score = 0
        components['cmf_volume'] = cmf_score
        score += cmf_score
        
        thresholds = self.thresholds.get('#1', {})
        return ScenarioScore(
            scenario_id='#1',
            score=score,
            entry_threshold=thresholds.get('entry', 68),
            hold_threshold=thresholds.get('hold', 58),
            exit_threshold=thresholds.get('exit', 48),
            components=components,
            bonuses=bonuses,
            penalties=penalties
        )
    
    def calculate_scenario_2_score(self, row: pd.Series, htf_data: Optional[Dict] = None) -> ScenarioScore:
        """
        Scenario #2: Momentum breakout
        - Price breaking above resistance
        - Strong MACD histogram
        - High volume
        """
        score = 0.0
        components = {}
        bonuses = {}
        penalties = {}
        
        # Price vs Bollinger Bands (25 points)
        if row['close'] > row['bb_upper']:
            bb_score = 25
        elif row['close'] > row['bb_middle']:
            bb_score = 15
        else:
            bb_score = 0
        components['bb_position'] = bb_score
        score += bb_score
        
        # MACD histogram strength (25 points)
        if row['macd_hist_z'] > 1.5:
            macd_hist_score = 25
        elif row['macd_hist_z'] > 1.0:
            macd_hist_score = 18
        elif row['macd_hist_z'] > 0.5:
            macd_hist_score = 10
        else:
            macd_hist_score = 0
        components['macd_hist'] = macd_hist_score
        score += macd_hist_score
        
        # Volume surge (20 points)
        if row['obv_z'] > 1.0:
            volume_score = 20
        elif row['obv_z'] > 0.5:
            volume_score = 12
        else:
            volume_score = 0
        components['volume_surge'] = volume_score
        score += volume_score
        
        # RSI momentum (15 points)
        if 60 <= row['rsi'] <= 80:
            rsi_score = 15
        elif 55 <= row['rsi'] < 60:
            rsi_score = 10
        else:
            rsi_score = 5
        components['rsi_momentum'] = rsi_score
        score += rsi_score
        
        # TSI confirmation (15 points)
        if row['tsi'] > 0 and row['tsi'] > row['tsi_signal']:
            tsi_score = 15
        elif row['tsi'] > 0:
            tsi_score = 8
        else:
            tsi_score = 0
        components['tsi_confirm'] = tsi_score
        score += tsi_score
        
        thresholds = self.thresholds.get('#2', {})
        return ScenarioScore(
            scenario_id='#2',
            score=score,
            entry_threshold=thresholds.get('entry', 62),
            hold_threshold=thresholds.get('hold', 52),
            exit_threshold=thresholds.get('exit', 42),
            components=components,
            bonuses=bonuses,
            penalties=penalties
        )
    
    def calculate_scenario_3_score(self, row: pd.Series, htf_data: Optional[Dict] = None) -> ScenarioScore:
        """
        Scenario #3: Oversold bounce
        - RSI oversold (<30)
        - Price near lower Bollinger Band
        - Positive divergence
        """
        score = 0.0
        components = {}
        bonuses = {}
        penalties = {}
        
        # RSI oversold (30 points)
        rsi = row['rsi']
        if rsi < 25:
            rsi_score = 30
        elif rsi < 30:
            rsi_score = 25
        elif rsi < 35:
            rsi_score = 15
        else:
            rsi_score = 0
        components['rsi_oversold'] = rsi_score
        score += rsi_score
        
        # Stochastic RSI oversold (20 points)
        if row['stochrsi_k'] < 20:
            stochrsi_score = 20
        elif row['stochrsi_k'] < 30:
            stochrsi_score = 12
        else:
            stochrsi_score = 0
        components['stochrsi_oversold'] = stochrsi_score
        score += stochrsi_score
        
        # Price near lower BB (20 points)
        bb_position = (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])
        if bb_position < 0.2:
            bb_score = 20
        elif bb_position < 0.3:
            bb_score = 15
        elif bb_position < 0.4:
            bb_score = 10
        else:
            bb_score = 0
        components['bb_position'] = bb_score
        score += bb_score
        
        # Volume accumulation (15 points)
        if row['cmf'] > 0 and row['adi_z'] > 0:
            volume_score = 15
        elif row['cmf'] > 0 or row['adi_z'] > 0:
            volume_score = 8
        else:
            volume_score = 0
        components['volume_accumulation'] = volume_score
        score += volume_score
        
        # Support from EMA (15 points)
        if row['close'] > row['ema_mid']:
            ema_score = 15
        elif row['close'] > row['ema_slow']:
            ema_score = 10
        else:
            ema_score = 5
        components['ema_support'] = ema_score
        score += ema_score
        
        thresholds = self.thresholds.get('#3', {})
        return ScenarioScore(
            scenario_id='#3',
            score=score,
            entry_threshold=thresholds.get('entry', 65),
            hold_threshold=thresholds.get('hold', 55),
            exit_threshold=thresholds.get('exit', 45),
            components=components,
            bonuses=bonuses,
            penalties=penalties
        )
    
    def calculate_scenario_7_score(self, row: pd.Series, htf_data: Optional[Dict] = None) -> ScenarioScore:
        """
        Scenario #7: Mean reversion
        - Price deviated from mean (VWAP, EMA mid)
        - Low volatility (BB width)
        - Reverting indicators
        """
        score = 0.0
        components = {}
        bonuses = {}
        penalties = {}
        
        # VWAP deviation (25 points)
        vwap_dist = abs(row.get('vwap_distance_pct', 0))
        if 0.01 < vwap_dist < 0.03:
            vwap_score = 25
        elif 0.005 < vwap_dist < 0.04:
            vwap_score = 18
        elif vwap_dist > 0.04:
            vwap_score = 10
        else:
            vwap_score = 0
        components['vwap_deviation'] = vwap_score
        score += vwap_score
        
        # BB width percentile (low volatility) (25 points)
        if row['bb_width_pctile'] < 20:
            bb_width_score = 25
        elif row['bb_width_pctile'] < 30:
            bb_width_score = 18
        elif row['bb_width_pctile'] < 40:
            bb_width_score = 10
        else:
            bb_width_score = 0
        components['bb_width'] = bb_width_score
        score += bb_width_score
        
        # RSI mean reversion zone (20 points)
        rsi = row['rsi']
        rsi_dev = abs(rsi - 50)
        if 15 < rsi_dev < 25:
            rsi_score = 20
        elif 10 < rsi_dev < 30:
            rsi_score = 15
        else:
            rsi_score = 5
        components['rsi_deviation'] = rsi_score
        score += rsi_score
        
        # Price vs EMA mid (15 points)
        ema_dist = abs(row['close'] - row['ema_mid']) / row['ema_mid']
        if 0.01 < ema_dist < 0.03:
            ema_score = 15
        elif 0.005 < ema_dist < 0.04:
            ema_score = 10
        else:
            ema_score = 5
        components['ema_distance'] = ema_score
        score += ema_score
        
        # ADX (low trend strength) (15 points)
        if row['adx'] < 20:
            adx_score = 15
        elif row['adx'] < 25:
            adx_score = 10
        else:
            adx_score = 0
        components['adx_low'] = adx_score
        score += adx_score
        
        thresholds = self.thresholds.get('#7', {})
        return ScenarioScore(
            scenario_id='#7',
            score=score,
            entry_threshold=thresholds.get('entry', 75),
            hold_threshold=thresholds.get('hold', 62),
            exit_threshold=thresholds.get('exit', 52),
            components=components,
            bonuses=bonuses,
            penalties=penalties
        )
    
    def apply_bonuses_penalties(
        self,
        scenario_score: ScenarioScore,
        row: pd.Series,
        htf_data: Optional[Dict[str, pd.Series]] = None
    ) -> ScenarioScore:
        """
        Apply bonuses and penalties based on HTF alignment and other factors.
        
        Args:
            scenario_score: Base scenario score
            row: Current bar data
            htf_data: Higher timeframe data (optional)
            
        Returns:
            Updated scenario score with bonuses/penalties
        """
        if not self.bonus_penalty:
            return scenario_score
        
        total_bonus = 0.0
        total_penalty = 0.0
        
        # HTF alignment bonus/penalty
        if htf_data and self.bonus_penalty.get('htf_alignment'):
            htf_config = self.bonus_penalty['htf_alignment']
            
            # Check 10m and 15m alignment
            m10_trend = htf_data.get('10m', {}).get('trend')
            m15_trend = htf_data.get('15m', {}).get('trend')
            
            m10_aligned = m10_trend == 'up'
            m15_aligned = m15_trend == 'up'
            
            if m10_aligned and m15_aligned:
                bonus = htf_config.get('m10_m15', [4, 8])[1]
                scenario_score.bonuses['htf_m10_m15'] = bonus
                total_bonus += bonus
            
            # Check 30m and 1h alignment
            m30_trend = htf_data.get('30m', {}).get('trend')
            h1_trend = htf_data.get('1h', {}).get('trend')
            
            m30_aligned = m30_trend == 'up'
            h1_aligned = h1_trend == 'up'
            
            if m30_aligned and h1_aligned:
                bonus = htf_config.get('m30_h1', [8, 15])[1]
                scenario_score.bonuses['htf_m30_h1'] = bonus
                total_bonus += bonus
            
            # Conflict penalty
            if (m10_trend == 'down' or m15_trend == 'down' or m30_trend == 'down' or h1_trend == 'down'):
                penalty = abs(htf_config.get('conflict', [-20, -10])[0])
                scenario_score.penalties['htf_conflict'] = penalty
                total_penalty += penalty
        
        # Flow bias adjustment
        if 'flow_bias' in self.bonus_penalty:
            flow_range = self.bonus_penalty['flow_bias']
            
            # Positive flow bias
            if row['cmf'] > 0.1 and row['adi_z'] > 0.5:
                bonus = flow_range[1]
                scenario_score.bonuses['flow_positive'] = bonus
                total_bonus += bonus
            # Negative flow bias
            elif row['cmf'] < -0.1 and row['adi_z'] < -0.5:
                penalty = abs(flow_range[0])
                scenario_score.penalties['flow_negative'] = penalty
                total_penalty += penalty
        
        # Update total score
        scenario_score.score += total_bonus - total_penalty
        
        return scenario_score
    
    def evaluate_all_scenarios(self, row: pd.Series, htf_data: Optional[Dict[str, pd.Series]] = None) -> List[ScenarioScore]:
        """
        Evaluate all scenarios and return sorted list by score.
        
        Args:
            row: Current bar data with all indicators
            htf_data: Higher timeframe data (optional)
            
        Returns:
            List of ScenarioScore objects sorted by score (descending)
        """
        scenarios = []
        
        # Calculate scores for implemented scenarios
        scenarios.append(self.calculate_scenario_1_score(row, htf_data))
        scenarios.append(self.calculate_scenario_2_score(row, htf_data))
        scenarios.append(self.calculate_scenario_3_score(row, htf_data))
        scenarios.append(self.calculate_scenario_7_score(row, htf_data))
        
        # Apply bonuses and penalties
        scenarios = [
            self.apply_bonuses_penalties(s, row, htf_data)
            for s in scenarios
        ]
        
        # Sort by score (descending)
        scenarios.sort(key=lambda x: x.score, reverse=True)
        
        return scenarios
    
    def generate_signal(
        self,
        row: pd.Series,
        bar_index: int,
        position: float = 0.0,
        htf_data: Optional[Dict[str, pd.Series]] = None
    ) -> Tuple[str, Optional[ScenarioScore]]:
        """
        Generate trading signal based on scenario scores.
        
        Args:
            row: Current bar data
            bar_index: Current bar index
            position: Current position size
            htf_data: Higher timeframe data (optional)
            
        Returns:
            Tuple of (signal, best_scenario) where signal is 'entry', 'hold', 'exit', or 'none'
        """
        if not self.enabled:
            return 'none', None
        
        # Evaluate all scenarios
        scenarios = self.evaluate_all_scenarios(row, htf_data)
        best_scenario = scenarios[0] if scenarios else None
        
        if not best_scenario:
            return 'none', None
        
        # Check hysteresis cooldown
        bars_since_last_signal = bar_index - self.last_signal_bar
        if bars_since_last_signal < self.hysteresis_cooldown:
            return 'hold', best_scenario
        
        # Determine signal based on position and thresholds
        if position == 0:
            # No position - check entry
            if best_scenario.score >= best_scenario.entry_threshold:
                self.last_signal_bar = bar_index
                self.current_scenario = best_scenario.scenario_id
                return 'entry', best_scenario
        else:
            # Have position - check hold or exit
            if best_scenario.score >= best_scenario.hold_threshold:
                return 'hold', best_scenario
            elif best_scenario.score < best_scenario.exit_threshold:
                self.last_signal_bar = bar_index
                return 'exit', best_scenario
        
        return 'none', best_scenario
