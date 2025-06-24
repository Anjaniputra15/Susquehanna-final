import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class AdvancedTradingAlgorithm:
    """
    Advanced Trading Algorithm for the competition
    Features:
    - Multi-strategy ensemble (momentum, mean reversion, volatility)
    - Risk management with position limits
    - Dynamic parameter optimization
    - Commission-aware trading
    - Robust signal generation
    """
    
    def __init__(self):
        self.position_limit = 10000  # $10k limit per stock
        self.commission_rate = 0.0005  # 5 bps
        self.lookback_periods = [5, 10, 20, 50]  # Multiple lookback periods
        self.volatility_window = 20
        self.momentum_threshold = 0.02
        self.mean_reversion_threshold = 0.03
        self.risk_aversion = 0.1
        self.max_positions = 15  # Maximum number of positions to hold
        
        # Initialize models with fixed random state
        self.scaler = StandardScaler()
        self.rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        
    def calculate_returns(self, prices):
        """Calculate daily returns"""
        return np.diff(np.log(prices), axis=1)
    
    def calculate_volatility(self, returns, window=20):
        """Calculate rolling volatility"""
        if returns.shape[1] < window:
            return np.std(returns, axis=1, keepdims=True)
        
        volatility = np.zeros_like(returns)
        for i in range(window, returns.shape[1]):
            volatility[:, i] = np.std(returns[:, i-window:i], axis=1)
        
        # Fill initial values
        for i in range(window):
            volatility[:, i] = np.std(returns[:, :i+1], axis=1)
            
        return volatility
    
    def calculate_momentum_signals(self, prices, lookback_periods):
        """Calculate momentum signals for multiple lookback periods"""
        signals = np.zeros((prices.shape[0], len(lookback_periods)))
        
        for i, period in enumerate(lookback_periods):
            if prices.shape[1] > period:
                # Price momentum
                price_momentum = (prices[:, -1] - prices[:, -period-1]) / prices[:, -period-1]
                
                # Use simple smoothing instead of Savitzky-Golay for consistency
                # Simple moving average smoothing
                if len(price_momentum) >= 5:
                    window_size = min(5, len(price_momentum))
                    smoothed_momentum = np.convolve(price_momentum, np.ones(window_size)/window_size, mode='same')
                else:
                    smoothed_momentum = price_momentum
                    
                signals[:, i] = smoothed_momentum
            else:
                signals[:, i] = 0
                
        return signals
    
    def calculate_mean_reversion_signals(self, prices, lookback_periods):
        """Calculate mean reversion signals"""
        signals = np.zeros((prices.shape[0], len(lookback_periods)))
        
        for i, period in enumerate(lookback_periods):
            if prices.shape[1] > period:
                # Calculate moving average
                ma = np.mean(prices[:, -period:], axis=1)
                current_price = prices[:, -1]
                
                # Mean reversion signal: negative when price > MA, positive when price < MA
                mean_reversion = (ma - current_price) / current_price
                
                # Add volatility adjustment
                returns = self.calculate_returns(prices[:, -period:])
                volatility = np.std(returns, axis=1)
                adjusted_signal = mean_reversion / (volatility + 1e-8)
                
                signals[:, i] = adjusted_signal
            else:
                signals[:, i] = 0
                
        return signals
    
    def calculate_volatility_signals(self, prices):
        """Calculate volatility-based signals"""
        if prices.shape[1] < self.volatility_window:
            return np.zeros(prices.shape[0])
        
        returns = self.calculate_returns(prices)
        volatility = self.calculate_volatility(returns, self.volatility_window)
        
        # Volatility breakout signal
        current_vol = volatility[:, -1]
        avg_vol = np.mean(volatility[:, -self.volatility_window:], axis=1)
        
        # Signal is positive when volatility is low (mean reversion opportunity)
        # and negative when volatility is high (momentum opportunity)
        vol_signal = (avg_vol - current_vol) / (avg_vol + 1e-8)
        
        return vol_signal
    
    def calculate_technical_indicators(self, prices):
        """Calculate various technical indicators"""
        if prices.shape[1] < 20:
            return np.zeros((prices.shape[0], 4))
        
        indicators = np.zeros((prices.shape[0], 4))
        
        # RSI-like indicator
        returns = self.calculate_returns(prices)
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        
        avg_gains = np.mean(gains[:, -14:], axis=1)
        avg_losses = np.mean(losses[:, -14:], axis=1)
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        indicators[:, 0] = (rsi - 50) / 50  # Normalize to [-1, 1]
        
        # Bollinger Bands
        ma_20 = np.mean(prices[:, -20:], axis=1)
        std_20 = np.std(prices[:, -20:], axis=1)
        bb_position = (prices[:, -1] - ma_20) / (2 * std_20 + 1e-8)
        indicators[:, 1] = bb_position
        
        # MACD-like
        ma_12 = np.mean(prices[:, -12:], axis=1)
        ma_26 = np.mean(prices[:, -26:], axis=1)
        macd = (ma_12 - ma_26) / ma_26
        indicators[:, 2] = macd
        
        # Price acceleration
        if prices.shape[1] >= 5:
            accel = (prices[:, -1] - 2 * prices[:, -3] + prices[:, -5]) / prices[:, -3]
            indicators[:, 3] = accel
        
        return indicators
    
    def calculate_commission_awareness(self, prices):
        """Calculate commission-aware signals based on price volatility"""
        if prices.shape[1] < 5:
            return np.zeros(prices.shape[0])
        
        # Calculate recent price volatility
        recent_returns = self.calculate_returns(prices[:, -5:])
        recent_volatility = np.std(recent_returns, axis=1)
        
        # Higher volatility = higher expected trading costs
        # Signal should be more conservative in high volatility
        commission_signal = -recent_volatility * self.commission_rate * 100
        
        return commission_signal
    
    def ensemble_signal_generation(self, prices):
        """Generate ensemble signals from multiple strategies"""
        if prices.shape[1] < 5:
            return np.zeros(prices.shape[0])
        
        # Get all signals
        momentum_signals = self.calculate_momentum_signals(prices, self.lookback_periods)
        mean_reversion_signals = self.calculate_mean_reversion_signals(prices, self.lookback_periods)
        volatility_signals = self.calculate_volatility_signals(prices)
        technical_indicators = self.calculate_technical_indicators(prices)
        commission_signals = self.calculate_commission_awareness(prices)
        
        # Combine signals with weights
        weights = {
            'momentum': 0.25,
            'mean_reversion': 0.25,
            'volatility': 0.15,
            'technical': 0.20,
            'commission': 0.15
        }
        
        # Average momentum signals across lookback periods
        avg_momentum = np.mean(momentum_signals, axis=1)
        avg_mean_reversion = np.mean(mean_reversion_signals, axis=1)
        avg_technical = np.mean(technical_indicators, axis=1)
        
        # Combine all signals
        combined_signal = (
            weights['momentum'] * avg_momentum +
            weights['mean_reversion'] * avg_mean_reversion +
            weights['volatility'] * volatility_signals +
            weights['technical'] * avg_technical +
            weights['commission'] * commission_signals
        )
        
        return combined_signal
    
    def apply_position_limits(self, positions, prices):
        """Apply $10k position limits"""
        current_prices = prices[:, -1]
        dollar_positions = positions * current_prices
        
        # Clip positions to $10k limit
        max_shares = self.position_limit / current_prices
        clipped_positions = np.clip(positions, -max_shares, max_shares)
        
        return clipped_positions.astype(int)
    
    def optimize_portfolio(self, signals, prices):
        """Optimize portfolio allocation considering risk and constraints"""
        if len(signals) == 0:
            return np.zeros(prices.shape[0])
        
        # Sort instruments by signal strength (use absolute values for deterministic sorting)
        sorted_indices = np.argsort(np.abs(signals))[::-1]
        
        # Select top instruments
        top_instruments = sorted_indices[:self.max_positions]
        
        # Initialize positions
        positions = np.zeros(prices.shape[0])
        
        # Allocate capital to top instruments
        current_prices = prices[:, -1]
        available_capital = self.position_limit * self.max_positions
        
        for i, inst_idx in enumerate(top_instruments):
            signal_strength = signals[inst_idx]
            price = current_prices[inst_idx]
            
            if abs(signal_strength) > 0.01:  # Minimum signal threshold
                # Calculate position size based on signal strength and risk
                base_position = signal_strength * self.position_limit / price
                
                # Apply risk adjustment
                risk_adjusted_position = base_position * (1 - self.risk_aversion * abs(signal_strength))
                
                # Ensure position is within limits
                max_shares = self.position_limit / price
                final_position = np.clip(risk_adjusted_position, -max_shares, max_shares)
                
                positions[inst_idx] = int(final_position)
        
        return positions
    
    def getMyPosition(self, prices):
        """
        Main function required by the competition
        
        Args:
            prices: numpy array of shape (nInst, nt) where nInst=50 and nt is number of days
            
        Returns:
            numpy array of shape (nInst,) with integer positions for each instrument
        """
        try:
            # Ensure input is numpy array
            prices = np.array(prices)
            
            # Handle edge cases
            if prices.shape[0] != 50:
                raise ValueError(f"Expected 50 instruments, got {prices.shape[0]}")
            
            if prices.shape[1] < 2:
                # Not enough data, return zero positions
                return np.zeros(50, dtype=int)
            
            # Generate ensemble signals
            signals = self.ensemble_signal_generation(prices)
            
            # Optimize portfolio allocation
            positions = self.optimize_portfolio(signals, prices)
            
            # Apply position limits
            final_positions = self.apply_position_limits(positions, prices)
            
            return final_positions
            
        except Exception as e:
            # Fallback: return zero positions if any error occurs
            print(f"Error in getMyPosition: {e}")
            return np.zeros(50, dtype=int)

# Global instance of the trading algorithm
trading_algorithm = AdvancedTradingAlgorithm()

def getMyPosition(prices):
    """
    Competition entry point function
    
    Args:
        prices: numpy array of shape (50, nt) with price data
        
    Returns:
        numpy array of shape (50,) with integer positions
    """
    return trading_algorithm.getMyPosition(prices)

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    test_prices = np.random.rand(50, 100) * 100 + 50  # 50 instruments, 100 days
    
    positions = getMyPosition(test_prices)
    print(f"Generated positions: {positions}")
    print(f"Position range: {positions.min()} to {positions.max()}")
    print(f"Non-zero positions: {np.count_nonzero(positions)}")
