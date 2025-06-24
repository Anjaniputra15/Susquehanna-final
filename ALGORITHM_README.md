# Advanced Trading Algorithm for Algothon Quant Competition

## Overview

This is a sophisticated quantitative trading algorithm designed for the Algothon Quant competition. The algorithm implements a multi-strategy ensemble approach that combines momentum, mean reversion, volatility, and technical indicators to generate optimal trading positions.

## Key Features

### 🎯 **Competition Compliance**
- ✅ Function signature: `getMyPosition(prices)` with correct input/output format
- ✅ Position limits: $10,000 maximum per instrument
- ✅ Commission awareness: 5 bps (0.0005) transaction costs
- ✅ Runtime optimization: Under 10 minutes execution time
- ✅ Integer positions: All positions returned as integers
- ✅ Robust error handling: Graceful fallback to zero positions

### 🧠 **Advanced Strategy Components**

#### 1. **Multi-Period Momentum Strategy**
- Calculates momentum signals across multiple lookback periods (5, 10, 20, 50 days)
- Uses Savitzky-Golay smoothing for signal refinement
- Adapts to different market timeframes

#### 2. **Mean Reversion Strategy**
- Identifies overbought/oversold conditions using moving averages
- Volatility-adjusted mean reversion signals
- Multiple lookback periods for robust signal generation

#### 3. **Volatility-Based Strategy**
- Detects volatility regimes and breakout opportunities
- Low volatility = mean reversion opportunities
- High volatility = momentum opportunities

#### 4. **Technical Indicators**
- **RSI-like indicator**: Relative strength index for overbought/oversold detection
- **Bollinger Bands**: Price position relative to volatility bands
- **MACD-like**: Moving average convergence/divergence
- **Price acceleration**: Second derivative of price movement

#### 5. **Risk Management**
- Position size optimization based on signal strength
- Maximum 15 active positions for diversification
- Commission-aware trading to minimize transaction costs
- Dynamic risk adjustment based on market conditions

### 📊 **Signal Generation Process**

1. **Data Preprocessing**
   - Calculate log returns and volatility
   - Handle edge cases and missing data
   - Normalize signals for consistency

2. **Multi-Strategy Ensemble**
   - Combine signals with weighted averaging:
     - Momentum: 25%
     - Mean Reversion: 25%
     - Volatility: 15%
     - Technical: 20%
     - Risk: 15%

3. **Portfolio Optimization**
   - Select top 15 instruments by signal strength
   - Apply risk-adjusted position sizing
   - Ensure position limits compliance

4. **Position Finalization**
   - Apply $10k position limits
   - Convert to integer positions
   - Update algorithm state

## Algorithm Architecture

```
Input: Price Data (50 instruments × n days)
    ↓
1. Calculate Returns & Volatility
    ↓
2. Generate Multi-Strategy Signals
    ├── Momentum (5, 10, 20, 50 day)
    ├── Mean Reversion (5, 10, 20, 50 day)
    ├── Volatility Regime
    ├── Technical Indicators (RSI, BB, MACD, Acceleration)
    └── Risk-Adjusted Signals
    ↓
3. Ensemble Signal Combination
    ↓
4. Portfolio Optimization
    ├── Top 15 Instrument Selection
    ├── Risk-Adjusted Position Sizing
    └── Position Limit Application
    ↓
5. Position Finalization
    ├── $10k Limit Enforcement
    ├── Integer Conversion
    └── State Update
    ↓
Output: Integer Positions (50 instruments)
```

## Risk Management Features

### **Position Limits**
- Maximum $10,000 per instrument (both long and short)
- Automatic position clipping to prevent limit breaches
- Dynamic adjustment based on current prices

### **Diversification**
- Maximum 15 active positions
- Balanced long/short exposure
- Correlation-aware position sizing

### **Commission Optimization**
- Minimizes unnecessary trading
- Considers 5 bps transaction costs
- Reduces position turnover

### **Volatility Adjustment**
- Reduces position sizes in high volatility
- Increases position sizes in low volatility
- Adaptive to market conditions

## Performance Characteristics

### **Expected Behavior**
- **Trending Markets**: Momentum strategies dominate
- **Sideways Markets**: Mean reversion strategies excel
- **Volatile Markets**: Reduced position sizes, increased diversification
- **Low Volatility**: Increased position sizes, trend following

### **Risk Metrics**
- Maximum drawdown: Controlled through position limits
- Sharpe ratio: Optimized through signal combination
- Correlation: Minimized through diversification

## Testing and Validation

The algorithm includes comprehensive testing:

```bash
python test_algorithm.py
```

### **Test Coverage**
1. ✅ Function signature validation
2. ✅ Position limit compliance ($10k max)
3. ✅ Runtime performance (< 10 minutes)
4. ✅ Edge case handling
5. ✅ Consistency and reproducibility
6. ✅ Commission awareness
7. ✅ Risk management validation
8. ✅ Strategy behavior analysis

## Dependencies

```txt
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

## Usage

```python
from main import getMyPosition
import numpy as np

# Example usage
prices = np.random.rand(50, 100) * 100 + 50  # 50 instruments, 100 days
positions = getMyPosition(prices)
print(f"Positions: {positions}")
```

## Competition Submission

### **Files Required**
- `main.py` - Contains the `getMyPosition()` function
- `requirements.txt` - Dependencies (if using non-standard packages)

### **Function Signature**
```python
def getMyPosition(prices):
    """
    Args:
        prices: numpy array of shape (50, nt) with price data
        
    Returns:
        numpy array of shape (50,) with integer positions
    """
```

### **Key Requirements Met**
- ✅ Returns integer positions for 50 instruments
- ✅ Respects $10k position limits
- ✅ Handles commission costs (5 bps)
- ✅ Runtime under 10 minutes
- ✅ Robust error handling
- ✅ Uses only standard Anaconda packages

## Strategy Advantages

1. **Adaptive**: Responds to different market conditions
2. **Robust**: Multiple signal sources reduce overfitting
3. **Risk-Aware**: Comprehensive risk management
4. **Efficient**: Optimized for speed and memory usage
5. **Scalable**: Handles varying data sizes gracefully

## Expected Performance

Based on the strategy design, the algorithm should:
- Generate positive risk-adjusted returns
- Maintain low correlation with market indices
- Exhibit controlled drawdowns
- Adapt to changing market regimes
- Minimize transaction costs


