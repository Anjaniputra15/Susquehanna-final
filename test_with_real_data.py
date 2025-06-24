import numpy as np
import time
from main import getMyPosition

def load_prices_data(filename='prices.txt'):
    """Load the actual competition price data"""
    print(f"Loading price data from {filename}...")
    
    # Load the data
    prices = np.loadtxt(filename)
    
    # The data is in format: days x instruments
    # We need to transpose it to: instruments x days
    prices = prices.T
    
    print(f"Loaded price data: {prices.shape[0]} instruments, {prices.shape[1]} days")
    print(f"Price range: ${prices.min():.2f} to ${prices.max():.2f}")
    print(f"Average price: ${prices.mean():.2f}")
    
    return prices

def test_algorithm_with_real_data():
    """Test the algorithm with actual competition data"""
    print("=== TESTING ALGORITHM WITH REAL COMPETITION DATA ===\n")
    
    # Load the actual price data
    prices = load_prices_data()
    
    # Test different time periods
    test_periods = [
        (0, 100, "First 100 days"),
        (100, 200, "Days 100-200"),
        (300, 400, "Days 300-400"),
        (500, 600, "Days 500-600"),
        (650, 750, "Last 100 days"),
        (0, 750, "Full dataset")
    ]
    
    results = []
    
    for start_day, end_day, description in test_periods:
        print(f"\n--- Testing {description} ---")
        
        # Extract the price data for this period
        period_prices = prices[:, start_day:end_day]
        
        # Time the algorithm
        start_time = time.time()
        positions = getMyPosition(period_prices)
        end_time = time.time()
        
        runtime = end_time - start_time
        
        # Calculate metrics
        current_prices = period_prices[:, -1]
        dollar_positions = positions * current_prices
        
        # Statistics
        total_exposure = np.sum(dollar_positions)
        num_positions = np.count_nonzero(positions)
        long_positions = np.sum(positions > 0)
        short_positions = np.sum(positions < 0)
        max_position = np.max(np.abs(dollar_positions))
        avg_position_size = np.mean(np.abs(positions[positions != 0])) if np.any(positions != 0) else 0
        
        print(f"Runtime: {runtime:.3f}s")
        print(f"Total exposure: ${total_exposure:,.2f}")
        print(f"Number of positions: {num_positions}")
        print(f"Long positions: {long_positions}")
        print(f"Short positions: {short_positions}")
        print(f"Max position: ${max_position:.2f}")
        print(f"Average position size: {avg_position_size:.1f} shares")
        
        # Check position limits
        if max_position > 10000:
            print(f"⚠️  WARNING: Position limit exceeded! Max: ${max_position:.2f}")
        else:
            print(f"✅ Position limits respected (max: ${max_position:.2f})")
        
        results.append({
            'period': description,
            'runtime': runtime,
            'total_exposure': total_exposure,
            'num_positions': num_positions,
            'max_position': max_position
        })
    
    # Summary
    print("\n=== SUMMARY ===")
    print("Period | Runtime | Exposure | Positions | Max Position")
    print("-" * 60)
    for result in results:
        print(f"{result['period']:15} | {result['runtime']:6.3f}s | ${result['total_exposure']:8,.0f} | {result['num_positions']:9} | ${result['max_position']:8.0f}")
    
    # Performance analysis
    print(f"\n=== PERFORMANCE ANALYSIS ===")
    avg_runtime = np.mean([r['runtime'] for r in results])
    print(f"Average runtime: {avg_runtime:.3f}s")
    
    if avg_runtime < 600:  # 10 minutes
        print("✅ Runtime performance: PASSED (< 10 minutes)")
    else:
        print("❌ Runtime performance: FAILED (> 10 minutes)")
    
    # Position limit compliance
    all_max_positions = [r['max_position'] for r in results]
    if all(mp <= 10000 for mp in all_max_positions):
        print("✅ Position limits: PASSED (all positions ≤ $10k)")
    else:
        print("❌ Position limits: FAILED (some positions > $10k)")
    
    print("\n=== ALGORITHM READY FOR COMPETITION! ===")

def analyze_market_conditions():
    """Analyze the market conditions in the data"""
    print("\n=== MARKET CONDITION ANALYSIS ===")
    
    prices = load_prices_data()
    
    # Calculate returns
    returns = np.diff(np.log(prices), axis=1)
    
    # Volatility analysis
    volatility = np.std(returns, axis=1)
    print(f"Average volatility: {np.mean(volatility):.4f}")
    print(f"Volatility range: {np.min(volatility):.4f} to {np.max(volatility):.4f}")
    
    # Trend analysis
    price_changes = (prices[:, -1] - prices[:, 0]) / prices[:, 0]
    print(f"Average price change: {np.mean(price_changes):.2%}")
    print(f"Price change range: {np.min(price_changes):.2%} to {np.max(price_changes):.2%}")
    
    # Correlation analysis
    correlation_matrix = np.corrcoef(returns)
    avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
    print(f"Average correlation: {avg_correlation:.3f}")
    
    # Market regime detection
    if np.mean(price_changes) > 0.1:
        regime = "Bull Market"
    elif np.mean(price_changes) < -0.1:
        regime = "Bear Market"
    else:
        regime = "Sideways Market"
    
    print(f"Detected market regime: {regime}")

if __name__ == "__main__":
    # Test with real data
    test_algorithm_with_real_data()
    
    # Analyze market conditions
    analyze_market_conditions() 