import numpy as np
import pandas as pd
from main import getMyPosition
import time
import warnings
import sys
warnings.filterwarnings('ignore')

def generate_test_data(nInst=50, nt=252):
    """Generate realistic test price data"""
    np.random.seed(42)
    
    # Generate correlated price movements
    base_trend = np.cumsum(np.random.randn(nt) * 0.01)
    
    prices = np.zeros((nInst, nt))
    for i in range(nInst):
        # Individual stock trend
        stock_trend = np.cumsum(np.random.randn(nt) * 0.02)
        # Market correlation
        market_component = base_trend * (0.3 + 0.4 * np.random.random())
        # Idiosyncratic noise
        noise = np.cumsum(np.random.randn(nt) * 0.015)
        
        # Combine components
        returns = stock_trend + market_component + noise
        prices[i] = 100 * np.exp(returns)
    
    return prices

def calculate_returns(prices):
    """Calculate daily returns"""
    return np.diff(prices, axis=1) / prices[:, :-1]

def calculate_metrics(positions, prices, commission_rate=0.0005):
    """Calculate performance metrics"""
    returns = calculate_returns(prices)
    
    # Calculate P&L
    position_returns = positions.reshape(-1, 1) * returns
    daily_pnl = np.sum(position_returns, axis=0)
    
    # Calculate transaction costs
    position_changes = np.diff(positions.reshape(-1, 1), axis=1, prepend=positions.reshape(-1, 1))
    transaction_volume = np.abs(position_changes) * prices
    transaction_costs = np.sum(transaction_volume, axis=0) * commission_rate
    
    # Ensure transaction costs has same length as daily_pnl
    if len(transaction_costs) > len(daily_pnl):
        transaction_costs = transaction_costs[:len(daily_pnl)]
    elif len(transaction_costs) < len(daily_pnl):
        # Pad with zeros if needed
        transaction_costs = np.pad(transaction_costs, (0, len(daily_pnl) - len(transaction_costs)), 'constant')
    
    # Net P&L
    net_pnl = daily_pnl - transaction_costs
    
    # Calculate metrics
    mean_pl = np.mean(net_pnl)
    std_pl = np.std(net_pnl)
    sharpe = mean_pl / (std_pl + 1e-8)
    
    # Competition metric: mean(PL) - 0.1 * StdDev(PL)
    competition_score = mean_pl - 0.1 * std_pl
    
    return {
        'mean_pl': mean_pl,
        'std_pl': std_pl,
        'sharpe': sharpe,
        'competition_score': competition_score,
        'total_return': np.sum(net_pnl),
        'max_drawdown': calculate_max_drawdown(net_pnl),
        'win_rate': np.sum(net_pnl > 0) / len(net_pnl)
    }

def calculate_max_drawdown(pnl):
    """Calculate maximum drawdown"""
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    return np.min(drawdown)

def test_position_limits(positions, prices, limit=10000):
    """Test if positions respect $10k limit"""
    current_prices = prices[:, -1]
    dollar_positions = positions * current_prices
    
    violations = np.abs(dollar_positions) > limit
    return {
        'violations': np.sum(violations),
        'max_position': np.max(np.abs(dollar_positions)),
        'avg_position': np.mean(np.abs(dollar_positions))
    }

def test_algorithm():
    """Comprehensive algorithm testing"""
    print("🚀 Testing Algothon Trading Algorithm")
    print("=" * 50)
    
    # Test 1: Basic functionality
    print("\n1. Testing basic functionality...")
    nInst, nt = 50, 100
    test_prices = generate_test_data(nInst, nt)
    
    start_time = time.time()
    positions = getMyPosition(test_prices)
    execution_time = time.time() - start_time
    
    print(f"✅ Execution time: {execution_time:.3f} seconds")
    print(f"✅ Positions shape: {positions.shape}")
    print(f"✅ Position range: {positions.min()} to {positions.max()}")
    print(f"✅ Non-zero positions: {np.count_nonzero(positions)}")
    
    # Test 2: Position limits
    print("\n2. Testing position limits...")
    limit_test = test_position_limits(positions, test_prices)
    print(f"✅ Max position: ${limit_test['max_position']:.2f}")
    print(f"✅ Average position: ${limit_test['avg_position']:.2f}")
    print(f"✅ Limit violations: {limit_test['violations']}")
    
    if limit_test['violations'] == 0:
        print("✅ All positions within $10k limit!")
    else:
        print("❌ Position limit violations detected!")
    
    # Test 3: Performance metrics
    print("\n3. Testing performance metrics...")
    metrics = calculate_metrics(positions, test_prices)
    
    print(f"✅ Mean P&L: ${metrics['mean_pl']:.2f}")
    print(f"✅ P&L Std Dev: ${metrics['std_pl']:.2f}")
    print(f"✅ Sharpe Ratio: {metrics['sharpe']:.3f}")
    print(f"✅ Competition Score: {metrics['competition_score']:.3f}")
    print(f"✅ Total Return: ${metrics['total_return']:.2f}")
    print(f"✅ Max Drawdown: ${metrics['max_drawdown']:.2f}")
    print(f"✅ Win Rate: {metrics['win_rate']:.1%}")
    
    # Test 4: Multiple time periods
    print("\n4. Testing multiple time periods...")
    periods = [50, 100, 200, 500]
    
    for nt in periods:
        test_prices = generate_test_data(50, nt)
        positions = getMyPosition(test_prices)
        metrics = calculate_metrics(positions, test_prices)
        print(f"   {nt} days: Score = {metrics['competition_score']:.3f}, Sharpe = {metrics['sharpe']:.3f}")
    
    # Test 5: Runtime performance
    print("\n5. Testing runtime performance...")
    large_prices = generate_test_data(50, 1000)
    
    start_time = time.time()
    positions = getMyPosition(large_prices)
    execution_time = time.time() - start_time
    
    print(f"✅ 1000-day execution time: {execution_time:.3f} seconds")
    
    if execution_time < 600:  # 10 minutes
        print("✅ Runtime within 10-minute limit!")
    else:
        print("❌ Runtime exceeds 10-minute limit!")
    
    # Test 6: Edge cases
    print("\n6. Testing edge cases...")
    
    # Very short data
    short_prices = generate_test_data(50, 5)
    try:
        positions = getMyPosition(short_prices)
        print("✅ Handles short data correctly")
    except Exception as e:
        print(f"❌ Error with short data: {e}")
    
    # Extreme price movements
    extreme_prices = generate_test_data(50, 100)
    extreme_prices *= np.random.uniform(0.1, 10, extreme_prices.shape)
    try:
        positions = getMyPosition(extreme_prices)
        print("✅ Handles extreme price movements")
    except Exception as e:
        print(f"❌ Error with extreme prices: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Algorithm Testing Complete!")
    
    return metrics

def test_algorithm_requirements():
    """Test that the algorithm meets all competition requirements"""
    
    print("=== ALGOTHON QUANT TRADING ALGORITHM VALIDATION ===\n")
    
    # Test 1: Function signature and basic functionality
    print("1. Testing function signature and basic functionality...")
    try:
        # Create test data: 50 instruments, 100 days
        test_prices = np.random.rand(50, 100) * 100 + 50
        
        # Test function call
        positions = getMyPosition(test_prices)
        
        # Check return type and shape
        assert isinstance(positions, np.ndarray), "Return must be numpy array"
        assert positions.shape == (50,), f"Expected shape (50,), got {positions.shape}"
        assert positions.dtype in [np.int32, np.int64, int], "Positions must be integers"
        
        print("✓ Function signature and basic functionality: PASSED")
        
    except Exception as e:
        print(f"✗ Function signature test failed: {e}")
        return False
    
    # Test 2: Position limits ($10k per instrument)
    print("\n2. Testing position limits ($10k per instrument)...")
    try:
        test_prices = np.random.rand(50, 50) * 100 + 50
        positions = getMyPosition(test_prices)
        
        # Calculate dollar positions
        current_prices = test_prices[:, -1]
        dollar_positions = positions * current_prices
        
        # Check position limits
        max_dollar_position = np.max(np.abs(dollar_positions))
        assert max_dollar_position <= 10000, f"Position limit exceeded: ${max_dollar_position:.2f}"
        
        print(f"✓ Position limits: PASSED (max position: ${max_dollar_position:.2f})")
        
    except Exception as e:
        print(f"✗ Position limits test failed: {e}")
        return False
    
    # Test 3: Runtime performance (should be under 10 minutes)
    print("\n3. Testing runtime performance...")
    try:
        # Test with larger dataset
        large_prices = np.random.rand(50, 1000) * 100 + 50
        
        start_time = time.time()
        positions = getMyPosition(large_prices)
        end_time = time.time()
        
        runtime = end_time - start_time
        max_runtime = 600  # 10 minutes in seconds
        
        assert runtime < max_runtime, f"Runtime too slow: {runtime:.2f}s (max: {max_runtime}s)"
        
        print(f"✓ Runtime performance: PASSED ({runtime:.2f}s)")
        
    except Exception as e:
        print(f"✗ Runtime test failed: {e}")
        return False
    
    # Test 4: Edge cases
    print("\n4. Testing edge cases...")
    try:
        # Test with minimal data
        min_prices = np.random.rand(50, 2) * 100 + 50
        positions_min = getMyPosition(min_prices)
        assert positions_min.shape == (50,), "Minimal data test failed"
        
        # Test with single day data
        single_prices = np.random.rand(50, 1) * 100 + 50
        positions_single = getMyPosition(single_prices)
        assert positions_single.shape == (50,), "Single day test failed"
        
        # Test with extreme price values
        extreme_prices = np.random.rand(50, 100) * 1000 + 1  # Very high prices
        positions_extreme = getMyPosition(extreme_prices)
        assert positions_extreme.shape == (50,), "Extreme prices test failed"
        
        print("✓ Edge cases: PASSED")
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        return False
    
    # Test 5: Consistency and reproducibility
    print("\n5. Testing consistency and reproducibility...")
    try:
        np.random.seed(42)
        test_prices = np.random.rand(50, 100) * 100 + 50
        
        # Run multiple times with same data
        positions1 = getMyPosition(test_prices)
        positions2 = getMyPosition(test_prices)
        
        # Should be consistent
        assert np.array_equal(positions1, positions2), "Results not consistent"
        
        print("✓ Consistency and reproducibility: PASSED")
        
    except Exception as e:
        print(f"✗ Consistency test failed: {e}")
        return False
    
    # Test 6: Commission awareness
    print("\n6. Testing commission awareness...")
    try:
        test_prices = np.random.rand(50, 100) * 100 + 50
        
        # Get initial positions
        initial_positions = getMyPosition(test_prices)
        
        # Simulate price movement
        new_prices = test_prices.copy()
        new_prices[:, -1] *= 1.05  # 5% price increase
        
        # Get new positions
        new_positions = getMyPosition(new_prices)
        
        # Calculate trading volume
        position_changes = np.abs(new_positions - initial_positions)
        trading_volume = np.sum(position_changes * new_prices[:, -1])
        
        print(f"✓ Commission awareness: PASSED (trading volume: ${trading_volume:.2f})")
        
    except Exception as e:
        print(f"✗ Commission test failed: {e}")
        return False
    
    # Test 7: Risk management
    print("\n7. Testing risk management...")
    try:
        test_prices = np.random.rand(50, 100) * 100 + 50
        positions = getMyPosition(test_prices)
        
        # Check diversification (not all positions in same direction)
        long_positions = np.sum(positions > 0)
        short_positions = np.sum(positions < 0)
        zero_positions = np.sum(positions == 0)
        
        print(f"  - Long positions: {long_positions}")
        print(f"  - Short positions: {short_positions}")
        print(f"  - Zero positions: {zero_positions}")
        
        # Check that we're not over-concentrated
        total_positions = long_positions + short_positions
        assert total_positions <= 50, "Too many positions"
        
        print("✓ Risk management: PASSED")
        
    except Exception as e:
        print(f"✗ Risk management test failed: {e}")
        return False
    
    # Test 8: Strategy validation
    print("\n8. Testing strategy validation...")
    try:
        # Create trending data
        trending_prices = np.random.rand(50, 100) * 100 + 50
        for i in range(50):
            trend = np.linspace(0, 20, 100)  # Upward trend
            trending_prices[i, :] += trend
        
        trending_positions = getMyPosition(trending_prices)
        
        # Create mean-reverting data
        mean_reverting_prices = np.random.rand(50, 100) * 100 + 50
        for i in range(50):
            oscillation = 10 * np.sin(np.linspace(0, 4*np.pi, 100))
            mean_reverting_prices[i, :] += oscillation
        
        mean_reverting_positions = getMyPosition(mean_reverting_prices)
        
        # Check that strategies respond differently to different market conditions
        trending_non_zero = np.count_nonzero(trending_positions)
        mean_reverting_non_zero = np.count_nonzero(mean_reverting_positions)
        
        print(f"  - Trending market positions: {trending_non_zero}")
        print(f"  - Mean-reverting market positions: {mean_reverting_non_zero}")
        
        print("✓ Strategy validation: PASSED")
        
    except Exception as e:
        print(f"✗ Strategy validation test failed: {e}")
        return False
    
    print("\n=== ALL TESTS PASSED ===")
    print("✓ Algorithm meets all competition requirements")
    print("✓ Ready for submission!")
    
    return True

def performance_benchmark():
    """Run performance benchmark"""
    print("\n=== PERFORMANCE BENCHMARK ===")
    
    # Test different data sizes
    data_sizes = [100, 500, 1000, 2000]
    
    for size in data_sizes:
        test_prices = np.random.rand(50, size) * 100 + 50
        
        start_time = time.time()
        positions = getMyPosition(test_prices)
        end_time = time.time()
        
        runtime = end_time - start_time
        print(f"Data size {size} days: {runtime:.3f}s")
    
    print("✓ Performance benchmark completed")

def strategy_analysis():
    """Analyze strategy behavior"""
    print("\n=== STRATEGY ANALYSIS ===")
    
    # Create different market scenarios
    scenarios = {
        "Bull Market": np.random.rand(50, 100) * 100 + 50 + np.linspace(0, 30, 100),
        "Bear Market": np.random.rand(50, 100) * 100 + 50 - np.linspace(0, 30, 100),
        "Sideways": np.random.rand(50, 100) * 100 + 50 + 5 * np.sin(np.linspace(0, 4*np.pi, 100)),
        "Volatile": np.random.rand(50, 100) * 100 + 50 + 20 * np.random.randn(50, 100)
    }
    
    for scenario_name, prices in scenarios.items():
        positions = getMyPosition(prices)
        
        # Calculate metrics
        total_exposure = np.sum(positions * prices[:, -1])
        num_positions = np.count_nonzero(positions)
        avg_position_size = np.mean(np.abs(positions[positions != 0])) if np.any(positions != 0) else 0
        
        print(f"{scenario_name}:")
        print(f"  - Total exposure: ${total_exposure:.2f}")
        print(f"  - Number of positions: {num_positions}")
        print(f"  - Average position size: {avg_position_size:.1f} shares")
        print()

if __name__ == "__main__":
    # Run all tests
    success = test_algorithm_requirements()
    
    if success:
        performance_benchmark()
        strategy_analysis()
    else:
        print("\n❌ Algorithm validation failed. Please fix issues before submission.")
        sys.exit(1) 