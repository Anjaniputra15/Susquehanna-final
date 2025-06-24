#!/usr/bin/env python3
"""
Test the trading algorithm with real prices.txt data
"""

import numpy as np
import pandas as pd
from main import getMyPosition, backtest_equity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def load_prices_from_file(filename='prices.txt'):
    """Load prices from the prices.txt file"""
    try:
        # Read the file
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse each line into price values
        prices_data = []
        for line in lines:
            # Split by whitespace and convert to float
            price_values = [float(x) for x in line.strip().split()]
            prices_data.append(price_values)
        
        # Convert to numpy array
        prices_array = np.array(prices_data)
        
        print(f"Loaded {prices_array.shape[0]} days of data for {prices_array.shape[1]} assets")
        print(f"Price range: ${prices_array.min():.2f} - ${prices_array.max():.2f}")
        
        return prices_array
        
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        return None
    except Exception as e:
        print(f"Error loading prices: {e}")
        return None

def run_algorithm_test():
    """Run the algorithm test with real price data"""
    print("=== Testing Trading Algorithm with Real Price Data ===\n")
    
    # Load prices
    prices = load_prices_from_file('prices.txt')
    if prices is None:
        return
    
    # Transpose prices to match algorithm expectation (assets x days)
    prices = prices.T
    print(f"Transposed to {prices.shape[0]} assets x {prices.shape[1]} days\n")
    
    # Test the algorithm
    print("Running algorithm...")
    positions = getMyPosition(prices)
    
    print(f"\nGenerated positions for {len(positions)} assets:")
    print(f"Total positions: {sum(abs(pos) for pos in positions)}")
    print(f"Long positions: {sum(1 for pos in positions if pos > 0)}")
    print(f"Short positions: {sum(1 for pos in positions if pos < 0)}")
    print(f"Cash positions: {sum(1 for pos in positions if pos == 0)}")
    
    # Show some sample positions
    print("\nSample positions (first 10 assets):")
    for i, pos in enumerate(positions[:10]):
        print(f"Asset {i+1}: {pos:8.2f}")
    
    # Run backtest
    print("\n=== Running Backtest ===")
    equity_curve = backtest_equity(prices, getMyPosition, starting_equity=100000)
    
    # Calculate performance metrics
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    max_drawdown = calculate_max_drawdown(equity_curve)
    sharpe_ratio = calculate_sharpe_ratio(equity_curve)
    
    print(f"\n=== Performance Results ===")
    print(f"Starting Equity: ${equity_curve[0]:,.2f}")
    print(f"Ending Equity: ${equity_curve[-1]:,.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
    
    # Plot results
    plot_results(equity_curve, prices)
    
    return equity_curve, positions

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    peak = equity_curve[0]
    max_dd = 0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_dd:
            max_dd = drawdown
    
    return max_dd

def calculate_sharpe_ratio(equity_curve):
    """Calculate Sharpe ratio"""
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(returns) == 0:
        return 0
    
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0
    
    # Annualized Sharpe ratio (assuming daily data)
    sharpe = (avg_return * 252) / (std_return * np.sqrt(252))
    return sharpe

def plot_results(equity_curve, prices):
    """Plot the backtest results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot equity curve
    ax1.plot(equity_curve, 'b-', linewidth=2)
    ax1.set_title('Portfolio Equity Curve')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)
    
    # Plot some sample asset prices
    ax2.plot(prices[0, :], 'r-', alpha=0.7, label='Asset 1')
    ax2.plot(prices[1, :], 'g-', alpha=0.7, label='Asset 2')
    ax2.plot(prices[2, :], 'b-', alpha=0.7, label='Asset 3')
    ax2.set_title('Sample Asset Prices')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('algorithm_results.png', dpi=300, bbox_inches='tight')
    print("\nResults plot saved as 'algorithm_results.png'")

if __name__ == "__main__":
    run_algorithm_test() 