#!/usr/bin/env python3
"""
Comprehensive Analysis: Day 1 to Day 999 (or full dataset)
Analyzes the trading algorithm performance over the entire period
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from main import getMyPosition, backtest_equity
import seaborn as sns

def load_prices_from_file(filename='prices.txt'):
    """Load prices from the prices.txt file"""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        prices_data = []
        for line in lines:
            price_values = [float(x) for x in line.strip().split()]
            prices_data.append(price_values)
        
        prices_array = np.array(prices_data)
        return prices_array
        
    except Exception as e:
        print(f"Error loading prices: {e}")
        return None

def calculate_performance_metrics(equity_curve):
    """Calculate comprehensive performance metrics"""
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # Basic metrics
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
    
    # Risk metrics
    volatility = np.std(returns) * np.sqrt(252)
    max_drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)
    max_dd = np.max(max_drawdown)
    
    # Risk-adjusted metrics
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    sortino_ratio = np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252) if np.std(returns[returns < 0]) > 0 else 0
    
    # Win/Loss metrics
    winning_days = np.sum(returns > 0)
    losing_days = np.sum(returns < 0)
    win_rate = winning_days / len(returns) if len(returns) > 0 else 0
    
    # Maximum consecutive wins/losses
    consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    
    for ret in returns:
        if ret > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'win_rate': win_rate,
        'winning_days': winning_days,
        'losing_days': losing_days,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'total_days': len(equity_curve),
        'final_equity': equity_curve[-1],
        'peak_equity': np.max(equity_curve)
    }

def analyze_periods(equity_curve, prices):
    """Analyze performance in different periods"""
    total_days = len(equity_curve)
    
    # Define periods
    periods = {
        'First Quarter': (0, total_days // 4),
        'Second Quarter': (total_days // 4, total_days // 2),
        'Third Quarter': (total_days // 2, 3 * total_days // 4),
        'Fourth Quarter': (3 * total_days // 4, total_days)
    }
    
    period_analysis = {}
    for period_name, (start, end) in periods.items():
        period_equity = equity_curve[start:end]
        if len(period_equity) > 1:
            period_return = (period_equity[-1] - period_equity[0]) / period_equity[0]
            period_volatility = np.std(np.diff(period_equity) / period_equity[:-1]) * np.sqrt(252)
            period_analysis[period_name] = {
                'return': period_return,
                'volatility': period_volatility,
                'days': len(period_equity)
            }
    
    return period_analysis

def plot_comprehensive_analysis(equity_curve, prices, metrics, period_analysis):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Trading Algorithm Performance Analysis: Day 1 to End', fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    axes[0, 0].plot(equity_curve, 'b-', linewidth=2, label='Portfolio Equity')
    axes[0, 0].axhline(y=100000, color='r', linestyle='--', alpha=0.7, label='Starting Equity')
    axes[0, 0].set_title('Portfolio Equity Curve')
    axes[0, 0].set_ylabel('Equity ($)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Daily Returns Distribution
    returns = np.diff(equity_curve) / equity_curve[:-1]
    axes[0, 1].hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Daily Returns Distribution')
    axes[0, 1].set_xlabel('Daily Return')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Drawdown Analysis
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (running_max - equity_curve) / running_max
    axes[1, 0].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
    axes[1, 0].plot(drawdown, 'r-', linewidth=1)
    axes[1, 0].set_title('Drawdown Analysis')
    axes[1, 0].set_ylabel('Drawdown (%)')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Period Performance
    if period_analysis:
        periods = list(period_analysis.keys())
        returns = [period_analysis[p]['return'] for p in periods]
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        bars = axes[1, 1].bar(periods, returns, color=colors, alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_title('Performance by Period')
        axes[1, 1].set_ylabel('Return (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{ret:.2%}', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("Comprehensive analysis plot saved as 'comprehensive_analysis.png'")

def print_detailed_analysis(metrics, period_analysis):
    """Print detailed analysis results"""
    print("\n" + "="*60)
    print("COMPREHENSIVE TRADING ALGORITHM ANALYSIS")
    print("Day 1 to Day", metrics['total_days'])
    print("="*60)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Starting Equity:     ${100000:,.2f}")
    print(f"   Final Equity:        ${metrics['final_equity']:,.2f}")
    print(f"   Peak Equity:         ${metrics['peak_equity']:,.2f}")
    print(f"   Total Return:        {metrics['total_return']:.2%}")
    print(f"   Annualized Return:   {metrics['annualized_return']:.2%}")
    print(f"   Total Trading Days:  {metrics['total_days']}")
    
    print(f"\n📈 RISK METRICS:")
    print(f"   Volatility:          {metrics['volatility']:.2%}")
    print(f"   Max Drawdown:        {metrics['max_drawdown']:.2%}")
    print(f"   Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
    print(f"   Sortino Ratio:       {metrics['sortino_ratio']:.3f}")
    
    print(f"\n🎯 TRADING STATISTICS:")
    print(f"   Win Rate:            {metrics['win_rate']:.2%}")
    print(f"   Winning Days:        {metrics['winning_days']}")
    print(f"   Losing Days:         {metrics['losing_days']}")
    print(f"   Max Consecutive Wins: {metrics['max_consecutive_wins']}")
    print(f"   Max Consecutive Losses: {metrics['max_consecutive_losses']}")
    
    if period_analysis:
        print(f"\n📅 PERIOD ANALYSIS:")
        for period, data in period_analysis.items():
            print(f"   {period}: {data['return']:.2%} return, {data['volatility']:.2%} volatility")
    
    print(f"\n💡 KEY INSIGHTS:")
    if metrics['total_return'] > 0:
        print(f"   ✅ Algorithm generated positive returns")
    else:
        print(f"   ❌ Algorithm generated negative returns")
    
    if metrics['sharpe_ratio'] > 0:
        print(f"   ✅ Positive risk-adjusted returns (Sharpe > 0)")
    else:
        print(f"   ❌ Negative risk-adjusted returns (Sharpe < 0)")
    
    if metrics['max_drawdown'] < 0.10:
        print(f"   ✅ Low maximum drawdown (< 10%)")
    else:
        print(f"   ⚠️  High maximum drawdown (> 10%)")
    
    if metrics['win_rate'] > 0.5:
        print(f"   ✅ More winning days than losing days")
    else:
        print(f"   ❌ More losing days than winning days")

def main():
    """Main analysis function"""
    print("=== Comprehensive Trading Algorithm Analysis ===")
    print("Analyzing performance from Day 1 to end of dataset\n")
    
    # Load prices
    prices = load_prices_from_file('prices.txt')
    if prices is None:
        return
    
    print(f"Loaded {prices.shape[0]} days of data for {prices.shape[1]} assets")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Transpose prices for algorithm
    prices = prices.T
    print(f"Transposed to {prices.shape[0]} assets x {prices.shape[1]} days\n")
    
    # Run backtest
    print("Running comprehensive backtest...")
    equity_curve = backtest_equity(prices, getMyPosition, 100000)
    
    # Calculate metrics
    metrics = calculate_performance_metrics(equity_curve)
    
    # Analyze periods
    period_analysis = analyze_periods(equity_curve, prices)
    
    # Print analysis
    print_detailed_analysis(metrics, period_analysis)
    
    # Create visualization
    plot_comprehensive_analysis(equity_curve, prices, metrics, period_analysis)
    
    print(f"\n✅ Analysis complete! Check 'comprehensive_analysis.png' for visualizations.")

if __name__ == "__main__":
    main() 