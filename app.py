from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import json
from main import getMyPosition, backtest_equity
import time
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def calculate_advanced_metrics(equity_curve):
    """Calculate advanced performance metrics"""
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # Basic metrics
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
    
    # Volatility
    volatility = np.std(returns) * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate of 0.02)
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Win rate
    winning_days = np.sum(returns > 0)
    total_days = len(returns)
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # Sortino ratio (using downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_days': total_days,
        'winning_days': winning_days
    }

def create_equity_chart(equity_curve):
    """Create interactive equity curve chart"""
    days = list(range(len(equity_curve)))
    
    # Calculate drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak * 100
    
    # Create equity curve trace
    equity_trace = go.Scatter(
        x=days,
        y=equity_curve,
        mode='lines',
        name='Equity Curve',
        line=dict(color='#00ff88', width=3),
        hovertemplate='Day: %{x}<br>Equity: $%{y:,.0f}<extra></extra>'
    )
    
    # Create drawdown trace
    drawdown_trace = go.Scatter(
        x=days,
        y=drawdown,
        mode='lines',
        name='Drawdown %',
        line=dict(color='#ff0077', width=2),
        yaxis='y2',
        hovertemplate='Day: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    )
    
    layout = go.Layout(
        title='Portfolio Performance & Drawdown',
        xaxis=dict(title='Days'),
        yaxis=dict(title='Equity ($)', side='left'),
        yaxis2=dict(title='Drawdown (%)', side='right', overlaying='y'),
        template='plotly_dark',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(x=0.02, y=0.98)
    )
    
    fig = go.Figure(data=[equity_trace, drawdown_trace], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_price_chart(prices, positions):
    """Create interactive price chart with positions"""
    # Show first 10 instruments for clarity
    n_show = min(10, prices.shape[0])
    days = list(range(prices.shape[1]))
    
    traces = []
    for i in range(n_show):
        color = '#00ff88' if positions[i] > 0 else '#ff0077' if positions[i] < 0 else '#6b7280'
        line_width = 3 if positions[i] != 0 else 1
        
        trace = go.Scatter(
            x=days,
            y=prices[i, :],
            mode='lines',
            name=f'Asset {i+1}',
            line=dict(color=color, width=line_width),
            hovertemplate='Day: %{x}<br>Price: $%{y:.2f}<br>Position: %{customdata}<extra></extra>',
            customdata=[positions[i]] * len(days)
        )
        traces.append(trace)
    
    layout = go.Layout(
        title='Asset Price Evolution & Current Positions',
        xaxis=dict(title='Days'),
        yaxis=dict(title='Price ($)'),
        template='plotly_dark',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_position_chart(positions, prices):
    """Create position distribution chart"""
    current_prices = prices[:, -1]
    dollar_positions = positions * current_prices
    
    # Only show non-zero positions
    non_zero_indices = np.where(positions != 0)[0]
    non_zero_positions = positions[non_zero_indices]
    non_zero_dollar = dollar_positions[non_zero_indices]
    
    colors = ['#00ff88' if pos > 0 else '#ff0077' for pos in non_zero_positions]
    
    trace = go.Bar(
        x=[f'Asset {i+1}' for i in non_zero_indices],
        y=non_zero_dollar,
        marker_color=colors,
        text=[f'${d:,.0f}' for d in non_zero_dollar],
        textposition='auto',
        hovertemplate='Asset: %{x}<br>Position: %{y:,.0f}<extra></extra>'
    )
    
    layout = go.Layout(
        title='Position Distribution (Dollar Value)',
        xaxis=dict(title='Assets'),
        yaxis=dict(title='Position Value ($)'),
        template='plotly_dark',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_performance_metrics(prices, positions, equity_curve=None):
    """Calculate and return comprehensive performance metrics"""
    current_prices = prices[:, -1]
    dollar_positions = positions * current_prices
    
    # Basic position metrics
    metrics = {
        'total_exposure': float(np.sum(dollar_positions)),
        'num_positions': int(np.count_nonzero(positions)),
        'long_positions': int(np.sum(positions > 0)),
        'short_positions': int(np.sum(positions < 0)),
        'max_position': float(np.max(np.abs(dollar_positions))) if np.any(dollar_positions != 0) else 0,
        'avg_position_size': float(np.mean(np.abs(dollar_positions[dollar_positions != 0]))) if np.any(dollar_positions != 0) else 0,
        'position_concentration': float(np.std(dollar_positions[dollar_positions != 0])) if np.any(dollar_positions != 0) else 0
    }
    
    # Add equity curve metrics if available
    if equity_curve is not None:
        advanced_metrics = calculate_advanced_metrics(equity_curve)
        metrics.update(advanced_metrics)
    
    return metrics

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        start_time = time.time()
        
        # Get uploaded file or use real data
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                prices = np.loadtxt(file)
                prices = prices.T  # Transpose to instruments x days
            else:
                prices = load_real_data()
        else:
            prices = load_real_data()
        
        # Run improved algorithm
        positions = getMyPosition(prices)
        
        # Run comprehensive backtest
        equity_curve = backtest_equity(prices, getMyPosition, starting_equity=100000)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Calculate comprehensive metrics
        metrics = create_performance_metrics(prices, positions, equity_curve)
        
        # Create charts
        price_chart = create_price_chart(prices, positions)
        position_chart = create_position_chart(positions, prices)
        equity_chart = create_equity_chart(equity_curve)
        
        # Prepare position details
        current_prices = prices[:, -1]
        dollar_positions = positions * current_prices
        position_details = []
        for i in range(len(positions)):
            if positions[i] != 0:
                position_details.append({
                    'instrument': f'Asset {i+1}',
                    'position': int(positions[i]),
                    'price': float(current_prices[i]),
                    'dollar_value': float(dollar_positions[i]),
                    'type': 'Long' if positions[i] > 0 else 'Short'
                })
        
        position_details.sort(key=lambda x: abs(x['dollar_value']), reverse=True)
        
        # Prepare response
        result = {
            'success': True,
            'price_chart': price_chart,
            'position_chart': position_chart,
            'equity_chart': equity_chart,
            'metrics': metrics,
            'position_details': position_details,
            'data_shape': prices.shape,
            'execution_time': execution_time,
            'final_equity': equity_curve[-1],
            'total_return_pct': metrics['total_return'] * 100,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown_pct': metrics['max_drawdown'] * 100
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/sample-data')
def get_sample_data():
    """Load real data and return analysis"""
    try:
        start_time = time.time()
        
        # Load real price data
        prices = load_real_data()
        positions = getMyPosition(prices)
        
        # Run comprehensive backtest
        equity_curve = backtest_equity(prices, getMyPosition, starting_equity=100000)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Create charts
        price_chart = create_price_chart(prices, positions)
        position_chart = create_position_chart(positions, prices)
        equity_chart = create_equity_chart(equity_curve)
        
        # Calculate comprehensive metrics
        metrics = create_performance_metrics(prices, positions, equity_curve)
        
        # Prepare position details
        current_prices = prices[:, -1]
        dollar_positions = positions * current_prices
        position_details = []
        for i in range(len(positions)):
            if positions[i] != 0:
                position_details.append({
                    'instrument': f'Asset {i+1}',
                    'position': int(positions[i]),
                    'price': float(current_prices[i]),
                    'dollar_value': float(dollar_positions[i]),
                    'type': 'Long' if positions[i] > 0 else 'Short'
                })
        
        position_details.sort(key=lambda x: abs(x['dollar_value']), reverse=True)
        
        return jsonify({
            'success': True,
            'price_chart': price_chart,
            'position_chart': position_chart,
            'equity_chart': equity_chart,
            'metrics': metrics,
            'position_details': position_details,
            'data_shape': prices.shape,
            'execution_time': execution_time,
            'final_equity': equity_curve[-1],
            'total_return_pct': metrics['total_return'] * 100,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown_pct': metrics['max_drawdown'] * 100
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/upload-prices', methods=['POST'])
def upload_prices():
    try:
        start_time = time.time()
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded.'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected.'})
        
        # Load prices from uploaded file
        prices = np.loadtxt(file)
        if prices.shape[0] == 50:
            prices = prices  # Already in correct shape
        elif prices.shape[1] == 50:
            prices = prices.T  # Transpose if needed
        else:
            return jsonify({'success': False, 'error': f'Expected 50 instruments, got {prices.shape}.'})
        
        # Run improved algorithm
        positions = getMyPosition(prices)
        
        # Run comprehensive backtest
        equity_curve = backtest_equity(prices, getMyPosition, starting_equity=100000)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Create charts
        price_chart = create_price_chart(prices, positions)
        position_chart = create_position_chart(positions, prices)
        equity_chart = create_equity_chart(equity_curve)
        
        # Calculate comprehensive metrics
        metrics = create_performance_metrics(prices, positions, equity_curve)
        
        # Prepare position details
        current_prices = prices[:, -1]
        dollar_positions = positions * current_prices
        position_details = []
        for i in range(len(positions)):
            if positions[i] != 0:
                position_details.append({
                    'instrument': f'Asset {i+1}',
                    'position': int(positions[i]),
                    'price': float(current_prices[i]),
                    'dollar_value': float(dollar_positions[i]),
                    'type': 'Long' if positions[i] > 0 else 'Short'
                })
        
        position_details.sort(key=lambda x: abs(x['dollar_value']), reverse=True)
        
        return jsonify({
            'success': True,
            'price_chart': price_chart,
            'position_chart': position_chart,
            'equity_chart': equity_chart,
            'metrics': metrics,
            'position_details': position_details,
            'data_shape': prices.shape,
            'execution_time': execution_time,
            'final_equity': equity_curve[-1],
            'total_return_pct': metrics['total_return'] * 100,
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown_pct': metrics['max_drawdown'] * 100
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def load_real_data():
    """Load real price data from prices.txt"""
    try:
        prices = np.loadtxt('prices.txt')
        return prices
    except FileNotFoundError:
        # Fallback to sample data if prices.txt not found
        return generate_sample_data()

def generate_sample_data():
    """Generate realistic sample price data"""
    np.random.seed(42)
    n_instruments = 50
    n_days = 250
    
    # Generate base prices
    base_prices = np.random.uniform(50, 200, n_instruments)
    
    # Generate price evolution with trends and volatility
    prices = np.zeros((n_instruments, n_days))
    prices[:, 0] = base_prices
    
    for day in range(1, n_days):
        # Add trend component
        trend = np.random.normal(0, 0.02, n_instruments)
        # Add volatility component
        volatility = np.random.normal(0, 0.03, n_instruments)
        
        returns = trend + volatility
        prices[:, day] = prices[:, day-1] * (1 + returns)
    
    return prices

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print("🚀 Starting Algothon Quant Trading Algorithm Dashboard...")
    print(f"📊 Open http://localhost:{port} in your browser")
    print("🎯 Upload prices.txt or use real data to test the improved algorithm")
    app.run(debug=False, host='0.0.0.0', port=port) 