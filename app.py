from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import json
from main import getMyPosition
import time
import os

app = Flask(__name__)

def create_price_chart(prices, positions):
    """Create interactive price chart with positions"""
    # Sample price data for visualization (first 5 instruments)
    days = list(range(prices.shape[1]))
    
    traces = []
    for i in range(min(5, prices.shape[0])):
        trace = go.Scatter(
            x=days,
            y=prices[i, :],
            mode='lines',
            name=f'Instrument {i+1}',
            line=dict(width=2),
            hovertemplate='Day: %{x}<br>Price: $%{y:.2f}<br>Position: %{customdata}<extra></extra>',
            customdata=[positions[i]] * len(days)
        )
        traces.append(trace)
    
    layout = go.Layout(
        title='Price Evolution & Positions',
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
        x=[f'Inst {i+1}' for i in non_zero_indices],
        y=non_zero_positions,
        marker_color=colors,
        text=[f'${d:.0f}' for d in non_zero_dollar],
        textposition='auto',
        hovertemplate='Instrument: %{x}<br>Position: %{y}<br>Value: %{text}<extra></extra>'
    )
    
    layout = go.Layout(
        title='Position Distribution',
        xaxis=dict(title='Instruments'),
        yaxis=dict(title='Position Size'),
        template='plotly_dark',
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_performance_metrics(prices, positions):
    """Calculate and return performance metrics"""
    current_prices = prices[:, -1]
    dollar_positions = positions * current_prices
    
    # Calculate returns (simplified)
    if prices.shape[1] > 1:
        returns = (prices[:, -1] - prices[:, -2]) / prices[:, -2]
        portfolio_return = np.sum(positions * returns)
    else:
        portfolio_return = 0
    
    metrics = {
        'total_exposure': float(np.sum(dollar_positions)),
        'num_positions': int(np.count_nonzero(positions)),
        'long_positions': int(np.sum(positions > 0)),
        'short_positions': int(np.sum(positions < 0)),
        'max_position': float(np.max(np.abs(dollar_positions))),
        'portfolio_return': float(portfolio_return),
        'avg_position_size': float(np.mean(np.abs(dollar_positions[dollar_positions != 0]))) if np.any(dollar_positions != 0) else 0,
        'position_concentration': float(np.std(dollar_positions[dollar_positions != 0])) if np.any(dollar_positions != 0) else 0
    }
    
    return metrics

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # Get uploaded file or use sample data
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                prices = np.loadtxt(file)
                prices = prices.T  # Transpose to instruments x days
            else:
                prices = generate_sample_data()
        else:
            prices = generate_sample_data()
        
        # Run algorithm
        start_time = time.time()
        positions = getMyPosition(prices)
        execution_time = time.time() - start_time
        
        # Create visualizations
        price_chart = create_price_chart(prices, positions)
        position_chart = create_position_chart(positions, prices)
        metrics = create_performance_metrics(prices, positions)
        
        # Prepare detailed position data
        current_prices = prices[:, -1]
        dollar_positions = positions * current_prices
        
        position_details = []
        for i in range(len(positions)):
            if positions[i] != 0:
                position_details.append({
                    'instrument': f'Instrument {i+1}',
                    'position': int(positions[i]),
                    'price': float(current_prices[i]),
                    'dollar_value': float(dollar_positions[i]),
                    'type': 'Long' if positions[i] > 0 else 'Short'
                })
        
        # Sort by absolute dollar value
        position_details.sort(key=lambda x: abs(x['dollar_value']), reverse=True)
        
        return jsonify({
            'success': True,
            'execution_time': execution_time,
            'price_chart': price_chart,
            'position_chart': position_chart,
            'metrics': metrics,
            'position_details': position_details,
            'data_shape': prices.shape
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/sample-data')
def get_sample_data():
    """Generate and return sample data"""
    prices = generate_sample_data()
    positions = getMyPosition(prices)
    
    price_chart = create_price_chart(prices, positions)
    position_chart = create_position_chart(positions, prices)
    metrics = create_performance_metrics(prices, positions)
    
    current_prices = prices[:, -1]
    dollar_positions = positions * current_prices
    
    position_details = []
    for i in range(len(positions)):
        if positions[i] != 0:
            position_details.append({
                'instrument': f'Instrument {i+1}',
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
        'metrics': metrics,
        'position_details': position_details,
        'data_shape': prices.shape
    })

def generate_sample_data():
    """Generate realistic sample price data"""
    np.random.seed(42)
    n_instruments = 50
    n_days = 100
    
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
    port = int(os.environ.get('PORT', 5000))
    print("🚀 Starting Algothon Quant Trading Algorithm Dashboard...")
    print(f"📊 Open http://localhost:{port} in your browser")
    print("🎯 Upload prices.txt or use sample data to test the algorithm")
    app.run(debug=False, host='0.0.0.0', port=port) 