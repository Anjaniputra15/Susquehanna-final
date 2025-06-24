from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import sys
import os

# Add the parent directory to the path to import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import getMyPosition

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

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/sample-data':
            try:
                # Generate sample data
                prices = generate_sample_data()
                positions = getMyPosition(prices)
                metrics = create_performance_metrics(prices, positions)
                
                # Prepare response
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
                
                response = {
                    'success': True,
                    'metrics': metrics,
                    'position_details': position_details,
                    'data_shape': prices.shape
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers() 