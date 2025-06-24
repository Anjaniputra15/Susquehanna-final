export default function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  // Simulate algorithm results (since we can't run Python on Vercel)
  const simulateAlgorithm = () => {
    // Generate realistic sample data
    const nInstruments = 50;
    const positions = [];
    const prices = [];
    
    // Generate positions (mostly zeros, some active)
    for (let i = 0; i < nInstruments; i++) {
      if (Math.random() < 0.3) { // 30% chance of having a position
        const position = Math.random() > 0.5 ? 
          Math.floor(Math.random() * 100) + 1 : 
          -(Math.floor(Math.random() * 100) + 1);
        positions.push(position);
      } else {
        positions.push(0);
      }
      
      // Generate realistic prices
      prices.push(50 + Math.random() * 150);
    }
    
    return { positions, prices };
  };

  const { positions, prices } = simulateAlgorithm();
  
  // Calculate metrics
  const dollarPositions = positions.map((pos, i) => pos * prices[i]);
  const totalExposure = dollarPositions.reduce((sum, val) => sum + Math.abs(val), 0);
  const numPositions = positions.filter(pos => pos !== 0).length;
  const longPositions = positions.filter(pos => pos > 0).length;
  const shortPositions = positions.filter(pos => pos < 0).length;
  const maxPosition = Math.max(...dollarPositions.map(Math.abs));
  const avgPositionSize = numPositions > 0 ? 
    dollarPositions.filter(val => val !== 0).reduce((sum, val) => sum + Math.abs(val), 0) / numPositions : 0;
  const portfolioReturn = (Math.random() - 0.5) * 0.1; // Simulate -5% to +5% return

  // Create position details
  const positionDetails = [];
  positions.forEach((position, i) => {
    if (position !== 0) {
      positionDetails.push({
        instrument: `Instrument ${i + 1}`,
        position: position,
        price: prices[i].toFixed(2),
        dollar_value: dollarPositions[i],
        type: position > 0 ? 'Long' : 'Short'
      });
    }
  });

  // Sort by absolute dollar value
  positionDetails.sort((a, b) => Math.abs(b.dollar_value) - Math.abs(a.dollar_value));

  const response = {
    success: true,
    metrics: {
      total_exposure: totalExposure,
      num_positions: numPositions,
      long_positions: longPositions,
      short_positions: shortPositions,
      max_position: maxPosition,
      portfolio_return: portfolioReturn,
      avg_position_size: avgPositionSize,
      position_concentration: Math.random() * 1000 // Simulate concentration
    },
    position_details: positionDetails,
    data_shape: [50, 100] // Simulate 50 instruments, 100 days
  };

  res.status(200).json(response);
} 