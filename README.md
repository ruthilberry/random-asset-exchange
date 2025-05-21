# Random Asset Exchange Simulation

An economic simulation that models market dynamics and wealth distribution through agent-based trading. This project explores how wealth inequality emerges from simple market interactions and trading behaviors.

## Overview

The simulation implements a market with multiple economic agents trading stocks in a single company. It models realistic market behavior including:
- Order book mechanics
- Price discovery
- Dividend distribution
- Wealth accumulation
- Market psychology through price perception errors

## Key Components

### Order System
- `Order` class: Represents individual buy/sell orders
- `OrderBook` class: Maintains priority queues for buy/sell orders
- Implements price-time priority matching

### Population
- Models a group of economic agents
- Each agent has:
  - Money (wealth)
  - Stocks in company A
  - Savings rate

### Economy
- Main simulation engine
- Handles:
  - Trading cycles
  - Dividend distribution
  - Wealth tracking
  - Market dynamics

## Features

- **Realistic Market Mechanics**: Implements a proper order book system
- **Psychological Factors**: Models price perception errors
- **Wealth Distribution**: Tracks and visualizes wealth distribution over time
- **Configurable Parameters**: 
  - Number of agents
  - Initial wealth distribution
  - Savings rates
  - Trading limits
  - Market error parameters

## Usage

```python
from markets_ys import Economy

# Create an economy with default parameters
economy = Economy(
    n_individuals=1000,
    initial_money=1.0,
    initial_stocksA=1000,
    savings_rates=0.99,
    trading_limits=0.05,
    r=0.1,  # intertemporal discount rate
    E=0.2,  # error std deviation
    p=0.5   # error probability
)

# Run the simulation for 1000 steps
economy.run(n_steps=1000)

# Plot the wealth distribution
economy.plot_wealth_distribution()
```

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Research Applications

This simulation is particularly useful for studying:
- Wealth inequality emergence
- Market efficiency
- Price discovery mechanisms
- Impact of trading behaviors on market stability
- Effects of psychological factors on market dynamics
