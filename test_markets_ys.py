import unittest
import numpy as np
import matplotlib.pyplot as plt
from markets_ys import Population, Economy
import os
import datetime

class TestPopulation(unittest.TestCase):
    def setUp(self):
        self.n_individuals = 1000
        self.pop = Population(n_individuals=self.n_individuals)

    def test_initialization(self):
        """Test basic initialization of Population class"""
        self.assertEqual(self.pop.n_individuals, self.n_individuals)
        self.assertEqual(len(self.pop.money), self.n_individuals)
        self.assertEqual(len(self.pop.stocksA), self.n_individuals)
        self.assertEqual(len(self.pop.savings_rates), self.n_individuals)

    def test_array_initialization(self):
        """Test initialization with array inputs"""
        custom_money = np.random.uniform(0.5, 1.5, self.n_individuals)
        custom_stocks = np.random.randint(500, 1501, self.n_individuals)
        custom_savings = np.random.uniform(0.8, 0.99, self.n_individuals)
        
        pop = Population(
            n_individuals=self.n_individuals,
            initial_money=custom_money,
            initial_stocksA=custom_stocks,
            savings_rates=custom_savings
        )
        
        np.testing.assert_array_almost_equal(pop.money, custom_money)
        np.testing.assert_array_almost_equal(pop.stocksA, custom_stocks)
        np.testing.assert_array_almost_equal(pop.savings_rates, custom_savings)

    def test_invalid_array_size(self):
        """Test that invalid array sizes raise ValueError"""
        with self.assertRaises(ValueError):
            Population(n_individuals=1000, initial_money=np.ones(500))

class TestEconomy(unittest.TestCase):
    def setUp(self):
        self.n_individuals = 1000
        self.economy = Economy(n_individuals=self.n_individuals)

    def test_initialization(self):
        """Test basic initialization of Economy class"""
        self.assertEqual(self.economy.total_money, self.n_individuals)
        self.assertEqual(self.economy.total_stocksA, self.n_individuals)
        self.assertEqual(self.economy.treasury, 0.0)
        self.assertEqual(self.economy.firmA_wealth, 0.0)
        self.assertEqual(self.economy.t, 0)

    def test_money_conservation(self):
        """Test that total money in the system is conserved"""
        initial_total = self.economy.total_money
        self.economy.step()
        final_total = (self.economy.population.money.sum() + 
                      self.economy.firmA_wealth + 
                      self.economy.treasury)
        self.assertAlmostEqual(initial_total, final_total)

    def test_stocks_conservation(self):
        """Test that total stocks remain constant"""
        initial_stocks = self.economy.total_stocksA
        self.economy.step()
        final_stocks = self.economy.population.stocksA.sum()
        self.assertEqual(initial_stocks, final_stocks)

    def test_no_negative_wealth(self):
        """Test that no agent's wealth goes below zero"""
        self.economy.run(n_steps=100)
        self.assertTrue(np.all(self.economy.population.money >= 0))

    def test_no_negative_stocks(self):
        """Test that no agent's stocks go below zero"""
        self.economy.run(n_steps=100)
        self.assertTrue(np.all(self.economy.population.stocksA >= 0))

    def test_trade_limits_respected(self):
        """Test that agents don't trade more than their limits"""
        self.economy.step()  # Run one step to generate share prices
        for i in range(self.n_individuals):
            # Check buy orders don't exceed money
            max_buy = self.economy.population.money[i] / self.economy.share_prices[i]
            self.assertLessEqual(max_buy, self.economy.max_money_to_trade[i]/2)
            
            # Check sell orders don't exceed stocks
            max_sell = self.economy.population.stocksA[i]
            self.assertLessEqual(max_sell, self.economy.max_money_to_trade[i]/2/self.economy.share_prices[i])

    def test_dividend_distribution(self):
        """Test that dividends are distributed proportionally to stock ownership"""
        # Set up a simple case with two agents
        economy = Economy(n_individuals=2, initial_stocksA=[100, 900])
        economy.firmA_wealth = 1000.0
        
        # Store initial money
        initial_money = economy.population.money.copy()
        
        # Pay dividends
        economy._pay_dividends()
        
        # Check that dividends were distributed proportionally
        total_dividends = economy.population.money - initial_money
        expected_ratio = 100/900  # Ratio of stocks
        actual_ratio = total_dividends[0]/total_dividends[1]
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=5)

    def test_order_book_consistency(self):
        """Test that order book maintains proper ordering and quantities"""
        economy = Economy(n_individuals=10)
        economy.step()  # Generate share prices
        
        # Add some orders
        for i in range(5):
            economy.order_book.add_buy_order(i, 6 + i, 1)
            economy.order_book.add_sell_order(i+5, 3+i, 1)
        
        # Check that best buy is highest price
        best_buy = economy.order_book.get_best_buy()
        self.assertIsNotNone(best_buy)
        self.assertAlmostEqual(best_buy.price, 10.0)
        
        # Check that best sell is lowest price
        best_sell = economy.order_book.get_best_sell()
        self.assertIsNotNone(best_sell)
        self.assertAlmostEqual(best_sell.price, 3.0)
        
        # Check that quantities are positive
        self.assertTrue(all(order.quantity > 0 for order in economy.order_book.buy_orders))
        self.assertTrue(all(order.quantity > 0 for order in economy.order_book.sell_orders))

def test_steady_state_hypothesis():
    """Test the hypothesis that steady state wealth is proportional to stocksA/(1-savings_rate)"""
    n_individuals = 500
    n_steps = 100
    
    # Initialize with random uniform distributions
    initial_money = np.random.uniform(0.3, 1.7, n_individuals)
    initial_stocks = np.random.randint(300, 1701, n_individuals)
    savings_rates = np.random.uniform(0.9, 0.95, n_individuals)
    
    # Create and run economy
    economy = Economy(
        n_individuals=n_individuals,
        initial_money=initial_money,
        initial_stocksA=initial_stocks,
        savings_rates=savings_rates
    )
    
    economy.run(n_steps)
    
    # Calculate the theoretical ratio
    theoretical_ratio = economy.population.stocksA / (1 - economy.population.savings_rates)
    
    # Get final wealths
    final_wealths = economy.population.money
    
    # Fit linear regression
    slope, intercept = np.polyfit(theoretical_ratio, final_wealths, 1)
    r_squared = np.corrcoef(theoretical_ratio, final_wealths)[0,1]**2
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(theoretical_ratio, final_wealths, alpha=0.6, label='Agent Data')
    
    # Plot fitted line
    x_line = np.linspace(min(theoretical_ratio), max(theoretical_ratio), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'r-', label=f'Fit (R² = {r_squared:.3f})')
    
    plt.xlabel('stocksA/(1-savings_rate)')
    plt.ylabel('Final Wealth')
    plt.title('Steady State Wealth vs Theoretical Ratio')
    plt.legend()
    plt.grid(True)
    
    # Save plot with timestamp
    os.makedirs('experiments/steady_state', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'experiments/steady_state/steady_state_{timestamp}.jpg')
    plt.close()
    
    print(f"Linear fit: wealth = {slope:.3f} * ratio + {intercept:.3f}")
    print(f"R-squared: {r_squared:.3f}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run steady state hypothesis test
    test_steady_state_hypothesis() 