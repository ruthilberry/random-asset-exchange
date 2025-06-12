import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
import heapq
from dataclasses import dataclass
from typing import List
import os
import datetime

@dataclass
class Order:
    """Represents a buy or sell order in the market."""
    agent_id: int
    price: float
    order_id: int
    quantity: int = 1  # Default to 1 stock per order

    def __lt__(self, other):
        # For buy orders: higher price first, then earlier order_id
        # For sell orders: lower price first, then earlier order_id
        if self.price != other.price:
            return self.price > other.price  # Note: > for buy orders, < for sell orders
        return self.order_id < other.order_id

class OrderBook:
    """Maintains buy and sell orders in the market."""
    def __init__(self):
        self.buy_orders: List[Order] = []  # Max heap for buy orders
        self.sell_orders: List[Order] = []  # Min heap for sell orders
        self.order_counter = 0

    def add_buy_order(self, agent_id: int, price: float, quantity: int = 1) -> None:
        """Add a buy order to the order book."""
        order = Order(agent_id, price, self.order_counter, quantity)
        heapq.heappush(self.buy_orders, order)
        self.order_counter += 1

    def add_sell_order(self, agent_id: int, price: float, quantity: int = 1) -> None:
        """Add a sell order to the order book."""
        order = Order(agent_id, -price, self.order_counter, quantity)  # Negative price for min heap
        heapq.heappush(self.sell_orders, order)
        self.order_counter += 1

    def get_best_buy(self) -> Optional[Order]:
        """Get the best buy order (highest price)."""
        return self.buy_orders[0] if self.buy_orders else None

    def get_best_sell(self) -> Optional[Order]:
        """Get the best sell order (lowest price)."""
        if not self.sell_orders:
            return None
        order = self.sell_orders[0]
        return Order(order.agent_id, -order.price, order.order_id, order.quantity)  # Convert back to positive price

    def fill_partially_best_buy(self, quantity: int) -> None:
        """Partially fill the best buy order with the specified quantity.
        If the order is fully filled (quantity reaches 0), it is removed."""
        if not self.buy_orders:
            return
        
        order = self.buy_orders[0]
        if quantity >= order.quantity:
            # Fully filled, remove the order
            heapq.heappop(self.buy_orders)
        else:
            # Partially filled, reduce quantity
            order.quantity -= quantity

    def fill_partially_best_sell(self, quantity: int) -> Optional[Order]:
        """Partially fill the best sell order with the specified quantity.
        If the order is fully filled (quantity reaches 0), it is removed."""
        if not self.sell_orders:
            return
        
        order = self.sell_orders[0]
        if quantity >= order.quantity:
            # Fully filled, remove the order
            heapq.heappop(self.sell_orders)
        else:
            # Partially filled, reduce quantity
            order.quantity -= quantity

    def clear(self) -> None:
        """Clear all orders from the book."""
        self.buy_orders.clear()
        self.sell_orders.clear()
        self.order_counter = 0

class Population:
    """Represents a population of economic agents.
    
    Each agent has:
    - money: their current wealth
    - stocksA: number of stocks owned in company A
    - savings_rates: fraction of wealth they save (s)
    - min_stock_frac: minimum fraction of stocks they must keep
    - lambda_risk: risk aversion parameter for trading decisions
    
    Attributes are stored in NumPy arrays for efficient operations.
    """
    def __init__(
        self,
        n_individuals: int = 1000, 
        initial_money: Union[float, np.ndarray] = 1.0,
        initial_stocksA: Union[int, np.ndarray] = 1000,
        savings_rates: Union[float, np.ndarray] = 0.99,
        min_stock_frac: Union[float, np.ndarray] = 0.0,  # minimum stock fraction
        lambda_risk: Union[float, np.ndarray] = 0.0,   # risk aversion parameter
    ):
        self.n_individuals = n_individuals
        
        self.money = self._initialize_array(initial_money, "initial_money")
        self.stocksA = np.asarray(self._initialize_array(initial_stocksA, "initial_stocksA"), dtype=int)
        self.savings_rates = self._initialize_array(savings_rates, "savings_rates")
        self.min_stock_frac = self._initialize_array(min_stock_frac, "min_stock_frac")
        self.lambda_risk = self._initialize_array(lambda_risk, "lambda_risk")

    def _initialize_array(self, value: Union[float, np.ndarray], name: str) -> np.ndarray:
        """Helper method to initialize arrays from scalar or array inputs."""
        if isinstance(value, (float, int)):
            return np.full(self.n_individuals, float(value))
        arr = np.asarray(value, dtype=float)
        if arr.size != self.n_individuals:
            raise ValueError(f"{name} size mismatch")
        return arr.copy()


class Economy:
    """Implements a simple economy with one firm and a population of agents.
    
    The economy operates through:
    1. Agents buying goods from the firm (spending non-saved wealth)
    2. Firm paying dividends to agents based on their stock ownership
    3. Agents exchanging stocks
    4. Time evolution through discrete steps
    5. Wealth taxation and redistribution
    
    Attributes:
        population: The population of agents
        total_money: Total money in the economy
        total_stocksA: Total number of stocks in company A
        firmA_wealth: Current wealth of firm A
        t: Current time step
        money_tax_rate: Tax rate on liquid money
        stock_tax_rate: Tax rate on stock wealth
    """

    # TODO: Elegir y justificar valores realistas de los parametros
    # TODO: Introducir un parametro dt que permita controlar el tiempo de cada paso
    #       Serviria para hacer simulaciones mas rapidas cuando nos sea interesante

    # TODO: Documentar wealth condensation. 
    # Se observa cuando las taxes son suficientemente bajas :)

    def __init__(
        self,
        n_individuals: int = 1000,
        initial_money: Union[float, np.ndarray] = 1.0,
        initial_stocksA: Union[int, np.ndarray] = 1000,
        savings_rates: Union[float, np.ndarray] = 0.99,
        trading_limits: Union[float, np.ndarray] = 0.05,
        r: float = 0.1, # intertemporal discount rate
        E: float = 0.2, # error std deviation
        p: float = 0.01, # error probability
        money_tax_rate: float = 0.0015,  # tax on liquid money, default 0.15%
        stock_tax_rate: float = 0.0,  # tax on stock wealth (currently not working)
        min_stock_frac: Union[float, np.ndarray] = 0.0,  # minimum stock fraction (currently not working)
        lambda_risk: Union[float, np.ndarray] = 0.0,   # risk aversion parameter
        markets_enabled: bool = True,  # whether to enable stock trading
    ):
        self.population = Population(
            n_individuals=n_individuals,
            initial_money=initial_money,
            initial_stocksA=initial_stocksA,
            savings_rates=savings_rates,
            min_stock_frac=min_stock_frac,
            lambda_risk=lambda_risk
        )
        self.n_individuals = n_individuals
        self.trading_limits = self._initialize_array(trading_limits, "trading_limits")
        self.total_money = self.population.money.sum()
        self.total_stocksA = self.population.stocksA.sum()
        self.firmA_wealth = 0.0
        self.t = 0
        self.r = r
        self.E = E
        self.p = p
        self.money_tax_rate = money_tax_rate
        self.stock_tax_rate = stock_tax_rate
        self.min_stock_frac = self._initialize_array(min_stock_frac, "min_stock_frac")
        self.lambda_risk = self._initialize_array(lambda_risk, "lambda_risk")
        self.markets_enabled = markets_enabled
        self.order_book = OrderBook() if markets_enabled else None
        
        # Store initial values for parameter documentation
        self.initial_money = initial_money
        self.initial_stocksA = initial_stocksA

    def _initialize_array(self, value: Union[float, np.ndarray], name: str) -> np.ndarray:
        """Helper method to initialize arrays from scalar or array inputs."""
        if isinstance(value, (float, int)):
            return np.full(self.n_individuals, float(value))
        arr = np.asarray(value, dtype=float)
        if arr.size != self.n_individuals:
            raise ValueError(f"{name} size mismatch")
        return arr.copy()

    
    def _buy_firmA(self) -> None:
        """Agents buy goods from firm A, spending their non-saved wealth."""
        spent_money = self.population.money * (1 - self.population.savings_rates)
        self.firmA_wealth = spent_money.sum()
        self.prev_revenue = self.firmA_wealth
        self.population.money *= self.population.savings_rates
    
    def _pay_dividends(self) -> None:
        """Firm A pays dividends to agents proportional to their stock ownership."""
        if self.total_stocksA == 0:
            return
        dividends = self.firmA_wealth * self.population.stocksA / self.total_stocksA
        self.population.money += dividends
        self.firmA_wealth = 0.0  # Reset firm wealth after paying dividends

    def _compute_share_prices(self) -> np.ndarray:
        """Compute the prices that each agent would pay for a share of firm A."""
        EV = self.prev_revenue / self.total_stocksA / self.r # expected value of a share

        # error factor (random values corresponding to judgement errors)
        f = np.ones(self.n_individuals)
        f += np.random.binomial(1, self.p, size=self.n_individuals) * np.random.normal(0, self.E, size=self.n_individuals)
        f = np.clip(f, 0.1, 3) # avoid prices that are too close to zero
        # this way some fraction of individuals will wrongly estimate the value

        # Multiply by the error factor and the risk aversion factor
        self.share_prices = EV * (1 - self.population.lambda_risk) * f
        assert np.all(self.share_prices > 0), f"""All share prices must 
        be strictly positive at time step {self.t}, previous revenue per share: {self.prev_revenue/self.total_stocksA}, error factor: {np.min(f)}, EV: {EV}"""

    def _compute_trade_limits(self) -> None:
        """Compute the trade limits for each agent.
            Note that each agent can perform (at most) that value of buy orders and that same value of sell orders. """
        self.max_money_to_trade = self.trading_limits * (self.population.money + self.population.stocksA * self.share_prices)


    def _perform_trades(self) -> None:
        """Perform trades from the order book."""

        # choose random permutation to add orders to order book
        order_indices = np.random.permutation(2*self.n_individuals)

        eps = 1e-14 # small deviation to avoid equal buy and sell price
        
        for i in order_indices:
            if i < self.n_individuals:
                # buy order
                p = self.share_prices[i]*(1-eps) # multiply by 1-eps to avoid all buy and sell values being the same

                quant = np.floor(min(self.population.money[i], self.max_money_to_trade[i])/p)
                quant = min(quant, self.total_stocksA)
                if quant == 0:
                    continue
                self.order_book.add_buy_order(i, p*(1-eps), int(quant))
            else:
                # sell order
                idx = i - self.n_individuals
                p = self.share_prices[idx]* (1 + eps) # multiply by 1+eps to avoid all buy and sell values being the same

                quant = np.floor(min(self.population.stocksA[idx], self.max_money_to_trade[idx]/p))

                # Apply portfolio reserve rule
                # TODO: Apply portfolio reserve rule

                if quant == 0:
                    continue
                self.order_book.add_sell_order(idx, p*(1+eps), int(quant))
            
            # perform a trade if appropriate
            if self.order_book.get_best_buy() and self.order_book.get_best_sell():
                if self.order_book.get_best_buy().price >= self.order_book.get_best_sell().price:
                    # perform a trade
                    sell_order = self.order_book.get_best_sell()
                    buy_order = self.order_book.get_best_buy()
                    q = min(sell_order.quantity, buy_order.quantity)
                    p = (sell_order.price + buy_order.price)/2
                    self.population.money[sell_order.agent_id] += p*q
                    self.population.money[buy_order.agent_id] -= p*q
                    self.population.stocksA[sell_order.agent_id] -= q
                    self.population.stocksA[buy_order.agent_id] += q
                    # Fill orders partially
                    self.order_book.fill_partially_best_sell(q)
                    self.order_book.fill_partially_best_buy(q)
        
        self.order_book.clear()

    def _trade_phase(self) -> None:
        """Agents trade shares of firm A."""
        if not self.markets_enabled:
            return
            
        self._compute_share_prices() # stored in self.share_prices
        self._compute_trade_limits() # stored in self.max_money_to_trade
        self._perform_trades()

    def _apply_taxes(self) -> None:
        """Apply separate taxes on money and stock wealth, and redistribute collected taxes among individuals"""
        # TODO: Figure out how to apply stock wealth tax without sending agents into negative wealth
        # Maybe if you can't pay with money you are forced to sell some shares to the government and then the government
        # participates in the stock market only to sell those shares
        # First apply stock wealth tax
        if self.money_tax_rate == 0.0 and self.stock_tax_rate == 0.0:
            return
        stock_wealth = self.population.stocksA * self.prev_revenue/self.total_stocksA/self.r # compute stock wealth based on EV
        stock_tax = stock_wealth * self.stock_tax_rate
        
        # Then apply money tax (after dividends)
        money_tax = self.population.money * self.money_tax_rate
        
        # Total tax collection
        total_tax = stock_tax.sum() + money_tax.sum()
        
        # Redistribute equally among all agents
        redistribution = total_tax / self.n_individuals
        
        # Apply taxes and redistribution
        self.population.money += redistribution - money_tax - stock_tax

    def step(self) -> None:
        """Perform one step of the economic simulation."""
        self._buy_firmA()
        self._pay_dividends()
        self._trade_phase()
        self._apply_taxes()
        self.t += 1

    def run(self, n_steps: int = 1000) -> None:
        """Run the economy for a specified number of steps."""
        for _ in range(n_steps):
            self.step()

    def document_wealth_distribution(self) -> None:
        """Plot and store wealth distribution analysis including:
        - Configuration parameters
        - Lorenz curve of total wealth
        - Histogram of money distribution
        - Histogram of stock ownership
        """
        # Compute wealth components
        stock_value = self.prev_revenue / self.total_stocksA / self.r
        money = self.population.money
        stocks = self.population.stocksA
        total_wealth = money + stock_value * stocks

        
        # Compute Lorenz curve
        sorted_wealth = np.sort(total_wealth)
        cumulative_wealth = np.cumsum(sorted_wealth) / np.sum(sorted_wealth)
        population_shares = np.arange(1, len(total_wealth) + 1) / len(total_wealth)
        
        # Calculate Gini coefficient
        gini = 1 - 2 * np.trapezoid(cumulative_wealth, population_shares)


        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7.5))
        
        # Create parameter text
        def format_array_param(arr):
            if isinstance(arr, float) or isinstance(arr, int):
                return str(arr)
            if np.min(arr) == np.max(arr):
                return str(arr[0])
            return f'non-constant array with average {np.mean(arr):.3f}'

        param_text = f"""Simulation Parameters (t={self.t}):
markets_enabled: {self.markets_enabled}
n_individuals: {self.n_individuals}
initial_money: {format_array_param(self.initial_money)}
initial_stocksA: {format_array_param(self.initial_stocksA)}
savings_rates: {format_array_param(self.population.savings_rates)}
trading_limits: {format_array_param(self.trading_limits)}
r: {self.r}
E: {self.E}
p: {self.p}
money_tax_rate: {self.money_tax_rate}
stock_tax_rate: {self.stock_tax_rate}
min_stock_frac: {format_array_param(self.min_stock_frac)}
lambda_risk: {format_array_param(self.lambda_risk)}
Gini coefficient: {gini:.3f}"""
        
        # Plot parameters in first subplot
        ax1.axis('off')
        ax1.text(0.1, 0.5, param_text, fontsize=11, family='monospace', va='center')
        
        # Plot Lorenz curve in second subplot
        ax2.plot(population_shares, cumulative_wealth, 'b-', label='Lorenz Curve')
        ax2.plot([0, 1], [0, 1], 'k--', label='Perfect Equality')
        ax2.set_title(f'Lorenz Curve (Gini = {gini:.3f})', fontsize=11)
        ax2.set_xlabel('Cumulative Population Share', fontsize=11)
        ax2.set_ylabel('Cumulative Wealth Share', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        ax2.tick_params(axis='both', which='major', labelsize=11)
        
        # Plot money histogram in third subplot
        ax3.hist(money, bins=50, alpha=0.7, color='blue')
        ax3.set_title('Distribution of Money', fontsize=11)
        ax3.set_xlabel('Money', fontsize=11)
        ax3.set_ylabel('Number of Agents', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=11)
        
        # Plot stocks histogram in fourth subplot
        ax4.hist(stocks, bins=50, alpha=0.7, color='green')
        ax4.set_title('Distribution of Stock Ownership', fontsize=11)
        ax4.set_xlabel('Number of Stocks', fontsize=11)
        ax4.set_ylabel('Number of Agents', fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', which='major', labelsize=11)
        
        # Adjust layout and add main title
        plt.suptitle(f'Wealth Distribution Analysis at t = {self.t}\nPopulation Size: {self.n_individuals}', fontsize=11)
        plt.tight_layout()
        
        # Save plot with human-readable timestamp
        os.makedirs('experiments/markets_ys', exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%H:%M_%B_%d") 
        plt.savefig(f'experiments/markets_ys/wdist_{timestamp}.jpg')
        
        # Show plot
        plt.show()
        
        # Close the figure to free memory
        plt.close()

        
    def document_lorenz_curve(self) -> None:
        """Plot and store Lorenz curve of total wealth
        """
        # Compute wealth components
        stock_value = self.prev_revenue / self.total_stocksA / self.r
        money = self.population.money
        stocks = self.population.stocksA
        total_wealth = money + stock_value * stocks

        
        # Compute Lorenz curve
        sorted_wealth = np.sort(total_wealth)
        cumulative_wealth = np.cumsum(sorted_wealth) / np.sum(sorted_wealth)
        population_shares = np.arange(1, len(total_wealth) + 1) / len(total_wealth)
        
        # Calculate Gini coefficient
        gini = 1 - 2 * np.trapezoid(cumulative_wealth, population_shares)


        # Create figure with 2x2 subplots
        fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))

        # Plot Lorenz curve in second subplot
        ax.plot(population_shares, cumulative_wealth, 'b-', label='Lorenz Curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Equality')
        ax.set_title(f'Lorenz Curve (Gini = {gini:.3f})', fontsize=20)
        ax.set_xlabel('Cumulative Population Share', fontsize=20)
        ax.set_ylabel('Cumulative Wealth Share', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

        # Adjust layout and add main title
        plt.tight_layout()
        
        # Save plot with human-readable timestamp
        os.makedirs('experiments/markets_ys', exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%H:%M_%B_%d") 
        plt.savefig(f'experiments/markets_ys/lorenz_{timestamp}.jpg')
        
        # Show plot
        plt.show()
        
        # Close the figure to free memory
        plt.close()