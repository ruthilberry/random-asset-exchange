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
        min_stock_frac_range: tuple = (0.1, 0.5),  # Range for minimum stock fraction
        lambda_risk_range: tuple = (0.05, 0.30),   # Range for risk aversion
    ):
        self.n_individuals = n_individuals
        
        self.money = self._initialize_array(initial_money, "initial_money")
        self.stocksA = np.asarray(self._initialize_array(initial_stocksA, "initial_stocksA"), dtype=int)
        self.savings_rates = self._initialize_array(savings_rates, "savings_rates")
        
        # each individual has a different min stock fraction and risk aversion
        self.min_stock_frac = np.random.uniform(min_stock_frac_range[0], min_stock_frac_range[1], n_individuals)
        self.lambda_risk = np.random.uniform(lambda_risk_range[0], lambda_risk_range[1], n_individuals)

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
        treasury: Government treasury (currently unused)
        firmA_wealth: Current wealth of firm A
        t: Current time step
        money_tax_rate: Tax rate on liquid money
        stock_tax_rate: Tax rate on stock wealth
    """
    def __init__(
        self,
        n_individuals: int = 1000,
        initial_money: Union[float, np.ndarray] = 1.0,
        initial_stocksA: Union[int, np.ndarray] = 1000,
        savings_rates: Union[float, np.ndarray] = 0.99,
        trading_limits: Union[float, np.ndarray] = 0.05,
        r: float = 0.1, # intertemporal discount rate
        E: float = 0.2, # error std deviation
        p: float = 0.5, # error probability
        money_tax_rate: float = 0.01,  # 1% tax on liquid money
        stock_tax_rate: float = 0.005,  # 0.5% tax on stock wealth
        min_stock_frac_range: tuple = (0.2, 0.5),  # Range for minimum stock fraction
        lambda_risk_range: tuple = (0.05, 0.30),   # Range for risk aversion
    ):
        self.population = Population(
            n_individuals=n_individuals,
            initial_money=initial_money,
            initial_stocksA=initial_stocksA,
            savings_rates=savings_rates,
            min_stock_frac_range=min_stock_frac_range,
            lambda_risk_range=lambda_risk_range
        )
        self.n_individuals = n_individuals
        self.trading_limits = self._initialize_array(trading_limits, "trading_limits")
        self.total_money = self.population.money.sum()
        self.total_stocksA = self.population.stocksA.sum()
        self.treasury = 0.0
        self.firmA_wealth = 0.0
        self.t = 0
        self.r = r
        self.E = E
        self.p = p
        self.money_tax_rate = money_tax_rate
        self.stock_tax_rate = stock_tax_rate
        self.order_book = OrderBook()  # Initialize the order book


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
        # error factor
        f = np.ones(self.n_individuals)
        f += np.random.binomial(1, self.p, size=self.n_individuals) * np.random.normal(0, self.E, size=self.n_individuals)
        f = np.clip(f, 0.1, 3) # avoid prices that are too close to zero
        # this way some fraction of individuals will wrongly estimate the value
        self.share_prices = EV * f
        assert np.all(self.share_prices > 0), f"""All share prices must 
        be strictly positive at time step {self.t}, previous revenue per share: {self.prev_revenue/self.total_stocksA}, error factor: {np.min(f)}, EV: {EV}"""

    def _compute_trade_limits(self) -> None:
        """Compute the trade limits for each agent.
            Note that each agent can perform (at most) that value of buy orders and that same value of sell orders. """
        self.max_money_to_trade = self.trading_limits * (self.population.money + self.population.stocksA * self.share_prices)


    def _perform_trades(self) -> None:
        """Perform trades from the order book."""

        # choose random permutation to add orders to order book
        eps = 1e-12 # small deviation to avoid equal buy and sell price
        order_indices = np.random.permutation(2*self.n_individuals)
        
        # Compute expected value for risk premium calculations
        EV = self.prev_revenue / self.total_stocksA / self.r
        
        for i in order_indices:
            if i < self.n_individuals:
                # buy order
                p = self.share_prices[i]
                
                # Apply risk premium rule for buying
                if self.t > 0:
                    reservation_buy = (1 + self.population.lambda_risk[i]) * EV
                    if p > reservation_buy:
                        continue  # price too high, skip buy order, not worth the risk
                
                quant = np.floor(min(self.population.money[i], self.max_money_to_trade[i]/2)/p)
                quant = min(quant, self.total_stocksA)
                self.order_book.add_buy_order(i, p*(1-eps), int(quant))
            else:
                # sell order
                idx = i - self.n_individuals
                p = self.share_prices[idx]
                
                # Apply portfolio reserve rule
                s_min = self.population.min_stock_frac[idx] * self.population.stocksA[idx]
                max_qty = self.population.stocksA[idx] - s_min
                if max_qty <= 0:
                    continue  # forbidden to sell; skip order
                
                # Apply risk premium rule for selling
                if self.t > 0:  # Skip at t=0 when EV is undefined
                    reservation_sell = (1 - self.population.lambda_risk[idx]) * EV
                    if p < reservation_sell:
                        continue  # price too low, skip sell order
                
                quant = np.floor(min(max_qty, self.max_money_to_trade[idx]/2/p))
                quant = min(quant, self.total_stocksA)
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
        self._compute_share_prices() # stored in self.share_prices
        self._compute_trade_limits() # stored in self.max_money_to_trade
        self._perform_trades()

    def _apply_wealth_tax(self) -> None:
        """Apply separate taxes on money and stock wealth, with money tax applied after dividends."""
        # First apply stock wealth tax
        stock_wealth = self.population.stocksA * self.share_prices
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
        self._apply_wealth_tax()
        self.t += 1

    def run(self, n_steps: int = 1000) -> None:
        """Run the economy for a specified number of steps."""
        for _ in range(n_steps):
            self.step()

    def plot_wealth_distribution(self) -> None:
        """Plot the Lorenz curve of agent wealth.
        
        Wealth is computed as: money + (prev_revenue/total_stocksA/r) * stocksA
        """
        # Compute wealth for each agent
        stock_value = self.prev_revenue / self.total_stocksA / self.r
        wealth = self.population.money + stock_value * self.population.stocksA
        
        # Sort wealth for Lorenz curve
        sorted_wealth = np.sort(wealth)
        
        # Compute cumulative wealth shares
        total_wealth = np.sum(sorted_wealth)
        cumulative_wealth = np.cumsum(sorted_wealth) / total_wealth
        
        # Create population shares (x-axis of Lorenz curve)
        n = len(wealth)
        population_shares = np.arange(1, n + 1) / n
        
        # Plot
        plt.figure(figsize=(8, 8))  # Square figure
        plt.plot(population_shares, cumulative_wealth, 'b-', label='Lorenz Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Equality')
        plt.xlabel('Cumulative Population Share')
        plt.ylabel('Cumulative Wealth Share')
        plt.title(f'Lorenz Curve of Wealth Distribution\nDefault Economy with n = {self.n_individuals} at t = {self.t}')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        os.makedirs('experiments/markets_ys', exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'experiments/markets_ys/lorenz_curve_t{self.t}_{timestamp}.jpg')
        plt.close()