import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional
import heapq
from dataclasses import dataclass
from typing import List

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
    
    Attributes are stored in NumPy arrays for efficient operations.
    """
    def __init__(
        self,
        n_individuals: int = 1000, 
        initial_money: Union[float, np.ndarray] = 1.0,
        initial_stocksA: Union[int, np.ndarray] = 1000,
        savings_rates: Union[float, np.ndarray] = 0.99,
    ):
        self.n_individuals = n_individuals
        
        # Handle scalar or array inputs for each attribute
        self.money = self._initialize_array(initial_money, "initial_money")
        self.stocksA = np.asarray(self._initialize_array(initial_stocksA, "initial_stocksA"), dtype=int)
        self.savings_rates = self._initialize_array(savings_rates, "savings_rates")

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
    3. Time evolution through discrete steps
    
    Attributes:
        population: The population of agents
        total_money: Total money in the economy
        total_stocksA: Total number of stocks in company A
        treasury: Government treasury (currently unused)
        firmA_wealth: Current wealth of firm A
        t: Current time step
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
    ):
        self.population = Population(
            n_individuals=n_individuals,
            initial_money=initial_money,
            initial_stocksA=initial_stocksA,
            savings_rates=savings_rates
        )
        self.trading_limits = self._initialize_array(trading_limits, "trading_limits")
        self.total_money = self.population.money.sum()
        self.total_stocksA = self.population.stocksA.sum()
        self.treasury = 0.0
        self.firmA_wealth = 0.0
        self.t = 0
        self.r = r
        self.E = E
        self.p = p
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
        self.firmA_wealth += spent_money.sum()
        self.population.money *= self.population.savings_rates
    
    def _pay_dividends(self) -> None:
        """Firm A pays dividends to agents proportional to their stock ownership."""
        if self.total_stocksA == 0:
            return
        dividends = self.firmA_wealth * self.population.stocksA / self.total_stocksA
        self.prev_dividends = dividends
        self.population.money += dividends
        self.firmA_wealth = 0.0  # Reset firm wealth after paying dividends

    def _compute_share_prices(self) -> np.ndarray:
        """Compute the prices that each agent would pay for a share of firm A."""
        EV = self.prev_dividends / self.r
        # error factor
        f = np.ones(self.n_individuals)
        f += np.random.binomial(1, self.p, size=self.n_individuals) * np.random.normal(0, self.E, size=self.n_individuals)
        f = np.clip(f, 0.1, 3) # avoid prices that are too close to zero
        # this way some fraction of individuals will wrongly estimate the value
        self.share_prices = EV * f

    def _compute_trade_limits(self) -> None:
        """Compute the trade limits for each agent.
            Note that each agent can perform (at most) that value of buy orders and that same value of sell orders. """
        self.max_money_to_trade = self.trading_limits * (self.population.money + self.population.stocksA * self.share_prices)


    def _perform_trades(self) -> None:
        """Perform trades from the order book."""
        # choose random order to add orders to order book
        eps = 1e-12 # small deviation to avoid equal buy and sell price
        order_indices = np.random.permutation(2*self.n_individuals)
        for i in order_indices:
            if i < self.n_individuals:
                # buy order
                p = self.share_prices[i]
                quant = np.floor(min(self.population.money[i], self.max_money_to_trade[i]/2)/p)
                self.order_book.add_buy_order(i, p*(1-eps), int(quant))
            else:
                # sell order
                p = self.share_prices[i-self.n_individuals]
                quant = np.floor(min(self.population.stocksA[i-self.n_individuals], self.max_money_to_trade[i-self.n_individuals]/2/p))
                self.order_book.add_sell_order(i, p*(1+eps), int(quant))
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


    def step(self) -> None:
        """Perform one time step of the economy."""
        self._buy_firmA()
        self._pay_dividends()
        if self.t != 0:
            self._trade_phase()
        self.t += 1

    def run(self, n_steps: int = 1000) -> None:
        """Run the economy for a specified number of steps."""
        for _ in range(n_steps):
            self.step()