import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional

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
        dt: float = 1.0,  # Time step size for future continuous-time implementation
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
        self.dt = dt
        self.r = r
        self.E = E
        self.p = p


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
        # this way some fraction of individuals will wrongly estimate the value
        self.share_prices = EV * f

    def _compute_trade_limits(self) -> None:
        """Compute the trade limits for each agent."""
        

    def _trade_shares(self) -> None:
        """Agents trade shares of firm A."""
        self._compute_share_prices()
        self._compute_trade_limits()

        

    def step(self) -> None:
        """Perform one time step of the economy."""
        self._buy_firmA()
        self._pay_dividends()
        if self.t != 0:
            self._trade_shares()
        self.t += 1

    def run(self, n_steps: int = 1000) -> None:
        """Run the economy for a specified number of steps."""
        for _ in range(n_steps):
            self.step()