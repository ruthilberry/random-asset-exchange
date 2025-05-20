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
        initial_stocksA: Union[float, np.ndarray] = 1.0,
        savings_rates: Union[float, np.ndarray] = 0.99,
    ):
        self.n_individuals = n_individuals
        
        # Handle scalar or array inputs for each attribute
        self.money = self._initialize_array(initial_money, "initial_money")
        self.stocksA = self._initialize_array(initial_stocksA, "initial_stocksA")
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
        initial_stocksA: Union[float, np.ndarray] = 1.0,
        savings_rates: Union[float, np.ndarray] = 0.99,
        dt: float = 1.0,  # Time step size for future continuous-time implementation
    ):
        self.population = Population(
            n_individuals=n_individuals,
            initial_money=initial_money,
            initial_stocksA=initial_stocksA,
            savings_rates=savings_rates
        )
        self.total_money = self.population.money.sum()
        self.total_stocksA = self.population.stocksA.sum()
        self.treasury = 0.0
        self.firmA_wealth = 0.0
        self.t = 0
        self.dt = dt
    
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
        self.population.money += dividends
        self.firmA_wealth = 0.0  # Reset firm wealth after paying dividends

    def step(self) -> None:
        """Perform one time step of the economy."""
        self._buy_firmA()
        self._pay_dividends()
        self.t += 1

    def run(self, n_steps: int = 1000) -> None:
        """Run the economy for a specified number of steps."""
        for _ in range(n_steps):
            self.step()