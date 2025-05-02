import numpy as np
import matplotlib.pyplot as plt


class Agent:
    """Represents an economic agent with a single attribute: wealth."""
    __slots__ = ("wealth",) # to save memory when many agents
    # maybe some day add risk aversion

    def __init__(self, wealth: float):
        self.wealth = float(wealth)


class TransactionRule:
    """Abstract base class for a rule that decides the outcome of transactions."""

    def outcome(self, w_i: float, w_j: float) -> int:
        """Return +1 if i wins, -1 if j wins."""
        raise NotImplementedError


class FairCoinRule(TransactionRule):
    """Both agents win with probability 1/2 irrespective of their wealth."""

    def outcome(self, *_):
        return 1 if np.random.rand() < 0.5 else -1


class WAARule(TransactionRule):
    """Wealth‑Attained Advantage (WAA) rule.

    The richer agent wins with probability:
        p_rich = 0.5 + zeta * Δw / (2 * (w_i + w_j)),  clipped to [0,1].
    Here zeta represents zeta/sqrt(gamma), following standard notation.
    """

    def __init__(self, zeta: float):
        self.zeta = float(zeta)

    def outcome(self, w_i: float, w_j: float):
        if w_i == w_j:
            return 1 if np.random.rand() < 0.5 else -1

        rich_is_i = w_i > w_j
        delta = abs(w_i - w_j)
        p_rich = 0.5 + self.zeta * delta / (2 * (w_i + w_j))
        p_rich = np.clip(p_rich, 0.0, 1.0)

        if np.random.rand() < p_rich:
            return 1 if rich_is_i else -1
        else:
            return -1 if rich_is_i else 1


class Taxation:
    """Pair‑wise flat redistribution (rate chi).
        The mean_wealth value is stored to avoid 
        computing it repeatedly"""

    def __init__(self, chi: float, mean_wealth: float):
        self.chi = float(chi)
        self.mean_wealth = mean_wealth

    def apply(self, wealth: float):
        return self.chi * (self.mean_wealth - wealth)


class YardSaleModel:
    """Implements baseline YSM and the Extended YSM (EYSM).
       For simplicity, we choose 
       dt = 1, f = sqrt(gamma * dt), zeta = zeta/sqrt(gamma) """

    def __init__(
        self,
        n_agents: int = 10_000,
        initial_wealth=1.0, # Admits scalar or array of initial wealths
        f: float = 0.1,
        transaction_rule: TransactionRule | None = None,
        taxation: Taxation | None = None,
        omega=0.0, # wealth tax rate, before any trade, agent loses fraction omega of their wealth
        tau=0.0, # Tobin tax rate per trade, when dw moves from loser to winner the winner keeps only (1-tau) * dw and the rest goes to the treasury
    ):
        self.n_agents = n_agents
        self.f = float(f)
        self.treasury = 0.0

        if isinstance(initial_wealth, (float, int)):
            self.agents = np.full(n_agents, float(initial_wealth))
        else:
            w = np.asarray(initial_wealth, dtype=float)
            if w.size != n_agents:
                raise ValueError("initial_wealth size mismatch.")
            self.agents = w.copy()

        self.transaction_rule = transaction_rule or FairCoinRule()
        self.tax = taxation

    # --- Diagnostics -----------------------------------------------------
    def gini(self) -> float:
        w = np.sort(self.agents)
        n = w.size
        cum = np.cumsum(w)
        return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

    def wealth_distribution(self):
        return self.agents.copy()

    # --- Core dynamics ----------------------------------------------------
    def _exchange(self, i, j):
        w_i, w_j = self.agents[i], self.agents[j]
        poorer = min(w_i, w_j)
        dw = self.f * poorer

        sgn = self.rule.outcome(w_i, w_j)

        self.agents[i] += sgn * dw
        self.agents[j] -= sgn * dw

    def _redistribute(self, i: int, j: int):
        if self.tax is None or self.tax.chi == 0.0:
            return
        mean_w = self.agents.mean()
        self.agents[i] += self.tax.apply(self.agents[i])
        self.agents[j] += self.tax.apply(self.agents[j])

    def _apply_wealth_tax(self):
        if self.omega == 0.0:
            return
        tax = self.omega * self.agents
        self.agents -= tax
        self.treasury += tax.sum()


    def _apply_wealth_tax(self):
        if self.omega == 0.0:
            return
        tax = self.omega * self.agents
        self.agents -= tax
        self.treasury += tax.sum()


    def step(self, n_pairs: int = None):
        n_pairs = n_pairs or self.n_agents
        for _ in range(n_pairs):
            i, j = np.random.randint(0, self.n_agents, 2)
            if i == j:
                continue
            self._exchange(i, j)
            self._redistribute(i, j)

    def run(self, n_steps: int = 1000, record_interval: int = 100):
        history = []
        for t in range(1, n_steps + 1):
            self.step()
            if t % record_interval == 0:
                history.append((t, self.gini()))
        return np.array(history)


# quantes steps perque canvii distribucio, fer gif
# taxacio 0, 5, 10, 20, 40, 80
# iva? wealth tax
# iva, aduanas, 
# intrinsecament liquid
# graduacions liquiditat
# herencia? 
# tipus interes mes baix que pib o no?
# tipus interes efectiu mes alt que el pib 
# wealth tax io toring tax
# liquiditat absoluta perden inflacio, bons, accions, sense derivats 
# islm keynes diners i 1 be financer, 2 bens gradients liquidat 