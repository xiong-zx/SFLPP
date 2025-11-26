"""
Data structures for SFLPP configurations and instances.

1. Config: high-level parameters for instance generation
2. Instance: full problem instance at the distribution level
3. ScenarioData: realized second-stage parameters for a single scenario
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import json
import pickle
import numpy as np


@dataclass
class Config:
    """
    High-level configuration: instance size + distributional parameters.
    """

    n_customers: int = 10
    n_facilities: int = 5

    # linear demand calibration: h_i(p) = -a_i p + b_i
    p_min: float = 20.0
    p_max: float = 80.0
    gamma: float | List[float] = 0.3  # demand at p_max is gamma * base_demand

    # base demand range
    base_demand_min: int = 50
    base_demand_max: int = 150

    # fixed cost range
    f_min: float = 500.0
    f_max: float = 1000.0

    # transport cost range
    c_min: float = 10.0
    c_max: float = 50.0

    # tariff distribution (for default recourse distribution)
    tau_max: float = 0.4  # proportional max on c_ij

    # capacity distribution parameters
    base_capacity_factor: float = 1.2  # ~ factor * total_base_demand / n_facilities
    cap_fluctuation: float = 0.2  # +/- 20% per scenario

    # random seed (for deterministic generation of Instance)
    seed: Optional[int] = 0

    # ---- IO helpers ----
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return cls(**d)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)


@dataclass
class Instance:
    """
    Full problem instance at the *distribution level*:

    - First-stage parameters (I, J, f, c, a, b, base_demand)
    - Distributional information for second-stage (tau_max, base_capacity, etc.)

    No scenario enumeration here.
    """

    config: Config

    I: List[int]  # customers
    J: List[int]  # facilities

    # first-stage parameters
    f: Dict[int, float]  # fixed open cost f_j
    c: Dict[Tuple[int, int], float]  # base transport cost c_ij
    a: Dict[int, float]  # demand slope a_i > 0
    b: Dict[int, float]  # demand intercept b_i
    base_demand: Dict[int, float]  # base_demand[i]

    # second-stage distribution info
    tau_max: float  # tariff proportional max
    base_capacity_factor: float  # used to compute base_capacity
    cap_fluctuation: float  # scenario capacity fluctuations
    base_capacity: Dict[int, float]  # base capacity for each facility j

    # --- Optional visualization data ---
    customer_coords: Optional[Dict[int, Tuple[float, float]]] = None
    facility_coords: Optional[Dict[int, Tuple[float, float]]] = None

    # ---------- construction ----------

    @classmethod
    def from_config(cls, cfg: Config, use_distance_costs: bool = False) -> "Instance":
        """
        Sample first-stage parameters and distribution info from Config.
        If use_distance_costs is True, generate coordinates and distance-based costs.
        """
        rng = np.random.default_rng(cfg.seed)

        I = list(range(cfg.n_customers))
        J = list(range(cfg.n_facilities))

        # fixed costs
        f = {j: float(rng.uniform(cfg.f_min, cfg.f_max)) for j in J}

        if use_distance_costs:
            # coordinates for visualization (assuming a 100x100 grid)
            customer_coords = {i: (rng.uniform(0, 100), rng.uniform(0, 100)) for i in I}
            facility_coords = {j: (rng.uniform(0, 100), rng.uniform(0, 100)) for j in J}

            # base transport costs c_ij proportional to distance
            c = {}
            unit_cost_per_distance = 0.3  # An example factor
            for i in I:
                for j in J:
                    cust_coord = customer_coords[i]
                    fac_coord = facility_coords[j]
                    distance = (
                        (cust_coord[0] - fac_coord[0]) ** 2
                        + (cust_coord[1] - fac_coord[1]) ** 2
                    ) ** 0.5
                    # Base cost is distance-based, plus a small random factor
                    c[(i, j)] = float(
                        distance * unit_cost_per_distance * rng.uniform(0.9, 1.1)
                    )
        else:
            customer_coords = None
            facility_coords = None
            # Original method: random cost independent of distance
            c = {(i, j): float(rng.uniform(cfg.c_min, cfg.c_max)) for i in I for j in J}

        # base demands & linear demand parameters
        base_demand = {
            i: float(rng.integers(cfg.base_demand_min, cfg.base_demand_max + 1))
            for i in I
        }

        p_min, p_max, gamma = cfg.p_min, cfg.p_max, cfg.gamma
        if p_max <= p_min:
            raise ValueError("p_max must be greater than p_min.")
        if isinstance(gamma, float):
            gamma = np.full(cfg.n_customers, gamma)
        else:
            gamma = np.array(gamma)
        if not (0.0 <= gamma).all() or not (gamma < 1.0).all():
            raise ValueError("gamma should be in [0,1).")

        a: Dict[int, float] = {}
        b: Dict[int, float] = {}

        for i, d0 in base_demand.items():
            d0 = float(d0)
            a_i = (1.0 - gamma[i]) * d0 / (p_max - p_min)
            b_i = d0 + a_i * p_min
            a[i] = a_i
            b[i] = b_i

        # base capacities (common for all scenarios, used by sampling)
        total_demand = sum(base_demand.values())
        avg_cap = cfg.base_capacity_factor * total_demand / len(J)
        cap_lb, cap_ub = int((1 - cfg.cap_fluctuation) * avg_cap), int(
            (1 + cfg.cap_fluctuation) * avg_cap
        )
        base_capacity = {j: float(rng.integers(cap_lb, cap_ub + 1)) for j in J}

        return cls(
            config=cfg,
            I=I,
            J=J,
            f=f,
            c=c,
            a=a,
            b=b,
            base_demand=base_demand,
            tau_max=cfg.tau_max,
            base_capacity_factor=cfg.base_capacity_factor,
            cap_fluctuation=cfg.cap_fluctuation,
            base_capacity=base_capacity,
            customer_coords=customer_coords,
            facility_coords=facility_coords,
        )

    # ---------- IO helpers ----------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a pure-JSON-serializable dict.
        (Tuple keys are converted to 'i,j' strings.)
        """
        d: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "I": self.I,
            "J": self.J,
            "f": self.f,
            "a": self.a,
            "b": self.b,
            "base_demand": self.base_demand,
            "tau_max": self.tau_max,
            "base_capacity_factor": self.base_capacity_factor,
            "cap_fluctuation": self.cap_fluctuation,
            "base_capacity": self.base_capacity,
            "customer_coords": self.customer_coords,
            "facility_coords": self.facility_coords,
        }

        # serialize c with string keys "i,j"
        d["c"] = {f"{i},{j}": val for (i, j), val in self.c.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Instance":
        cfg = Config.from_dict(d["config"])
        I = list(d["I"])
        J = list(d["J"])

        f = {int(k): float(v) for k, v in d["f"].items()}
        a = {int(k): float(v) for k, v in d["a"].items()}
        b = {int(k): float(v) for k, v in d["b"].items()}
        base_demand = {int(k): float(v) for k, v in d["base_demand"].items()}
        tau_max = float(d["tau_max"])
        base_capacity_factor = float(d["base_capacity_factor"])
        cap_fluctuation = float(d["cap_fluctuation"])
        base_capacity = {int(k): float(v) for k, v in d["base_capacity"].items()}

        # Deserialize coordinates if they exist
        customer_coords_raw = d.get("customer_coords")
        customer_coords = (
            {int(k): tuple(v) for k, v in customer_coords_raw.items()}
            if customer_coords_raw
            else None
        )
        facility_coords_raw = d.get("facility_coords")
        facility_coords = (
            {int(k): tuple(v) for k, v in facility_coords_raw.items()}
            if facility_coords_raw
            else None
        )
        # Handle older JSONs that might have string keys for coords
        if customer_coords and isinstance(list(customer_coords.keys())[0], str):
            customer_coords = {int(k): tuple(v) for k, v in customer_coords.items()}

        # deserialize c from "i,j"
        c_raw = d["c"]
        c: Dict[Tuple[int, int], float] = {}
        for key, val in c_raw.items():
            i_str, j_str = key.split(",")
            c[(int(i_str), int(j_str))] = float(val)

        return cls(
            config=cfg,
            I=I,
            J=J,
            f=f,
            c=c,
            a=a,
            b=b,
            base_demand=base_demand,
            tau_max=tau_max,
            base_capacity_factor=base_capacity_factor,
            cap_fluctuation=cap_fluctuation,
            base_capacity=base_capacity,
            customer_coords=customer_coords,
            facility_coords=facility_coords,
        )

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "Instance":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)


@dataclass
class ScenarioData:
    """
    Realized second-stage parameters for a single scenario ω.
    """

    bar_c: Dict[Tuple[int, int], float]  # combined cost \bar c_ij(ω)
    u: Dict[int, float]  # capacity u_j(ω)
    weight: float  # scenario weight (sum=1 for expectation)


# ---- generic pickle helpers (useful for large EF objects) ----
def save_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
