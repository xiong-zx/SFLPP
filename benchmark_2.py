"""
Benchmark_2: Compare Continuous Price MIQP vs. Discrete Price MILP.

Loads a pre-generated instance and its sampled extensive form scenarios, then:
  1) Solves the extensive form MIQP with continuous prices.
  2) Solves the discrete-price MILP where prices are discretized into levels.

Results are printed and optionally saved to results/ as CSV and JSON.
"""

# %%
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt

from core.data import Instance
from core.extensive_form import ExtensiveForm, build_extensive_form_model
from core.discrete_price_milp import build_discrete_price_milp_model
from core.solver import solve_gurobi_model
from core.progressive_hedging import solve_with_ph, evaluate_first_stage_solution

# Import utilities from the new utils module
from core.utils import (
    setup_directories,
    instance_file_path,
    ef_file_path,
    load_gurobi_params,
    apply_gurobi_defaults,
    save_benchmark_results,
    print_section_header,
    print_subsection_header,
)

# Setup directories
DIRS = setup_directories(use_dist_version=True, create=True)
ROOT = DIRS["root"]
CONFIG_DIR = DIRS["config"]
DATA_DIR = DIRS["data"]


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass
class Benchmark2Settings:
    config_name: str = "c5_f5_cf1"
    instance_list: Optional[List[int]] = None
    scenarios_list: Optional[List[int]] = None
    alpha: float = 0.1
    price_levels: int = 10  # Number of discrete price levels for MILP
    price_levels_list: Optional[List[int]] = None
    save_results: bool = True
    run_ph: bool = True
    ph_rho: float = 5000.0
    ph_max_iter: int = 100
    ph_tol: float = 1e-4

    def __post_init__(self):
        if self.instance_list is None:
            self.instance_list = [1]
        if self.scenarios_list is None:
            self.scenarios_list = [10, 20, 50]
        if self.price_levels_list is None:
            self.price_levels_list = [self.price_levels]


# --------------------------------------------------------------------------- #
# Solution methods
# --------------------------------------------------------------------------- #
def run_extensive_form(
    ext_form: ExtensiveForm,
    alpha: float,
    verbose: int = 0,
    gurobi_params: Dict[str, float | str] | None = None,
) -> Dict:
    """
    Solve the extensive form MIQP with continuous prices.
    """
    n_scenarios = len(ext_form.scenarios)
    start = time.time()
    model, vars_dict = build_extensive_form_model(ext_form, alpha=alpha)
    params = dict(gurobi_params or {})
    params.setdefault("OutputFlag", verbose)
    solve_gurobi_model(model, params=params)
    elapsed = time.time() - start

    status = model.Status
    result: Dict = {
        "method": "Continuous Price MIQP",
        "status": status,
        "objective": None,
        "time": elapsed,
        "n_scenarios": n_scenarios,
    }
    if status == gp.GRB.OPTIMAL:
        result["objective"] = model.ObjVal
        result["n_facilities"] = sum(1 for j in ext_form.J if vars_dict["x"][j].X > 0.5)

        # Print x and p values
        print("\n--- Continuous Price MIQP Solution ---")
        print("x (facility locations):")
        for j in ext_form.J:
            if vars_dict["x"][j].X > 0.5:
                print(f"  Facility {j}: OPEN (x={vars_dict['x'][j].X:.4f})")

        if "p" in vars_dict:
            print("\np (prices per customer):")
            for i in ext_form.I:
                print(f"  Customer {i}: p={vars_dict['p'][i].X:.4f}")

    return result


def run_discrete_price_milp(
    ext_form: ExtensiveForm,
    alpha: float,
    price_levels: int,
    verbose: int = 0,
    gurobi_params: Dict[str, float | str] | None = None,
) -> Dict:
    """
    Solve the discrete price MILP where prices are discretized into levels.
    """
    n_scenarios = len(ext_form.scenarios)
    start = time.time()
    model, vars_dict = build_discrete_price_milp_model(
        ext_form, alpha=alpha, price_levels=price_levels
    )
    params = dict(gurobi_params or {})
    params.setdefault("OutputFlag", verbose)
    solve_gurobi_model(model, params=params)
    elapsed = time.time() - start

    status = model.Status
    result: Dict = {
        "method": "Discrete Price MILP",
        "status": status,
        "objective": None,
        "time": elapsed,
        "n_scenarios": n_scenarios,
        "price_levels": price_levels,
    }
    if status == gp.GRB.OPTIMAL:
        result["objective"] = model.ObjVal
        result["n_facilities"] = sum(1 for j in ext_form.J if vars_dict["x"][j].X > 0.5)

        # Print x and p values
        print("\n--- Discrete Price MILP Solution ---")
        print("x (facility locations):")
        for j in ext_form.J:
            if vars_dict["x"][j].X > 0.5:
                print(f"  Facility {j}: OPEN (x={vars_dict['x'][j].X:.4f})")

        if "p" in vars_dict:
            print("\np (prices per customer):")
            for i in ext_form.I:
                price_val = vars_dict["p"][i].X
                # Find which discrete level was selected
                if "z" in vars_dict:
                    selected_level = None
                    for l in range(price_levels):
                        if vars_dict["z"][i, l].X > 0.5:
                            selected_level = l
                            break
                    print(f"  Customer {i}: p={price_val:.4f} (level {selected_level})")
                else:
                    print(f"  Customer {i}: p={price_val:.4f}")

    return result


def run_progressive_hedging(
    ext_form: ExtensiveForm,
    alpha: float,
    rho: float,
    max_iter: int,
    tol: float,
    verbose: int = 0,
    gurobi_params: Dict[str, float | str] | None = None,
) -> Dict:
    """
    Solve the first-stage price model with Progressive Hedging (heuristic).
    """
    n_scenarios = len(ext_form.scenarios)
    start = time.time()
    ph_res = solve_with_ph(
        ext_form=ext_form,
        rho=rho,
        max_iter=max_iter,
        alpha=alpha,
        tol=tol,
        gurobi_params=gurobi_params,
    )
    elapsed = time.time() - start

    final_x = ph_res["final_x"]
    final_p = ph_res["final_p"]
    objective = evaluate_first_stage_solution(
        final_x, final_p, ext_form, alpha=alpha, gurobi_params=gurobi_params
    )

    result: Dict = {
        "method": "Progressive Hedging",
        "status": "converged" if ph_res["converged"] else "max_iter",
        "objective": objective,
        "time": elapsed,
        "n_scenarios": n_scenarios,
        "rho": rho,
        "iterations": ph_res["iterations"],
        "n_facilities": sum(1 for j in ext_form.J if final_x[j] > 0.5),
    }
    return result


# --------------------------------------------------------------------------- #
# Plotting functions
# --------------------------------------------------------------------------- #
def plot_gap_vs_price_levels(df: pd.DataFrame, settings: Benchmark2Settings) -> None:
    """
    Plot gap vs number of discrete pricing levels for different scenarios.
    """
    # Filter only discrete price results with valid gap
    discrete_df = df[
        (df["method"] == "Discrete Price MILP") & (df["gap_vs_continuous"].notna())
    ].copy()

    if discrete_df.empty:
        print("No discrete price results with gap data to plot.")
        return

    scenarios_list = sorted(discrete_df["n_scenarios"].unique())

    plt.figure(figsize=(10, 6))

    # Use different colors and markers for each scenario count
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios_list)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    for idx, n_scen in enumerate(scenarios_list):
        scen_df = discrete_df[discrete_df["n_scenarios"] == n_scen]
        scen_df = scen_df.sort_values("price_levels")

        plt.plot(
            scen_df["price_levels"],
            scen_df["gap_vs_continuous"] * 100,  # Convert to percentage
            marker=markers[idx % len(markers)],
            color=colors[idx],
            label=f"S={n_scen}",
            linewidth=2,
            markersize=8,
        )

    plt.xlabel("Number of Discrete Price Levels", fontsize=12)
    plt.ylabel("Gap vs Continuous MIQP (%)", fontsize=12)
    plt.title(
        "Optimality Gap vs Number of Discrete Price Levels",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(title="Scenarios", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    Path("results").mkdir(exist_ok=True)
    scenarios_str = "_".join(map(str, settings.scenarios_list))
    price_str = "_".join(map(str, settings.price_levels_list))
    inst_str = "_".join(map(str, settings.instance_list))
    plot_path = (
        Path("results")
        / f"{settings.config_name}_ins{inst_str}_gap_vs_price_levels.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nGap plot saved to: {plot_path}")
    plt.close()


def plot_gap_vs_scenarios(df: pd.DataFrame, settings: Benchmark2Settings) -> None:
    """
    Plot optimality gap vs number of scenarios for discrete MILP and PH.
    """
    continuous_df = df[df["method"] == "Continuous Price MIQP"]
    if continuous_df.empty:
        print("No continuous MIQP results; cannot compute gaps.")
        return

    # Only keep rows where gap_vs_continuous is available
    gap_df = df[df["gap_vs_continuous"].notna()].copy()
    if gap_df.empty:
        print("No gap data to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Discrete MILP: group by price_levels
    discrete_df = gap_df[gap_df["method"] == "Discrete Price MILP"]
    if not discrete_df.empty:
        price_levels_list = sorted(discrete_df["price_levels"].unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(price_levels_list)))
        for idx, p_level in enumerate(price_levels_list):
            p_df = discrete_df[discrete_df["price_levels"] == p_level]
            p_df = p_df.sort_values("n_scenarios")
            plt.plot(
                p_df["n_scenarios"],
                p_df["gap_vs_continuous"] * 100,
                marker="o",
                linewidth=2,
                markersize=7,
                label=f"Discrete MILP p={p_level}",
                color=colors[idx],
            )

    # PH: single series
    ph_df = gap_df[gap_df["method"] == "Progressive Hedging"]
    if not ph_df.empty:
        ph_df = ph_df.sort_values("n_scenarios")
        plt.plot(
            ph_df["n_scenarios"],
            ph_df["gap_vs_continuous"] * 100,
            marker="x",
            linewidth=2.5,
            markersize=9,
            label="Progressive Hedging",
            color="red",
            linestyle="--",
        )

    plt.xlabel("Number of Scenarios", fontsize=12)
    plt.ylabel("Gap vs Continuous MIQP (%)", fontsize=12)
    plt.title("Optimality Gap vs Scenarios", fontsize=14, fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)
    scenarios_str = "_".join(map(str, settings.scenarios_list))
    price_str = "_".join(map(str, settings.price_levels_list))
    inst_str = "_".join(map(str, settings.instance_list))
    plot_path = (
        Path("results") / f"{settings.config_name}_ins{inst_str}_gap_vs_scenarios.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Gap-vs-scenarios plot saved to: {plot_path}")
    plt.close()


def plot_time_vs_scenarios(df: pd.DataFrame, settings: Benchmark2Settings) -> None:
    """
    Plot time complexity vs number of scenarios for different methods.
    """
    if df.empty:
        print("No results to plot.")
        return

    scenarios_list = sorted(df["n_scenarios"].unique())

    # Separate continuous and discrete methods
    continuous_df = df[df["method"] == "Continuous Price MIQP"].copy()
    discrete_df = df[df["method"] == "Discrete Price MILP"].copy()
    ph_df = df[df["method"] == "Progressive Hedging"].copy()

    plt.figure(figsize=(12, 7))

    # Plot continuous price MIQP
    if not continuous_df.empty:
        continuous_df = continuous_df.sort_values("n_scenarios")
        plt.plot(
            continuous_df["n_scenarios"],
            continuous_df["time"],
            marker="o",
            linewidth=2.5,
            markersize=10,
            label="Extensive Form (Continuous Price MIQP)",
            color="black",
            linestyle="-",
        )

    # Plot discrete price MILP for different price levels
    if not discrete_df.empty:
        price_levels_list = sorted(discrete_df["price_levels"].unique())

        # Define colors for different price levels
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(price_levels_list)))
        markers = ["s", "^", "D", "v", "p", "*", "h", "<", ">"]
        linestyles = ["-", "--", "-.", ":"]

        for idx, p_level in enumerate(price_levels_list):
            p_df = discrete_df[discrete_df["price_levels"] == p_level]
            p_df = p_df.sort_values("n_scenarios")

            # Categorize price levels as low, medium, high
            if p_level <= 5:
                category = "Low"
            elif p_level <= 10:
                category = "Medium"
            else:
                category = "High"

            plt.plot(
                p_df["n_scenarios"],
                p_df["time"],
                marker=markers[idx % len(markers)],
                linewidth=2,
                markersize=8,
                label=f"Discrete MILP (p={p_level}, {category})",
                color=colors[idx],
                linestyle=linestyles[idx % len(linestyles)],
            )

    # Plot PH
    if not ph_df.empty:
        ph_df = ph_df.sort_values("n_scenarios")
        plt.plot(
            ph_df["n_scenarios"],
            ph_df["time"],
            marker="x",
            linewidth=2.5,
            markersize=9,
            label="Progressive Hedging",
            color="red",
            linestyle="--",
        )

    plt.xlabel("Number of Scenarios", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title(
        "Computational Time vs Number of Scenarios", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=9, loc="best")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # Use log scale for better visualization
    plt.tight_layout()

    # Save plot
    Path("results").mkdir(exist_ok=True)
    scenarios_str = "_".join(map(str, settings.scenarios_list))
    price_str = "_".join(map(str, settings.price_levels_list))
    inst_str = "_".join(map(str, settings.instance_list))
    plot_path = (
        Path("results") / f"{settings.config_name}_ins{inst_str}_time_vs_scenarios.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Time plot saved to: {plot_path}")
    plt.close()


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_benchmark(settings: Benchmark2Settings) -> pd.DataFrame:
    """
    Compare Continuous Price MIQP vs Discrete Price MILP across instances and scenario counts.
    """
    config_name = settings.config_name
    instance_list = settings.instance_list
    scenarios_list = settings.scenarios_list
    price_levels_list = settings.price_levels_list

    gurobi_params = load_gurobi_params(CONFIG_DIR / "gurobi_params.json")
    apply_gurobi_defaults(gurobi_params)

    all_results: List[Dict] = []

    for inst_idx in instance_list:
        inst_path = instance_file_path(config_name, inst_idx, DATA_DIR)
        if not inst_path.exists():
            raise FileNotFoundError(f"Instance file not found: {inst_path}")

        print_section_header(
            f"BENCHMARK_2 | config={config_name} | instance={inst_idx}\n"
            f"Scenario counts: {scenarios_list} | alpha={settings.alpha} | price_levels={price_levels_list}"
        )

        # Load instance (for info only)
        inst = Instance.load_json(str(inst_path))
        print(f"Instance: {len(inst.I)} customers, {len(inst.J)} facilities")

        for n_scen in scenarios_list:
            ef_path = ef_file_path(config_name, inst_idx, n_scen, DATA_DIR)
            if not ef_path.exists():
                raise FileNotFoundError(f"Extensive form file not found: {ef_path}")
            ext_form = ExtensiveForm.load_pkl(str(ef_path))

            print_subsection_header(f"Scenarios = {n_scen}")

            # 1) Continuous price MIQP baseline
            ef_res = run_extensive_form(
                ext_form,
                alpha=settings.alpha,
                verbose=0,
                gurobi_params=gurobi_params,
            )
            ef_res["instance"] = inst_idx
            ef_res["config"] = config_name
            all_results.append(ef_res)
            print(
                f"[Continuous Price MIQP] status={ef_res['status']} obj={ef_res['objective']} time={ef_res['time']:.2f}s"
            )

            # 2) Discrete price MILP (sweep price levels)
            for p_levels in price_levels_list:
                discrete_res = run_discrete_price_milp(
                    ext_form,
                    alpha=settings.alpha,
                    price_levels=p_levels,
                    verbose=0,
                    gurobi_params=gurobi_params,
                )
                discrete_res["instance"] = inst_idx
                discrete_res["config"] = config_name

                if (
                    ef_res.get("objective") is not None
                    and discrete_res.get("objective") is not None
                ):
                    continuous_obj = ef_res["objective"]
                    discrete_obj = discrete_res["objective"]
                    discrete_res["gap_vs_continuous"] = (
                        (discrete_obj - continuous_obj) / abs(continuous_obj)
                        if abs(continuous_obj) > 1e-8
                        else None
                    )

                all_results.append(discrete_res)
                print(
                    f"[Discrete Price MILP ] levels={p_levels} status={discrete_res['status']} "
                    f"obj={discrete_res['objective']} time={discrete_res['time']:.2f}s "
                    f"gap_vs_continuous={discrete_res.get('gap_vs_continuous')}"
                )

            # 3) Progressive Hedging (optional)
            if settings.run_ph:
                ph_res = run_progressive_hedging(
                    ext_form,
                    alpha=settings.alpha,
                    rho=settings.ph_rho,
                    max_iter=settings.ph_max_iter,
                    tol=settings.ph_tol,
                    verbose=0,
                    gurobi_params=gurobi_params,
                )
                ph_res["instance"] = inst_idx
                ph_res["config"] = config_name

                if (
                    ef_res.get("objective") is not None
                    and ph_res.get("objective") is not None
                ):
                    continuous_obj = ef_res["objective"]
                    ph_obj = ph_res["objective"]
                    ph_res["gap_vs_continuous"] = (
                        (ph_obj - continuous_obj) / abs(continuous_obj)
                        if abs(continuous_obj) > 1e-8
                        else None
                    )

                all_results.append(ph_res)
                print(
                    f"[Progressive Hedging ] status={ph_res['status']} obj={ph_res['objective']} "
                    f"time={ph_res['time']:.2f}s gap_vs_continuous={ph_res.get('gap_vs_continuous')}"
                )

    df = pd.DataFrame(all_results)

    # Summary and save
    print_section_header("BENCHMARK_2 SUMMARY")
    print(df.to_string(index=False))

    if settings.save_results and len(all_results) > 0:
        save_benchmark_results(
            results=all_results,
            config_name=config_name,
            instance_list=instance_list,
            scenarios_list=scenarios_list,
            results_dir=DIRS["results"],
            suffix="bench2",
            price_levels_list=price_levels_list,
        )

    # Generate plots
    if len(all_results) > 0:
        print_section_header("GENERATING PLOTS")
        plot_gap_vs_price_levels(df, settings)
        plot_time_vs_scenarios(df, settings)
        plot_gap_vs_scenarios(df, settings)

    return df


# %%
if __name__ == "__main__":
    SETTINGS = Benchmark2Settings(
        config_name="c50_f20_cf6",
        instance_list=[1],
        scenarios_list=[10, 20, 50, 100, 200],
        price_levels_list=[3, 5, 8, 10, 15],
        alpha=0.1,
        save_results=True,
        run_ph=True,
        ph_rho=5000.0,
        ph_max_iter=100,
        ph_tol=1e-4,
    )
    run_benchmark(SETTINGS)
# %%
