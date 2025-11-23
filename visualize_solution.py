"""
Visualize the solution of a facility location problem instance.

This script reads an instance file (for coordinates) and a results file
(for decisions), and generates a network graph showing:
  - Customer and facility locations
  - Opened vs. closed facilities
  - Allocation flows from facilities to customers
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from core.data import Instance
from core.extensive_form import ExtensiveForm
from adjustText import adjust_text
from typing import List, Dict, Tuple, Any
from collections import defaultdict

def get_second_stage_flows_for_scenarios(
    x_solution: Dict[int, float],
    ext_form: ExtensiveForm,
    scenarios_to_solve: List[int],
    alpha: float = 0.1,
) -> Dict[int, Dict[str, Any]]:
    """
    Given a fixed first-stage solution x, solve the second-stage problems
    for a specific list of scenarios to get the flow values q_ij^w.
    Returns a dictionary mapping scenario index to a dict of its results (flows, prices, served_demand).
    """
    inst = ext_form.instance
    I, J = inst.I, inst.J
    cfg = inst.config

    m = gp.Model("evaluation_flows")
    m.setParam("OutputFlag", 0)

    q = m.addVars(I, J, scenarios_to_solve, lb=0.0, name="q")
    s = m.addVars(I, scenarios_to_solve, lb=0.0, name="s")
    p = m.addVars(I, scenarios_to_solve, lb=cfg.p_min, ub=cfg.p_max, name="p")

    for w in scenarios_to_solve:
        scen = ext_form.scenarios[w]
        for i in I:
            m.addConstr(s[i, w] == gp.quicksum(q[i, j, w] for j in J))
        for j in J:
            m.addConstr(gp.quicksum(q[i, j, w] for i in I) <= scen.u[j] * x_solution[j])
        for i in I:
            demand_at_price = -inst.a[i] * p[i, w] + inst.b[i]
            m.addConstr(s[i, w] <= demand_at_price)
            m.addConstr(s[i, w] >= (1.0 - alpha) * demand_at_price)

    # Objective is irrelevant as we only need feasible flows, but we set one.
    m.setObjective(0, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError("Flow evaluation model failed to solve.")

    # Extract flows for each requested scenario
    scenario_results = defaultdict(dict)
    for w in scenarios_to_solve:
        flows = {}
        prices = {}
        served_demand = {}
        for i in I:
            prices[i] = p[i, w].X
            served_demand[i] = s[i, w].X
            for j in J:
                flows[(i, j)] = q[i, j, w].X
        scenario_results[w] = {"flows": flows, "prices": prices, "served_demand": served_demand}
    return scenario_results


def visualize_network(instance: Instance, solution: dict, title: str, save_path: Path | None = None):
    """
    Generates and displays a network visualization.
    """
    if not instance.customer_coords or not instance.facility_coords:
        print("Error: Instance file does not contain coordinate information.")
        return

    # --- Prepare Graph and Node Positions ---
    G = nx.Graph()
    pos = {}
    node_labels = {}
    customer_nodes = []
    facility_nodes = []

    # Add customer nodes
    for i, coords in instance.customer_coords.items():
        node_id = f"C{i}"
        customer_nodes.append(node_id)
        G.add_node(node_id, type="customer")
        pos[node_id] = coords
        node_labels[node_id] = f"C{i}"

    # Add facility nodes
    opened_facilities = solution.get("opened_facilities", [])
    for j, coords in instance.facility_coords.items():
        node_id = f"F{j}"
        facility_nodes.append(node_id)
        is_open = j in opened_facilities
        G.add_node(node_id, type="facility", open=is_open)
        pos[node_id] = coords
        node_labels[node_id] = f"F{j}"

    # --- Add Edges based on Actual Flows ---
    flows = solution.get("flows", {})
    tariffs = solution.get("tariffs", {})
    edge_labels = {}

    # Separate edges into zero and non-zero flow lists for drawing
    non_zero_edges = []
    zero_flow_edges = []

    if not flows:
        print("Warning: No flow information provided in the solution.")
    
    # Get all non-zero flows to determine quartiles
    non_zero_flows = [val for val in flows.values() if val > 1e-4]
    
    # Define styles for flow edges
    flow_styles = {
        "Low (0-25%)":    {"color": "lightblue", "width": 1.0},
        "Medium (25-50%)": {"color": "deepskyblue", "width": 2.0},
        "High (50-75%)":   {"color": "royalblue", "width": 3.0},
        "Very High (75-100%)": {"color": "navy", "width": 4.0},
    }
    quartiles = np.percentile(non_zero_flows, [25, 50, 75]) if non_zero_flows else [0,0,0]

    # Process all possible (customer, facility) pairs
    for i in instance.I:
        for j in instance.J:
            flow_val = flows.get((i, j), 0.0)
            c_node, f_node = f"C{i}", f"F{j}"
            
            # Only consider edges connected to OPEN facilities
            if j in opened_facilities:
                # Add tariff label for this potential edge
                edge_labels[(c_node, f_node)] = f"τ: {tariffs.get((i,j), 0):.1%}"

                if flow_val > 1e-4:
                    if flow_val <= quartiles[0]:
                        style = flow_styles["Low (0-25%)"]
                    elif flow_val <= quartiles[1]:
                        style = flow_styles["Medium (25-50%)"]
                    elif flow_val <= quartiles[2]:
                        style = flow_styles["High (50-75%)"]
                    else:
                        style = flow_styles["Very High (75-100%)"]
                    G.add_edge(c_node, f_node, color=style["color"], width=style["width"], style='solid')
                    non_zero_edges.append((c_node, f_node))
                else:
                    # Add edge for zero flow since the facility is open
                    G.add_edge(c_node, f_node, color='gray', width=0.8, style='dashed')
                    zero_flow_edges.append((c_node, f_node))

    plt.figure(figsize=(14, 12))

    # Draw nodes
    node_colors = ["skyblue" if G.nodes[n]['type'] == 'customer' else ('green' if G.nodes[n].get('open') else 'red') for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, edgecolors='black')

    # Draw non-zero flow edges (solid)
    if non_zero_edges:
        edge_colors = [G[u][v]['color'] for u, v in non_zero_edges]
        edge_widths = [G[u][v]['width'] for u, v in non_zero_edges]
        nx.draw_networkx_edges(G, pos, edgelist=non_zero_edges, edge_color=edge_colors, width=edge_widths, alpha=0.8, style='solid')

    # Draw zero-flow edges (dashed)
    if zero_flow_edges:
        nx.draw_networkx_edges(G, pos, edgelist=zero_flow_edges, edge_color='gray', width=0.8, alpha=0.8, style='dashed')

    # Draw edge labels (tariffs)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='darkred', label_pos=0.3,
                                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0))

    # Draw node labels (C_i, F_j)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)

    # --- Add price/demand labels with auto-adjustment ---
    # 1. Collect text objects for adjustText
    texts = []
    prices = solution.get("prices", {})
    served_demand = solution.get("served_demand", {})
    for i, p_val in prices.items():
        s_val = served_demand.get(i, 0)
        demand_at_price = max(0, -instance.a[i] * p_val + instance.b[i])
        coords = instance.customer_coords[i]
        # Initial position is at the node's coordinates
        texts.append(plt.text(coords[0], coords[1], f"p: {p_val:.1f}\nh(p): {demand_at_price:.1f}\ns: {s_val:.1f}", fontsize=8, color='purple', ha='center', va='center'))

    # 2. Let adjustText find optimal positions for the labels
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='grey', lw=0.8), bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.title(title, fontsize=16)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)

    # Add a custom legend
    legend_elements = [plt.Line2D([0], [0], color=style['color'], lw=style['width'], label=f"Flow: {label}")
                       for label, style in flow_styles.items()]
    legend_elements.append(plt.Line2D([0], [0], color='gray', lw=0.8, linestyle='--', label='Flow: Zero'))
    plt.legend(handles=legend_elements, loc='lower right', title="Transportation Volume")
    
    # Save the figure before showing it
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Plot saved to {save_path}")

    # Show the plot and then close the figure to free memory
    plt.show()
    plt.close()





# def visualize_network(instance: Instance, solution: dict, title: str):
#     """
#     Generates and displays a network visualization.
#     """
#     if not instance.customer_coords or not instance.facility_coords:
#         print("Error: Instance file does not contain coordinate information.")
#         return

#     # --- Prepare Graph and Node Positions ---
#     G = nx.Graph()
#     pos = {}
#     customer_nodes = []
#     facility_nodes = []

#     # Add customer nodes
#     for i, coords in instance.customer_coords.items():
#         node_id = f"C{i}"
#         customer_nodes.append(node_id)
#         G.add_node(node_id, type="customer")
#         pos[node_id] = coords

#     # Add facility nodes
#     opened_facilities = solution.get("opened_facilities", [])
#     for j, coords in instance.facility_coords.items():
#         node_id = f"F{j}"
#         facility_nodes.append(node_id)
#         is_open = j in opened_facilities
#         G.add_node(node_id, type="facility", open=is_open)
#         pos[node_id] = coords

#     # --- Add Edges based on Actual Flows ---
#     flows = solution.get("flows", {})
#     if not flows:
#         print("Warning: No flow information provided in the solution.")
    
#     # Get all non-zero flows to determine quartiles
#     non_zero_flows = [val for val in flows.values() if val > 1e-4]
    if non_zero_flows:
        quartiles = np.percentile(non_zero_flows, [25, 50, 75])
        
        flow_styles = {
            "Low (0-25%)":    {"color": "lightblue", "width": 1.0},
            "Medium (25-50%)": {"color": "deepskyblue", "width": 2.0},
            "High (50-75%)":   {"color": "royalblue", "width": 3.0},
            "Very High (75-100%)": {"color": "navy", "width": 4.0},
        }

#         for (i, j), flow_val in flows.items():
#             if flow_val > 1e-4:  # Only draw edges for non-trivial flows
#                 c_node = f"C{i}"
#                 f_node = f"F{j}"
                
#                 if flow_val <= quartiles[0]:
#                     style = flow_styles["Low (0-25%)"]
#                 elif flow_val <= quartiles[1]:
#                     style = flow_styles["Medium (25-50%)"]
#                 elif flow_val <= quartiles[2]:
#                     style = flow_styles["High (50-75%)"]
#                 else:
#                     style = flow_styles["Very High (75-100%)"]
                
#                 G.add_edge(c_node, f_node, color=style["color"], width=style["width"])


if __name__ == "__main__":
    # --- Global Switch ---
    # This should always be True for visualization, as it needs coordinates.
    USE_DIST_VERSION = True

    # Create a directory for saving plots
    plots_dir = Path("plots_dist") if USE_DIST_VERSION else Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # --- 1. Configuration for Finding the Results File ---

    # 1. Specify the list of configs that were run to generate the summary file.
    # This is used to construct the correct summary filename.
    # This should match the CONFIG_NAMES list in EF_runner.py or PH_runner.py
    CONFIG_NAMES_IN_SUMMARY: List[str] = [
        "c10_f5_cf2",
        "c10_f5_cf3",
        "c10_f5_cf4",
    ]

    # 2. Specify which specific experiments you want to visualize from that summary.
    CONFIGS_TO_VISUALIZE: List[str] = ["c10_f5_cf4"]
    INSTANCE_IDX_LIST: List[int] = [1]
    SCENARIOS_LIST: List[int] = [50] # The total number of scenarios in the experiment to look for

    # 3. Choose which summary file to read from ('ef' or 'ph').
    RUNNER_TYPE = "ef"  # "ef" for EF_runner, "ph" for PH_runner

    # Specify which scenarios within each experiment you want to plot.
    SCENARIOS_TO_PLOT: List[int] = [0] # e.g., plot scenario 0 and scenario 10


    # --- 3. Main Loop for Visualization ---

    # Construct the summary filename automatically
    prefix = "summary" if RUNNER_TYPE == "ef" else "ph_summary"
    config_str = "_".join(CONFIG_NAMES_IN_SUMMARY)[:50]
    summary_filename = f"{prefix}_{config_str}.json"

    results_dir = "results_dist" if USE_DIST_VERSION else "results"
    RESULTS_FILE = Path(results_dir) / summary_filename

    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

    with open(RESULTS_FILE, "r") as f:
        all_results = json.load(f)

    for config_name in CONFIGS_TO_VISUALIZE:
        for instance_idx in INSTANCE_IDX_LIST:
            for n_scenarios in SCENARIOS_LIST:
                print(f"\n--- Preparing visualization for experiment: {config_name}, ins={instance_idx}, s={n_scenarios} ---")

                # Find the main result entry for this experiment
                target_result = next((
                    res for res in all_results
                    if res['config'] == config_name
                    and res['instance_idx'] == instance_idx
                    and res['n_scenarios'] == n_scenarios
                ), None)
                
                if not target_result:
                    print(f"Warning: Could not find a matching result in {RESULTS_FILE}")
                    continue

                # Load the corresponding extensive form file which contains all scenario data
                data_dir = "data_dist" if USE_DIST_VERSION else "data"
                ef_path = Path(f"{data_dir}/{config_name}_ins{instance_idx}_s{n_scenarios}.pkl")
                if not ef_path.exists():
                    print(f"Warning: Extensive form file not found, skipping: {ef_path}")
                    continue
                ext_form = ExtensiveForm.load_pkl(str(ef_path))

                # Get the first-stage solution (x) from the results
                x_sol = {j: 1.0 if j in target_result["opened_facilities"] else 0.0 for j in ext_form.J}

                # Re-solve to get flows, prices, etc. for the specific scenarios we want to plot
                scenario_results = get_second_stage_flows_for_scenarios(x_sol, ext_form, SCENARIOS_TO_PLOT)

                # Generate a plot for each requested scenario
                for w in SCENARIOS_TO_PLOT:
                    # Calculate average tariff rate for this scenario
                    scen_data = ext_form.scenarios[w]
                    tariff_rates = [(scen_data.bar_c[(i,j)] - ext_form.instance.c[(i,j)]) / ext_form.instance.c[(i,j)]
                                    for i in ext_form.I for j in ext_form.J if ext_form.instance.c[(i,j)] > 1e-6]
                    avg_tariff_rate = float(np.mean(tariff_rates)) if tariff_rates else 0.0

                    # Calculate relative tariffs tau_ij for this scenario
                    relative_tariffs = {
                        (i, j): (scen_data.bar_c[(i,j)] - ext_form.instance.c[(i,j)]) / ext_form.instance.c[(i,j)]
                        if ext_form.instance.c.get((i,j), 0) > 1e-6 else 0.0
                        for i in ext_form.I for j in ext_form.J 
                    }

                    solution_details = {
                        "opened_facilities": target_result["opened_facilities"],
                        "flows": scenario_results.get(w, {}).get("flows", {}),
                        "prices": scenario_results.get(w, {}).get("prices", {}),
                        "served_demand": scenario_results.get(w, {}).get("served_demand", {}),
                        "tariffs": relative_tariffs,
                    }
                    plot_title = f"Solution for {config_name}_ins{instance_idx} (Total Scenarios: {n_scenarios})\n--- VISUALIZING SCENARIO {w} (Avg Tariff Rate: {avg_tariff_rate:.1%}) ---\nOpened Facilities: {solution_details['opened_facilities']}"
                    
                    save_path = plots_dir / f"{config_name}_ins{instance_idx}_scen{w}_tariff{avg_tariff_rate:.2f}.png"
                    # Generate and save the plot
                    visualize_network(ext_form.instance, solution_details, plot_title, save_path=save_path)