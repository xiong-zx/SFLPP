"""
Utility functions for SFLPP project.

This module provides common utilities used across the project including:
- Path management and file naming conventions
- Directory setup with dist/non-dist versions
- Gurobi parameter handling
- File I/O operations (JSON, CSV, pickle)
- Random seed generation
- Result saving and formatting
- Plotting utilities

Author: SFLPP Project
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt


# =============================================================================
# Project Root and Directory Structure
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent


def setup_directories(
    use_dist_version: bool = True,
    create: bool = True
) -> Dict[str, Path]:
    """
    Set up project directories based on version flag.

    Args:
        use_dist_version: If True, use _dist suffix for data directories
        create: If True, create directories if they don't exist

    Returns:
        Dictionary with directory paths: config, data, results, log, ph_log, plots
    """
    root = get_project_root()

    dirs = {
        'root': root,
        'config': root / "config",
        'data': root / ("data_dist" if use_dist_version else "data"),
        'results': root / ("results_dist" if use_dist_version else "results"),
        'log': root / ("log_dist" if use_dist_version else "log"),
        'ph_log': root / ("ph_log_dist" if use_dist_version else "ph_log"),
        'plots': root / ("plots_dist" if use_dist_version else "plots"),
    }

    if create:
        for key in ['config', 'data', 'results', 'log', 'ph_log', 'plots']:
            dirs[key].mkdir(parents=True, exist_ok=True)

    return dirs


# =============================================================================
# File Path Construction (Naming Conventions)
# =============================================================================

def instance_file_path(
    config_name: str,
    instance_idx: int,
    data_dir: Optional[Path] = None
) -> Path:
    """
    Construct path for instance file.

    Args:
        config_name: Base config name (without .json)
        instance_idx: Instance index number
        data_dir: Data directory (if None, uses current DATA_DIR from setup)

    Returns:
        Path to instance JSON file
    """
    if data_dir is None:
        data_dir = setup_directories(create=False)['data']
    return data_dir / f"{config_name}_ins{instance_idx}.json"


def ef_file_path(
    config_name: str,
    instance_idx: int,
    n_scenarios: int,
    data_dir: Optional[Path] = None
) -> Path:
    """
    Construct path for extensive form (EF) file.

    Args:
        config_name: Base config name
        instance_idx: Instance index number
        n_scenarios: Number of scenarios
        data_dir: Data directory

    Returns:
        Path to extensive form pickle file
    """
    if data_dir is None:
        data_dir = setup_directories(create=False)['data']
    return data_dir / f"{config_name}_ins{instance_idx}_s{n_scenarios}.pkl"


def instance_name(config_name: str, inst_idx: int) -> str:
    """Generate standard instance name."""
    return f"{config_name}_ins{inst_idx}"


def ef_name(config_name: str, inst_idx: int, n_scenarios: int) -> str:
    """Generate standard extensive form name."""
    return f"{instance_name(config_name, inst_idx)}_s{n_scenarios}"


def config_file_path(config_name: str, config_dir: Optional[Path] = None) -> Path:
    """
    Construct path for config file.

    Args:
        config_name: Config name (without .json extension)
        config_dir: Config directory

    Returns:
        Path to config JSON file
    """
    if config_dir is None:
        config_dir = setup_directories(create=False)['config']
    return config_dir / f"{config_name}.json"


# =============================================================================
# Gurobi Utilities
# =============================================================================

def load_gurobi_params(path: str | Path) -> Dict[str, float | str]:
    """
    Load Gurobi parameters from a JSON file.

    Args:
        path: Path to JSON file containing Gurobi parameters

    Returns:
        Dictionary of Gurobi parameters (empty dict if file doesn't exist)
    """
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, dict):
        raise ValueError(f"Gurobi params file {path} must contain a JSON object.")
    return params


def apply_gurobi_defaults(params: Dict[str, float | str]) -> None:
    """
    Apply Gurobi parameters as global defaults so all models inherit them.

    Args:
        params: Dictionary of Gurobi parameter key-value pairs
    """
    for key, val in params.items():
        gp.setParam(key, val)


def solve_gurobi_model(
    model: gp.Model,
    log_file: Optional[Path | str] = None,
    params: Optional[Dict[str, float | str]] = None,
) -> None:
    """
    Solve a Gurobi model with optional parameters and logging.

    Args:
        model: Gurobi model to solve
        log_file: Optional path to log file
        params: Optional dict of Gurobi parameters
    """
    if params:
        for k, v in params.items():
            model.setParam(k, v)
    if log_file is not None:
        model.setParam("LogFile", str(log_file))
    model.optimize()


# =============================================================================
# Random Seed Generation
# =============================================================================

def seed_stream(base_seed: int | None) -> Iterator[int]:
    """
    Generate an infinite stream of integer seeds derived from a base seed.

    Args:
        base_seed: Base seed for RNG (None for non-deterministic)

    Yields:
        Random integer seeds
    """
    rng = np.random.default_rng(base_seed)
    while True:
        yield int(rng.integers(0, 2**32 - 1))


# =============================================================================
# File I/O Utilities
# =============================================================================

def save_json(data: Any, path: Path | str, indent: int = 2, sort_keys: bool = False) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        path: Output file path
        indent: JSON indentation level
        sort_keys: Whether to sort dictionary keys
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)


def load_json(path: Path | str) -> Any:
    """
    Load data from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Loaded data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe_results(
    df: pd.DataFrame,
    base_path: Path | str,
    save_csv: bool = True,
    save_json: bool = True,
    results_list: Optional[List[Dict]] = None
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Save DataFrame results to CSV and/or JSON.

    Args:
        df: DataFrame to save
        base_path: Base path (without extension)
        save_csv: Whether to save as CSV
        save_json: Whether to save as JSON
        results_list: Optional list of dicts for JSON (uses df if None)

    Returns:
        Tuple of (csv_path, json_path) or (None, None) if not saved
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = None
    json_path = None

    if save_csv:
        csv_path = base_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")

    if save_json:
        json_path = base_path.with_suffix('.json')
        data = results_list if results_list is not None else df.to_dict(orient='records')
        save_json(data, json_path, indent=2)
        print(f"Results saved to: {json_path}")

    return csv_path, json_path


# =============================================================================
# Results Management
# =============================================================================

def create_benchmark_filename(
    config_name: str,
    instance_list: List[int],
    scenarios_list: List[int],
    suffix: str = "",
    price_levels_list: Optional[List[int]] = None
) -> str:
    """
    Create a descriptive filename for benchmark results.

    Args:
        config_name: Configuration name
        instance_list: List of instance indices
        scenarios_list: List of scenario counts
        suffix: Additional suffix for filename
        price_levels_list: Optional list of price levels

    Returns:
        Formatted filename (without extension)
    """
    inst_str = "_".join(map(str, instance_list))
    scenarios_str = "_".join(map(str, scenarios_list))

    parts = [
        config_name,
        f"ins{inst_str}",
        f"scenarios_{scenarios_str}"
    ]

    if price_levels_list:
        price_str = "_".join(map(str, price_levels_list))
        parts.append(f"prices_{price_str}")

    if suffix:
        parts.append(suffix)

    return "_".join(parts)


def save_benchmark_results(
    results: List[Dict],
    config_name: str,
    instance_list: List[int],
    scenarios_list: List[int],
    results_dir: Path | str,
    suffix: str = "",
    price_levels_list: Optional[List[int]] = None
) -> Tuple[Path, Path]:
    """
    Save benchmark results with standardized naming.

    Args:
        results: List of result dictionaries
        config_name: Configuration name
        instance_list: List of instance indices
        scenarios_list: List of scenario counts
        results_dir: Results directory
        suffix: Additional suffix for filename
        price_levels_list: Optional list of price levels

    Returns:
        Tuple of (csv_path, json_path)
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = create_benchmark_filename(
        config_name,
        instance_list,
        scenarios_list,
        suffix,
        price_levels_list
    )

    df = pd.DataFrame(results)
    base_path = results_dir / filename

    return save_dataframe_results(df, base_path, save_csv=True, save_json=True, results_list=results)


def load_optimal_results_lookup(
    summary_path: Path | str
) -> Dict[Tuple[str, int, int], float]:
    """
    Load optimal results from summary JSON into a lookup dictionary.

    Args:
        summary_path: Path to summary JSON file from EF_runner

    Returns:
        Dictionary mapping (config, instance_idx, n_scenarios) to objective value
    """
    summary_path = Path(summary_path)
    if not summary_path.exists():
        print(f"Warning: Optimal results file not found at {summary_path}.")
        return {}

    ef_results = load_json(summary_path)

    lookup = {}
    for res in ef_results:
        key = (res["config"], res["instance_idx"], res["n_scenarios"])
        if res.get("objective") is not None:
            lookup[key] = res["objective"]

    return lookup


# =============================================================================
# Plotting Utilities
# =============================================================================

def setup_matplotlib_defaults(font_size: int = 10) -> None:
    """
    Set up matplotlib with consistent defaults for the project.

    Args:
        font_size: Base font size
    """
    plt.rcParams.update({
        'font.size': font_size,
        'axes.labelsize': font_size + 2,
        'axes.titlesize': font_size + 4,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size - 1,
        'figure.titlesize': font_size + 6,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def save_plot(
    fig: plt.Figure,
    filename: str,
    plots_dir: Optional[Path] = None,
    dpi: int = 300,
    **kwargs
) -> Path:
    """
    Save matplotlib figure with consistent settings.

    Args:
        fig: Matplotlib figure
        filename: Output filename
        plots_dir: Plots directory (uses default if None)
        dpi: Resolution in dots per inch
        **kwargs: Additional savefig arguments

    Returns:
        Path to saved file
    """
    if plots_dir is None:
        plots_dir = setup_directories(create=True)['plots']

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_path = plots_dir / filename
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', **kwargs)
    print(f"Plot saved to: {save_path}")

    return save_path


# =============================================================================
# Printing and Formatting Utilities
# =============================================================================

def print_section_header(
    title: str,
    width: int = 70,
    char: str = "#"
) -> None:
    """
    Print a formatted section header.

    Args:
        title: Section title
        width: Total width of header
        char: Character to use for border
    """
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


def print_subsection_header(
    title: str,
    width: int = 70,
    char: str = "*"
) -> None:
    """
    Print a formatted subsection header.

    Args:
        title: Subsection title
        width: Total width of header
        char: Character to use for border
    """
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


def print_separator(width: int = 70, char: str = "=") -> None:
    """
    Print a separator line.

    Args:
        width: Width of separator
        char: Character to use
    """
    print(char * width)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as percentage string.

    Args:
        value: Decimal value (e.g., 0.05 for 5%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_files_exist(
    file_paths: List[Path | str],
    error_on_missing: bool = True
) -> Tuple[List[Path], List[Path]]:
    """
    Check if files exist and return lists of existing and missing files.

    Args:
        file_paths: List of file paths to check
        error_on_missing: If True, raise FileNotFoundError for missing files

    Returns:
        Tuple of (existing_files, missing_files)
    """
    existing = []
    missing = []

    for path in file_paths:
        path = Path(path)
        if path.exists():
            existing.append(path)
        else:
            missing.append(path)

    if error_on_missing and missing:
        raise FileNotFoundError(
            f"Missing files:\n" + "\n".join(f"  - {p}" for p in missing)
        )

    return existing, missing


# =============================================================================
# Configuration Helpers
# =============================================================================

def load_config_set(
    config_names: List[str],
    config_dir: Optional[Path] = None,
    validate: bool = True
) -> Dict[str, Dict]:
    """
    Load multiple config files.

    Args:
        config_names: List of config names (without .json)
        config_dir: Config directory
        validate: If True, raise error if any config is missing

    Returns:
        Dictionary mapping config_name to config data
    """
    if config_dir is None:
        config_dir = setup_directories(create=False)['config']

    configs = {}
    missing = []

    for name in config_names:
        path = config_file_path(name, config_dir)
        if path.exists():
            configs[name] = load_json(path)
        else:
            missing.append(path)

    if validate and missing:
        raise FileNotFoundError(
            f"Missing config files:\n" + "\n".join(f"  - {p}" for p in missing)
        )

    return configs


# =============================================================================
# Summary Statistics
# =============================================================================

def compute_gap(actual: float, optimal: float) -> Optional[float]:
    """
    Compute relative gap between actual and optimal objective.

    Args:
        actual: Actual objective value
        optimal: Optimal objective value

    Returns:
        Relative gap (None if optimal is near zero)
    """
    if abs(optimal) < 1e-8:
        return None
    return (actual - optimal) / abs(optimal)


def compute_speedup(baseline_time: float, actual_time: float) -> Optional[float]:
    """
    Compute speedup factor.

    Args:
        baseline_time: Baseline computation time
        actual_time: Actual computation time

    Returns:
        Speedup factor (None if actual_time is near zero)
    """
    if actual_time < 1e-8:
        return None
    return baseline_time / actual_time


if __name__ == "__main__":
    # Example usage
    print("SFLPP Utils Module")
    print_section_header("Directory Setup")

    dirs = setup_directories(use_dist_version=True, create=False)
    for key, path in dirs.items():
        print(f"{key:>10}: {path}")

    print_section_header("Path Construction")
    print(f"Instance:  {instance_file_path('c5_f5_cf1', 1)}")
    print(f"EF file:   {ef_file_path('c5_f5_cf1', 1, 10)}")
    print(f"Config:    {config_file_path('c5_f5_cf1')}")

    print_section_header("Filename Generation")
    filename = create_benchmark_filename(
        'c5_f5_cf1',
        [1, 2, 3],
        [10, 20, 50],
        suffix='bench2',
        price_levels_list=[5, 10, 15]
    )
    print(f"Benchmark filename: {filename}")
