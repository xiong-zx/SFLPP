# Utils Module Guide

## Overview

The `utils.py` module centralizes common functionality used across the SFLPP project, making the codebase more organized, maintainable, and consistent.

### Module Structure

The `utils.py` module is organized into the following sections:

1. **Project Root and Directory Structure**
   - `get_project_root()`: Get the project root directory
   - `setup_directories()`: Set up project directories with dist/non-dist versions

2. **File Path Construction**
   - `instance_file_path()`: Construct path for instance JSON files
   - `ef_file_path()`: Construct path for extensive form pickle files
   - `config_file_path()`: Construct path for config JSON files
   - `instance_name()`: Generate standard instance name
   - `ef_name()`: Generate standard extensive form name

3. **Gurobi Utilities**
   - `load_gurobi_params()`: Load Gurobi parameters from JSON
   - `apply_gurobi_defaults()`: Apply Gurobi parameters globally
   - `solve_gurobi_model()`: Solve a Gurobi model with parameters

4. **Random Seed Generation**
   - `seed_stream()`: Generate infinite stream of random seeds

5. **File I/O Utilities**
   - `save_json()`: Save data to JSON file
   - `load_json()`: Load data from JSON file
   - `save_dataframe_results()`: Save DataFrame to CSV and/or JSON

6. **Results Management**
   - `create_benchmark_filename()`: Create descriptive filename for benchmarks
   - `save_benchmark_results()`: Save benchmark results with standardized naming
   - `load_optimal_results_lookup()`: Load optimal results into lookup dictionary

7. **Plotting Utilities**
   - `setup_matplotlib_defaults()`: Set up matplotlib with consistent defaults
   - `save_plot()`: Save matplotlib figure with consistent settings

8. **Printing and Formatting**
   - `print_section_header()`: Print formatted section header
   - `print_subsection_header()`: Print formatted subsection header
   - `print_separator()`: Print separator line
   - `format_time()`: Format time to human-readable string
   - `format_percentage()`: Format decimal as percentage string

9. **Validation Utilities**
   - `validate_files_exist()`: Check if files exist
   - `load_config_set()`: Load multiple config files

10. **Summary Statistics**
    - `compute_gap()`: Compute relative gap
    - `compute_speedup()`: Compute speedup factor

### EF_runner.py Example

```python
from core.utils import (
    setup_directories,
    instance_file_path,
    ef_file_path,
    load_gurobi_params,
    save_json,
)

# Use dist version
DIRS = setup_directories(use_dist_version=True, create=True)
DATA_DIR = DIRS['data']
RESULTS_DIR = DIRS['results']
LOG_DIR = DIRS['log']

# Use path utilities
inst_path = instance_file_path(config_name, inst_idx, DATA_DIR)
ef_path = ef_file_path(config_name, inst_idx, n_scenarios, DATA_DIR)

# Load params
gurobi_params = load_gurobi_params(DIRS['config'] / "gurobi_params.json")

# Save results
save_json(result, LOG_DIR / f"{config_name}_ins{inst_idx}_s{n_scenarios}_log.json")
```

### PH_runner.py Example

```python
from utils import (
    setup_directories,
    load_optimal_results_lookup,
    print_section_header,
    compute_gap,
)

# Load optimal results for comparison
optimal_lookup = load_optimal_results_lookup(EF_LOG_DIR / "summary.json")

# Compute gap
gap = compute_gap(ph_objective, optimal_lookup.get(key))

# Nice headers
print_section_header(f"Running PH for {config_name}")
```

### generate_data.py Example

```python
from utils import (
    setup_directories,
    instance_name,
    ef_name,
    seed_stream,
    print_section_header,
)

# Use seed stream
seeds = seed_stream(base_seed=42)
for _ in range(10):
    seed = next(seeds)
    # Use seed for generation...

# Use naming conventions
inst_name = instance_name(config_name, inst_idx)
ef_name = ef_name(config_name, inst_idx, n_scenarios)
```

## Testing

You can test the utils module by running:

```bash
python core/utils.py
```

This will display example output showing:
- Directory setup
- Path construction
- Filename generation

## Future Enhancements

Potential additions to the utils module:

1. **Parallel Processing Utilities**
   - Helper functions for multiprocessing
   - Progress bar integration

2. **Logging Utilities**
   - Standardized logging setup
   - Result logging formatters

3. **Data Validation**
   - Instance validation functions
   - Config validation functions

4. **More Plotting Utilities**
   - Common plot types (convergence plots, comparison plots)
   - Consistent color schemes
   - Legend formatting

## Migration Guide

To migrate existing files to use utils.py:

1. **Add import statement** at the top of the file
2. **Replace local functions** with utils functions
3. **Update path construction** to use utility functions
4. **Replace manual saving** with `save_benchmark_results()`
5. **Replace print statements** with `print_section_header()` / `print_subsection_header()`
6. **Test** to ensure everything works
