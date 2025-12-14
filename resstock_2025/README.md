# MUST READ BEFORE RUNNING OCHRE:

Read the OCHRE_FIXES.md file in this directory before running OCHRE with the 2025 ResStock dataset.

# Development Environment (Poetry)

This project uses Poetry
 to manage Python dependencies and virtual environments. Poetry ensures that everyone runs the project with the same Python version and package versions, which is especially important for reproducible simulation results.

 ## Python Version Requirements
This project requires (because of OCHRE):

- Python ≥ 3.12 and < 3.13

## Installing Poetry

From the Poetry documentation:

- Step 1:
~~~sh
curl -sSL https://install.python-poetry.org | python3 -
~~~

- Step 2:

~~~sh
export PATH="$HOME/.local/bin:$PATH"
~~~

- Step 3:
~~~ sh
poetry --version
~~~

## Installing Project Dependencies

- From the repository root (where pyproject.toml lives):

~~~sh
poetry install
~~~

# Load Allocation Workflow
This repo also contains scripts to:

- read simulation outputs from the dataset directory,

- compute customer-level and aggregate statistics, and

- run several transformer sizing / allocation methods.

## Key Scripts

- config.toml: 
    - Experiment configuration (dataset path, upgrades, run_id, results directory, and method parameters).

- load_profiles.py:
    - The “engine”: reads simulation output CSVs, builds per-customer load profiles, computes customer summaries, and exposes aggregate calculations.

- load_allocations_api.py:
    - The “runner”: reads config.toml, runs Methods 1–4, and writes results to CSV files.

## Results Output

Each method writes one or more CSV files to the configured results_dir, using the naming pattern:

~~~sh
results/<run_id>_<method-name>.csv
~~~

Examples:

- <run_id>_method1.csv

- <run_id>_method2_data.csv

- <run_id>_method2_regression.csv
 
- <run_id>_method3_kva_25.csv, <run_id>_method3_kva_50.csv, <run_id>_method3_kva_75.csv
 
- <run_id>_method3_regr.csv
 
- <run_id>_method4_allocation_results.csv
 
- <run_id>_method4_summary.csv

## How to run the scripts

From the resStock folder:

- Method 1: Diversity Factor:

~~~sh
poetry run python load_allocations_api.py method1
~~~

- Method 2: Load survey + regression (kWh vs peak kW)

~~~sh
poetry run python load_allocations_api.py method2
~~~

- Method 3: Transformer Load Management (TLM) + regression

~~~sh
poetry run python load_allocations_api.py method3
~~~

- Method 4: Method 4: Metered feeder max demand allocation

~~~sh
poetry run python load_allocations_api.py method4
~~~

