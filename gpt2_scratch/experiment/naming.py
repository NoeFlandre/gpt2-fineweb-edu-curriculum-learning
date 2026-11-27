"""Run names. Each run is named by its initialization seed and its data seed.

A run's folder, its Trackio name and its folder in the model repository all come from run_id().
"""

DATA_SEED = 42  # the data seed of every run so far; scripts/prepare_data.py uses the same value


def run_id(init_seed: int, data_seed: int = DATA_SEED) -> str:
    """The name of the runs with this initialization seed and data seed, such as init42-data42."""
    return f"init{init_seed}-data{data_seed}"
