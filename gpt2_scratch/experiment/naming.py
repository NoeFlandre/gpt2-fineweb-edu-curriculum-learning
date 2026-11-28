"""Run names. Each run is named by its learning rate, model size, token budget and both seeds.

A run's folder, its Trackio name and its folder in the model repository all come from run_id().
"""

from gpt2_scratch.model.config import STUDY_CONFIG
from gpt2_scratch.training.config import TrainConfig

DATA_SEED = 42  # the data seed of every run so far; scripts/prepare_data.py uses the same value


def run_id(init_seed: int, data_seed: int = DATA_SEED) -> str:
    """The name of the runs with this initialization seed and data seed.

    For example lr6e-4_params45.2M_tokens899.7M_init42_data42: the peak learning rate, the
    parameter count, the tokens actually trained, then the initialization and data seeds.
    """
    train = TrainConfig()
    params = _millions(STUDY_CONFIG.n_params)
    tokens = _millions(train.max_steps * train.total_batch_size)
    learning_rate = _scientific(train.max_lr)
    return f"lr{learning_rate}_params{params}M_tokens{tokens}M_init{init_seed}_data{data_seed}"


def _millions(count: int) -> str:
    return f"{count / 1e6:.1f}"


def _scientific(value: float) -> str:
    """6e-4 rather than 6e-04."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"
