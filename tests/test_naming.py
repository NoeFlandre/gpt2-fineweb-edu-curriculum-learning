from gpt2_scratch.experiment import naming
from scripts import prepare_data


def test_run_id_names_the_learning_rate_size_tokens_and_seeds():
    assert naming.run_id(43) == "lr6e-4_params45.2M_tokens899.7M_init43_data42"
    assert naming.run_id(43, data_seed=7) == "lr6e-4_params45.2M_tokens899.7M_init43_data7"


def test_the_data_seed_is_the_one_that_builds_the_data():
    assert naming.DATA_SEED == prepare_data.DATA_SEED
