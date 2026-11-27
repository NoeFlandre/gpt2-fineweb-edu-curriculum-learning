from gpt2_scratch.experiment import naming
from scripts import prepare_data


def test_run_id_names_the_initialization_and_data_seeds():
    assert naming.run_id(43) == "init43-data42"
    assert naming.run_id(43, data_seed=7) == "init43-data7"


def test_the_data_seed_is_the_one_that_builds_the_data():
    assert naming.DATA_SEED == prepare_data.DATA_SEED
