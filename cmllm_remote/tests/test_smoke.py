from cmllm.pipeline import normalize_name


def test_normalize_name():
    assert normalize_name("interaction between miner and drill pipe") == "interaction_between_miner_and_drill_pipe"

