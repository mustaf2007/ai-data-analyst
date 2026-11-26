"""Session-state invalidation helpers for dataset-dependent results."""


DATASET_DERIVED_STATE_DEFAULTS = {
    "model_result": None,
    "anomalies": None,
    "model_config": None,
    "anomaly_config": None,
    "statistical_test_result": None,
    "comparison_result": None,
    "comparison_signature": None,
}


def clear_dataset_derived_state(state):
    """Invalidate every stored result that depends on the active dataset."""
    for key, value in DATASET_DERIVED_STATE_DEFAULTS.items():
        state[key] = value
