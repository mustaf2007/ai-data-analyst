import numpy as np
import pandas as pd
import pytest
from modeling import detect_anomalies, train_predictive_model


def test_anomaly_output_columns():
    df = pd.DataFrame({"x": list(range(30)), "y": list(range(30, 60))})
    out = detect_anomalies(df)
    assert {"anomaly_flag", "anomaly_score"}.issubset(out.columns)
    assert set(out["anomaly_flag"].unique()).issubset({"Anomaly", "Normal"})


def test_anomaly_detection_handles_missing_and_infinite_values():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, np.inf] + list(range(5, 21))})
    out = detect_anomalies(df)
    assert len(out) == len(df)
    assert out["anomaly_score"].notna().all()


def test_anomaly_detection_rejects_non_numeric_data():
    with pytest.raises(ValueError):
        detect_anomalies(pd.DataFrame({"label": ["a"] * 12}))


def test_regression_model_returns_metrics():
    df = pd.DataFrame({
        "x": range(60),
        "z": [i % 7 for i in range(60)],
        "target": [2 * i + 1 for i in range(60)],
    })
    result = train_predictive_model(df, "target", "Regression")
    assert "RMSE" in result["metrics"]
    assert len(result["comparison"]) > 0
    assert set(result["comparison"].columns) == {"actual", "predicted"}


def test_classification_model_returns_metrics():
    df = pd.DataFrame({
        "x": range(80),
        "category": ["even" if i % 2 == 0 else "odd" for i in range(80)],
        "target": ["yes" if i % 2 == 0 else "no" for i in range(80)],
    })
    result = train_predictive_model(df, "target", "Classification")
    assert "Accuracy" in result["metrics"]
    assert "F1 weighted" in result["metrics"]
    assert set(result["comparison"]["actual"].unique()) == {"yes", "no"}


def test_model_rejects_small_dataset():
    with pytest.raises(ValueError):
        train_predictive_model(pd.DataFrame({"x": range(10), "target": range(10)}), "target")


def test_model_rejects_invalid_test_split():
    df = pd.DataFrame({"x": range(20), "target": range(20)})
    with pytest.raises(ValueError, match="Test-set share"):
        train_predictive_model(df, "target", "Regression", test_size=0)


def test_classification_rejects_test_split_too_small_for_classes():
    df = pd.DataFrame({
        "x": range(20),
        "target": ["A"] * 8 + ["B"] * 6 + ["C"] * 6,
    })
    with pytest.raises(ValueError, match="test split is too small"):
        train_predictive_model(df, "target", "Classification", test_size=0.1)


def test_classification_rejects_singleton_target_class():
    df = pd.DataFrame({
        "x": range(20),
        "target": ["A"] * 19 + ["rare"],
    })
    with pytest.raises(ValueError, match="at least two observations"):
        train_predictive_model(df, "target", "Classification")


def test_model_rejects_only_constant_features():
    df = pd.DataFrame({"constant": [1] * 20, "target": range(20)})
    with pytest.raises(ValueError, match="No varying, usable feature"):
        train_predictive_model(df, "target", "Regression")


def test_anomaly_detection_validates_contamination_and_reserved_columns():
    df = pd.DataFrame({"x": range(20)})
    with pytest.raises(ValueError, match="no more than 0.5"):
        detect_anomalies(df, contamination=0.6)
    with pytest.raises(ValueError, match="reserved anomaly output"):
        detect_anomalies(pd.DataFrame({"x": range(20), "anomaly_flag": ["x"] * 20}))
