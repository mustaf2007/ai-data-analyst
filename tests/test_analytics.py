import pandas as pd
import pytest
from analytics import (
    DataInputError, clean_dataset, load_dataset, profile_dataset, safe_csv_bytes,
    categorical_summary, correlation_matrix, grouped_summary, infer_datetime_columns,
)


def test_load_csv_and_unique_headers():
    df = load_dataset("data.csv", b"a,a\n1,2\n")
    assert list(df.columns) == ["a", "a.1"]


def test_reject_unsupported_file():
    with pytest.raises(DataInputError):
        load_dataset("data.txt", b"x")


def test_reject_empty_file():
    with pytest.raises(DataInputError):
        load_dataset("empty.csv", b"")


def test_profile_counts_missing_and_duplicates():
    df = pd.DataFrame({"x": [1, 1, None], "y": ["a", "a", "b"]})
    profile = profile_dataset(df)
    assert profile["rows"] == 3
    assert profile["missing_cells"] == 1
    assert profile["duplicate_rows"] == 1


def test_cleaning_does_not_mutate_original():
    df = pd.DataFrame({"x": [1, None, 1]})
    out = clean_dataset(
        df, drop_duplicates=True, missing_strategy="Fill numeric with median"
    )
    assert df.isna().sum().iloc[0] == 1
    assert out.isna().sum().iloc[0] == 0
    assert len(out) == 2


def test_cleaning_respects_selected_columns():
    df = pd.DataFrame({"x": [1, None], "y": [None, "ok"]})
    out = clean_dataset(
        df, missing_strategy="Fill numeric with zero", selected_columns=["x"]
    )
    assert out["x"].isna().sum() == 0
    assert out["y"].isna().sum() == 1


def test_grouped_summary_and_correlation():
    df = pd.DataFrame({"group": ["a", "a", "b"], "value": [1, 3, 10], "other": [2, 6, 20]})
    grouped = grouped_summary(df, "group", "value", "mean")
    assert grouped.loc[grouped.group == "a", "value"].iloc[0] == 2
    corr = correlation_matrix(df, "spearman")
    assert corr.loc["value", "other"] == pytest.approx(1.0)


def test_categorical_summary_includes_missing():
    df = pd.DataFrame({"kind": ["a", None, "b"]})
    result = categorical_summary(df)["kind"]
    assert "(Missing)" in result["value"].tolist()


def test_datetime_inference():
    df = pd.DataFrame({"date": ["2025-01-01", "2025-02-01", "2025-03-01"], "label": ["a", "b", "c"]})
    assert "date" in infer_datetime_columns(df)


def test_csv_formula_neutralization():
    raw = safe_csv_bytes(pd.DataFrame({"text": ["=1+1", "safe", "@SUM(A1:A2)"]})).decode("utf-8-sig")
    assert "'=1+1" in raw
    assert "'@SUM(A1:A2)" in raw
