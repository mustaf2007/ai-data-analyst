import pandas as pd
import pytest
from analytics import (
    DataInputError, clean_dataset, load_dataset, profile_dataset, safe_csv_bytes,
    categorical_summary, correlation_matrix, grouped_summary, infer_datetime_columns,
    build_cleaning_report, validate_analysis_readiness, column_quality_diagnostics,
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



def test_cleaning_report_counts_filled_cells_and_duplicates():
    df = pd.DataFrame({
        "number": [1.0, None, 1.0, 3.0],
        "category": ["a", "b", "a", None],
    })
    cleaned = clean_dataset(
        df, drop_duplicates=True, missing_strategy="Fill numeric with median"
    )
    report = build_cleaning_report(
        df, cleaned, drop_duplicates=True,
        missing_strategy="Fill numeric with median",
    )
    assert report["rows_before"] == 4
    assert report["rows_after"] == 3
    assert report["duplicates_removed"] == 1
    assert report["missing_filled"] == 1
    assert report["missing_before"] == 2
    assert report["missing_after"] == 1
    assert report["per_column"].loc[
        report["per_column"]["column"] == "number", "missing_after"
    ].iloc[0] == 0


def test_cleaning_report_counts_rows_removed_for_missing():
    df = pd.DataFrame({"x": [1, None, 3], "y": ["a", "b", None]})
    cleaned = clean_dataset(
        df, missing_strategy="Drop rows with missing values", selected_columns=["x"]
    )
    report = build_cleaning_report(
        df, cleaned, missing_strategy="Drop rows with missing values",
        selected_columns=["x"],
    )
    assert report["rows_removed_for_missing"] == 1
    assert report["rows_after"] == 2



def test_analysis_readiness_reports_all_checks_passed_for_clean_dataset():
    from analytics import validate_analysis_readiness
    df = pd.DataFrame({"sales": [10, 20, 30], "region": ["A", "B", "C"]})
    result = validate_analysis_readiness(df)
    assert result["status"] == "Ready"
    assert result["score"] == 100.0
    assert result["passed"] == result["total"]


def test_analysis_readiness_flags_missing_duplicates_and_constant_columns():
    from analytics import validate_analysis_readiness
    df = pd.DataFrame({
        "value": [1.0, 1.0, None],
        "constant": ["same", "same", "same"],
    })
    result = validate_analysis_readiness(df)
    assert result["status"] == "Ready with warnings"
    assert result["score"] < 100
    assert result["warnings"] >= 3


def test_analysis_readiness_blocks_infinite_numeric_values():
    from analytics import validate_analysis_readiness
    df = pd.DataFrame({"value": [1.0, float("inf"), 3.0], "group": ["a", "b", "c"]})
    result = validate_analysis_readiness(df)
    assert result["status"] == "Not ready"
    assert result["blockers"] >= 1
    assert result["checks"].loc[
        result["checks"]["check"] == "No infinite numeric values", "status"
    ].iloc[0] == "Blocked"



def test_column_diagnostics_classifies_missingness_and_flags_headers_and_numeric_text():
    df = pd.DataFrame({
        "Unnamed: 2": [1, None, None, None, None],
        "amount_text": ["10", "20", "30", "40", None],
    })
    result = column_quality_diagnostics(df).set_index("column")
    assert result.loc["Unnamed: 2", "missingness_level"] == "High"
    assert result.loc["Unnamed: 2", "header_review"] == "Review name"
    assert result.loc["amount_text", "type_review"] == "Possibly numeric stored as text"


def test_correlation_readiness_requires_two_numeric_columns():
    df = pd.DataFrame({"amount": [1, 2, 3], "label": ["a", "b", "c"]})
    result = validate_analysis_readiness(df, "Correlation analysis")
    assert result["status"] == "Not ready"
    assert "Correlation has at least two numeric variables" in result["checks"]["check"].tolist()


def test_regression_readiness_checks_numeric_target_and_features():
    df = pd.DataFrame({"target": ["low", "high", "medium"], "feature": [1, 2, 3]})
    result = validate_analysis_readiness(df, "Regression", "target")
    checks = result["checks"].set_index("check")
    assert checks.loc["Regression target is numeric", "status"] == "Blocked"


def test_classification_readiness_accepts_two_classes_and_feature():
    df = pd.DataFrame({"target": ["yes", "no", "yes"], "feature": [1, 2, 3]})
    result = validate_analysis_readiness(df, "Classification", "target")
    checks = result["checks"].set_index("check")
    assert checks.loc["Classification target has at least two classes", "status"] == "Passed"
    assert checks.loc["At least one varying feature is available", "status"] == "Passed"


def test_time_series_readiness_requires_detectable_date_column():
    df = pd.DataFrame({"period": ["2025-01-01", "2025-02-01"], "value": [1, 2]})
    result = validate_analysis_readiness(df, "Time-series analysis")
    checks = result["checks"].set_index("check")
    assert checks.loc["A date/time column is identifiable", "status"] == "Passed"



def test_combined_cleaning_fills_numeric_and_categorical_and_drops_empty_columns():
    df = pd.DataFrame({
        "number": [1.0, None, 3.0],
        "category": ["a", None, "b"],
        "empty": [None, None, None],
    })
    cleaned = clean_dataset(
        df,
        drop_duplicates=True,
        drop_empty_columns=True,
        missing_strategy="Fill numeric with median and categorical with mode",
    )
    assert "empty" not in cleaned.columns
    assert cleaned["number"].isna().sum() == 0
    assert cleaned["category"].isna().sum() == 0
    assert len(cleaned) == 3


def test_cleaned_dataset_can_be_downloaded_after_combined_cleaning():
    from analytics import safe_csv_bytes
    df = pd.DataFrame({"number": [1.0, None], "category": ["x", None]})
    cleaned = clean_dataset(
        df, drop_empty_columns=True,
        missing_strategy="Fill numeric with median and categorical with mode",
    )
    exported = pd.read_csv(__import__("io").BytesIO(safe_csv_bytes(cleaned)))
    assert exported.isna().sum().sum() == 0
