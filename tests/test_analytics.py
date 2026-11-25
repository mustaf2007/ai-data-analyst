import numpy as np
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



def test_cleaning_removes_whitespace_only_unnamed_columns_and_audit_handles_removed_columns():
    df = pd.DataFrame({
        "Year": [1961, 1962, 1963],
        "Sorghum production tonnes": [352500, 350000, 330700],
        "Unnamed: 2": ["", "   ", None],
        "Temperature anomaly": [-0.4264, -0.60355, -0.70704],
        "Unnamed: 4": [None, "", "  "],
        "Precipitation anomaly": [104.684, -14.9681, 133.1915],
    })
    cleaned = clean_dataset(
        df, drop_empty_columns=True,
        missing_strategy="Fill numeric with median and categorical with mode",
    )
    assert list(cleaned.columns) == [
        "Year", "Sorghum production tonnes", "Temperature anomaly", "Precipitation anomaly"
    ]
    assert cleaned["Sorghum production tonnes"].tolist() == [352500, 350000, 330700]
    report = build_cleaning_report(
        df, cleaned, drop_duplicates=True,
        missing_strategy="Fill numeric with median and categorical with mode",
    )
    removed = report["per_column"].set_index("column")
    assert bool(removed.loc["Unnamed: 2", "column_removed"])
    assert bool(removed.loc["Unnamed: 4", "column_removed"])
    assert report["missing_after"] == 0



def test_detailed_insights_include_quality_distribution_correlation_and_categories():
    from analytics import generate_detailed_insights
    df = pd.DataFrame({
        "sales": [10.0, 20.0, 30.0, 40.0],
        "cost": [5.0, 10.0, 15.0, 20.0],
        "region": ["North", "North", "South", None],
    })
    result = generate_detailed_insights(df)
    areas = {item["area"] for item in result}
    assert {"Dataset", "Distribution", "Relationships", "Categories"} <= areas
    relationship = next(item for item in result if item["area"] == "Relationships")
    assert "not causation" in relationship["caveat"].lower()
    assert all(item["evidence"] and item["caveat"] for item in result)


def test_detailed_insights_report_missingness_and_duplicates():
    from analytics import generate_detailed_insights
    df = pd.DataFrame({"value": [1.0, None, 1.0], "group": ["a", "b", "a"]})
    result = generate_detailed_insights(df)
    quality = [item["finding"] for item in result if item["area"] == "Data quality"]
    assert any("missing cells" in item for item in quality)
    assert any("duplicated rows" in item for item in quality)


def test_detailed_insights_empty_data_and_limit_are_safe():
    from analytics import generate_detailed_insights
    assert generate_detailed_insights(pd.DataFrame()) == []
    df = pd.DataFrame({"x": [1, 2, 3], "group": ["a", "b", "c"]})
    assert len(generate_detailed_insights(df, max_findings=1)) == 1



def test_data_story_includes_selected_evidence_and_limitations():
    from analytics import build_data_story
    findings = [
        {"area": "Dataset", "finding": "The dataset contains 100 rows.",
         "evidence": "100 observations; 4 variables.",
         "caveat": "Confirm the unit of analysis."},
        {"area": "Relationships", "finding": "A and B are correlated (r=0.8).",
         "evidence": "Pearson r=0.8.", "caveat": "Correlation is not causation."},
    ]
    story = build_data_story(
        findings, title="Somalia sector brief", audience="Policy makers",
        context="Review sector performance over time.",
    )
    assert "Somalia sector brief" in story
    assert "Policy makers" in story
    assert "Pearson r=0.8" in story
    assert "Correlation does not demonstrate causation" in story
    assert "Recommended next steps" in story


def test_data_story_handles_empty_selection():
    from analytics import build_data_story
    story = build_data_story([], title="Empty brief")
    assert "No supported findings were selected" in story



def test_time_series_monthly_aggregation_sorts_and_calculates_changes():
    from analytics import build_time_series_analysis
    df = pd.DataFrame({
        "date": ["2024-03-10", "2024-01-10", "2024-01-20", "2024-02-10"],
        "value": [7, 2, 3, 5],
    })
    result = build_time_series_analysis(
        df, "date", "value", frequency="MS", aggregation="sum",
        moving_average_window=2,
    )
    data = result["data"]
    assert data["value"].tolist() == [5.0, 5.0, 7.0]
    assert data["period_change"].iloc[1] == pytest.approx(0.0)
    assert data["moving_average"].iloc[1] == pytest.approx(5.0)
    assert data["period"].is_monotonic_increasing
    assert result["summary"]["periods"] == 3


def test_time_series_zero_previous_period_has_no_infinite_percentage():
    from analytics import build_time_series_analysis
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, freq="MS"),
        "value": [0, 4, 8],
    })
    result = build_time_series_analysis(df, "date", "value", frequency="MS")
    pct = result["data"]["period_change_pct"]
    assert pd.isna(pct.iloc[1])
    assert pct.iloc[2] == pytest.approx(100.0)
    assert any("previous period is zero" in warning for warning in result["warnings"])


def test_time_series_requires_two_aggregated_periods():
    from analytics import build_time_series_analysis
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-15"], "value": [1, 2]})
    with pytest.raises(ValueError, match="fewer than two"):
        build_time_series_analysis(df, "date", "value", frequency="YS")


def test_time_series_unusual_change_detection_requires_six_periods():
    from analytics import build_time_series_analysis
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=4, freq="MS"),
        "value": [1, 2, 3, 4],
    })
    result = build_time_series_analysis(df, "date", "value")
    assert not result["data"]["unusual_change"].any()
    assert any("six aggregated periods" in warning for warning in result["warnings"])


def test_time_series_rejects_invalid_columns_and_frequency():
    from analytics import build_time_series_analysis
    df = pd.DataFrame({"date": ["2024-01-01", "2024-02-01"], "value": [1, 2]})
    with pytest.raises(ValueError, match="exist"):
        build_time_series_analysis(df, "missing", "value")
    with pytest.raises(ValueError, match="Frequency"):
        build_time_series_analysis(df, "date", "value", frequency="D")



def test_baseline_forecast_uses_chronological_holdout_and_last_value():
    from analytics import forecast_time_series_baseline
    result = forecast_time_series_baseline(
        pd.Series([10, 12, 11, 13, 15, 14, 16, 18, 17, 19]),
        horizon=3,
    )
    assert result["metrics"]["training_periods"] == 8
    assert result["metrics"]["holdout_periods"] == 2
    assert result["validation"]["actual"].tolist() == [17.0, 19.0]
    assert result["validation"]["predicted"].tolist() == [18.0, 18.0]
    assert result["forecast"]["forecast"].tolist() == [19.0, 19.0, 19.0]
    assert result["metrics"]["mae"] == pytest.approx(1.0)


def test_baseline_forecast_rejects_short_series_and_invalid_horizon():
    from analytics import forecast_time_series_baseline
    with pytest.raises(ValueError, match="eight observed periods"):
        forecast_time_series_baseline([1, 2, 3, 4, 5, 6, 7], horizon=2)
    with pytest.raises(ValueError, match="from 1 to 12"):
        forecast_time_series_baseline(range(1, 10), horizon=13)


def test_baseline_forecast_handles_zero_values_in_smape():
    from analytics import forecast_time_series_baseline
    result = forecast_time_series_baseline([0, 0, 0, 0, 0, 0, 0, 0], horizon=1)
    assert result["metrics"]["smape_pct"] == 0.0
    assert np.isfinite(result["metrics"]["mae"])
