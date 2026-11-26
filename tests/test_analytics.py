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



def test_dataset_comparison_reports_overview_schema_and_numeric_differences():
    from analytics import compare_datasets
    reference = pd.DataFrame({
        "year": [2022, 2023, 2024],
        "exports": [10.0, 20.0, 30.0],
        "region": ["North", "South", "North"],
    })
    comparison = pd.DataFrame({
        "year": [2023, 2024, 2025, 2026],
        "exports": [20.0, 30.0, 50.0, 60.0],
        "region": ["South", "South", "East", None],
        "new_indicator": [1, 2, 3, 4],
    })
    result = compare_datasets(reference, comparison)
    assert result["summary"]["rows"].tolist() == [3, 4]
    assert "new_indicator" in result["schema"]["column"].tolist()
    exports = result["numeric"].set_index("column")
    assert exports.loc["exports", "reference_mean"] == pytest.approx(20.0)
    assert exports.loc["exports", "comparison_mean"] == pytest.approx(40.0)
    assert exports.loc["exports", "mean_difference"] == pytest.approx(20.0)
    assert not result["categorical"].empty
    assert result["warnings"] == []


def test_dataset_comparison_handles_zero_reference_mean_and_missing_shared_columns():
    from analytics import compare_datasets
    result = compare_datasets(
        pd.DataFrame({"value": [0.0, 0.0], "left_only": ["a", "b"]}),
        pd.DataFrame({"value": [2.0, 4.0], "right_only": [1, 2]}),
    )
    value = result["numeric"].iloc[0]
    assert pd.isna(value["mean_difference_pct"])
    assert result["shared_columns"] == ["value"]
    assert result["categorical"].empty
    assert result["schema"]["column"].tolist() == ["value", "left_only", "right_only"]


def test_dataset_comparison_rejects_empty_or_invalid_inputs():
    from analytics import compare_datasets
    with pytest.raises(ValueError, match="both datasets"):
        compare_datasets(pd.DataFrame({"x": []}), pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="DataFrames"):
        compare_datasets({"x": [1]}, pd.DataFrame({"x": [1]}))


def test_dataset_comparison_skips_high_cardinality_categories_with_warning():
    from analytics import compare_datasets
    left = pd.DataFrame({"category": [f"c{i}" for i in range(5)]})
    right = pd.DataFrame({"category": [f"d{i}" for i in range(5)]})
    result = compare_datasets(left, right, categorical_limit=3)
    assert result["categorical"].empty
    assert any("categorical distribution skipped" in warning for warning in result["warnings"])



def test_geographic_summary_aggregates_sorts_and_calculates_share():
    from analytics import build_geographic_summary
    df = pd.DataFrame({
        "region": ["North", "South", "North", "East"],
        "production": [10, 20, 15, 5],
    })
    result = build_geographic_summary(df, "region", "production")
    data = result["data"].set_index("geography")
    assert data.loc["North", "value"] == pytest.approx(25)
    assert data.loc["South", "rank"] == 2
    assert data.loc["North", "share_pct"] == pytest.approx(25 / 50 * 100)
    assert result["coverage"]["geographic_groups"] == 3
    assert result["coverage"]["included_rows"] == 4


def test_geographic_summary_excludes_missing_labels_and_invalid_values():
    from analytics import build_geographic_summary
    df = pd.DataFrame({
        "country": ["Somalia", None, "Kenya", "  "],
        "gdp": [100, 20, np.inf, 40],
    })
    result = build_geographic_summary(df, "country", "gdp")
    assert result["data"]["geography"].tolist() == ["Somalia"]
    assert result["coverage"]["excluded_rows"] == 3
    assert len(result["warnings"]) == 3
    assert any("Only one geographic group" in warning for warning in result["warnings"])


def test_geographic_summary_supports_mean_and_top_n():
    from analytics import build_geographic_summary
    df = pd.DataFrame({
        "district": ["A", "A", "B", "C"],
        "rate": [2, 4, 3, 8],
    })
    result = build_geographic_summary(df, "district", "rate", aggregation="mean", top_n=2)
    assert result["data"]["geography"].tolist() == ["C", "B"]
    assert result["data"]["value"].tolist() == [8.0, 3.0]
    assert result["data"]["share_pct"].isna().all()


def test_geographic_summary_validates_columns_and_measure():
    from analytics import build_geographic_summary
    df = pd.DataFrame({"area": ["A"], "count": [1]})
    with pytest.raises(ValueError, match="must exist"):
        build_geographic_summary(df, "missing", "count")
    with pytest.raises(ValueError, match="numeric measure"):
        build_geographic_summary(
            pd.DataFrame({"area": ["A"], "count": ["unknown"]}), "area", "count"
        )


def test_one_sample_t_test_reports_ci_effect_and_significance():
    from analytics import run_statistical_test
    df = pd.DataFrame({"score": [2, 3, 4, 5, 6]})
    result = run_statistical_test(
        df, "One-sample t-test", "score", hypothesized_mean=0, alpha=0.05
    )
    assert result["n"] == 5
    assert result["p_value"] < 0.05
    assert result["effect_size_name"] == "Cohen's d (one-sample)"
    assert result["confidence_interval"][0] < 4 < result["confidence_interval"][1]


def test_welch_t_test_handles_unequal_sample_sizes_and_reports_ci():
    from analytics import run_statistical_test
    df = pd.DataFrame({
        "group": ["A"] * 5 + ["B"] * 8,
        "value": [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11, 12],
    })
    result = run_statistical_test(df, "Independent-samples Welch t-test", "value", group_column="group")
    assert result["n"] == 13
    assert result["degrees_of_freedom"] > 0
    assert len(result["confidence_interval"]) == 2
    assert result["group_summary"]["n"].tolist() == [5, 8]


def test_one_way_anova_returns_eta_squared_and_variance_warning():
    from analytics import run_statistical_test
    df = pd.DataFrame({
        "group": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        "value": [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13],
    })
    result = run_statistical_test(df, "One-way ANOVA", "value", group_column="group")
    assert result["p_value"] < 0.05
    assert 0 <= result["effect_size"] <= 1
    assert result["degrees_of_freedom"] == [2, 9]


def test_chi_square_returns_contingency_expected_counts_and_cramers_v():
    from analytics import run_statistical_test
    df = pd.DataFrame({
        "region": ["North"] * 20 + ["South"] * 20,
        "choice": ["A"] * 15 + ["B"] * 5 + ["A"] * 5 + ["B"] * 15,
    })
    result = run_statistical_test(
        df, "Chi-square test of independence", "region", group_column="choice"
    )
    assert result["n"] == 40
    assert result["contingency"].shape == (2, 2)
    assert result["expected_frequencies"].shape == (2, 2)
    assert result["effect_size_name"] == "Cramer's V"
    assert 0 <= result["effect_size"] <= 1


def test_statistical_test_rejects_small_samples_bad_groups_and_invalid_alpha():
    from analytics import run_statistical_test
    with pytest.raises(ValueError, match="at least two finite"):
        run_statistical_test(pd.DataFrame({"x": [1]}), "One-sample t-test", "x")
    with pytest.raises(ValueError, match="exactly two groups"):
        run_statistical_test(
            pd.DataFrame({"g": ["a", "a", "b", "b", "c", "c"], "x": [1, 2, 3, 4, 5, 6]}),
            "Independent-samples Welch t-test", "x", group_column="g"
        )
    with pytest.raises(ValueError, match="alpha"):
        run_statistical_test(pd.DataFrame({"x": [1, 2]}), "One-sample t-test", "x", alpha=1)


def test_statistical_test_rejects_zero_variance_and_sparse_chi_square_is_flagged():
    from analytics import run_statistical_test
    with pytest.raises(ValueError, match="zero variance"):
        run_statistical_test(pd.DataFrame({"x": [2, 2, 2]}), "One-sample t-test", "x")
    sparse = pd.DataFrame({
        "a": ["x"] * 5 + ["y"] * 2,
        "b": ["m", "n", "o", "p", "q", "m", "n"],
    })
    result = run_statistical_test(sparse, "Chi-square test of independence", "a", group_column="b")
    assert any("expected cell counts" in warning for warning in result["warnings"])


def test_anova_runs_tukey_hsd_with_adjusted_pairwise_results():
    from analytics import run_statistical_test
    df = pd.DataFrame({
        "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "value": [1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14],
    })
    result = run_statistical_test(df, "One-way ANOVA", "value", group_column="group")
    pairwise = result["posthoc_comparisons"]
    assert len(pairwise) == 3
    assert {"group_1", "group_2", "mean_difference", "ci_lower", "ci_upper",
            "adjusted_p_value", "significant"}.issubset(pairwise.columns)
    assert pairwise["adjusted_p_value"].between(0, 1).all()
    assert pairwise["significant"].any()


def test_anova_skips_tukey_when_variances_are_unequal():
    from analytics import run_statistical_test
    df = pd.DataFrame({
        "group": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
        "value": [-100, -100, -100, 100, 100, 100, -1, -1, -1, 1, 1, 1, -2, -2, -2, 2, 2, 2],
    })
    result = run_statistical_test(df, "One-way ANOVA", "value", group_column="group")
    assert result["posthoc_comparisons"].empty
    assert any("Tukey HSD pairwise comparisons were not run" in warning
               for warning in result["warnings"])



def test_build_trend_frame_excludes_same_time_and_measure_column():
    from analytics import build_trend_frame, DataInputError
    df = pd.DataFrame({"Year": [2020, 2019, 2021], "GDP": [2.0, 1.0, 3.0]})
    with pytest.raises(DataInputError, match="different numeric indicator"):
        build_trend_frame(df, "Year", "Year")
    result = build_trend_frame(df, "Year", "GDP")
    assert result["Year"].tolist() == [2019, 2020, 2021]
    assert result["GDP"].tolist() == [1.0, 2.0, 3.0]


def test_build_trend_frame_handles_invalid_dates_and_non_numeric_values():
    from analytics import build_trend_frame
    df = pd.DataFrame({
        "period": ["2024-02-01", "not-a-date", "2024-01-01"],
        "measure": ["2.5", "3.0", "bad"],
    })
    result = build_trend_frame(df, "period", "measure", parse_datetime=True)
    assert len(result) == 1
    assert result.iloc[0]["measure"] == 2.5


def test_build_trend_frame_rejects_duplicate_headers():
    from analytics import build_trend_frame, DataInputError
    df = pd.DataFrame([[2020, 1, 2]], columns=["Year", "GDP", "GDP"])
    with pytest.raises(DataInputError, match="column names are duplicated"):
        build_trend_frame(df, "Year", "GDP")


def test_load_csv_makes_headers_unique_even_when_suffix_already_exists():
    df = load_dataset("headers.csv", b"a,a,a.1,a,a.2\n1,2,3,4,5\n")
    assert df.shape == (1, 5)
    assert df.columns.is_unique


def test_header_normalization_handles_generated_suffix_collisions():
    from analytics import _make_unique_column_names
    columns = _make_unique_column_names(["a", "a", "a.1", "a", "a.2"])
    assert columns == ["a", "a.1", "a.1.1", "a.2", "a.2.1"]
    assert len(columns) == len(set(columns))
