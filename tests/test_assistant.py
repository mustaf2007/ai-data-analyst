import pandas as pd

from assistant import answer_dataset_question, suggest_analyses


def test_suggestions_detect_missing_values_and_correlations():
    df = pd.DataFrame({
        "sales": [10.0, None, 30.0],
        "cost": [4.0, 5.0, 8.0],
        "region": ["North", "South", "North"],
    })
    suggestions = suggest_analyses(df)
    titles = {item["title"] for item in suggestions}
    assert "Review missing values" in titles
    assert "Explore relationships" in titles
    assert all({"title", "reason", "action", "priority"} <= set(item) for item in suggestions)


def test_suggestions_empty_dataset_requests_upload():
    result = suggest_analyses(pd.DataFrame())
    assert result[0]["title"] == "Load a dataset"


def test_chatbot_answers_mean_using_actual_values():
    df = pd.DataFrame({"sales": [10, 20, 30]})
    result = answer_dataset_question(df, "What is the average sales?")
    assert result["intent"] == "mean"
    assert "20" in result["answer"]
    assert result["evidence"].iloc[0]["mean"] == 20


def test_chatbot_answers_grouped_highest():
    df = pd.DataFrame({
        "region": ["North", "North", "South", "South"],
        "sales": [10, 20, 50, 60],
    })
    result = answer_dataset_question(df, "Which region has the highest sales?")
    assert result["intent"] == "grouped"
    assert "South" in result["answer"]
    assert result["evidence"].iloc[0]["region"] == "South"


def test_chatbot_reports_missing_and_duplicate_rows():
    df = pd.DataFrame({"x": [1, 1, None], "y": ["a", "a", "b"]})
    missing = answer_dataset_question(df, "How many missing values?")
    duplicates = answer_dataset_question(df, "Are there duplicate rows?")
    assert missing["intent"] == "missing"
    assert duplicates["intent"] == "duplicates"
    assert "1" in missing["answer"]
    assert "1" in duplicates["answer"]


def test_chatbot_handles_correlations_and_unsupported_questions():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    result = answer_dataset_question(df, "What is the correlation between variables?")
    assert result["intent"] == "correlation"
    assert result["evidence"].iloc[0]["correlation"] > 0.99
    unsupported = answer_dataset_question(df, "Write me a poem")
    assert unsupported["intent"] == "unsupported"


def test_chatbot_follow_up_reuses_previous_measure_and_group():
    df = pd.DataFrame({
        "region": ["North", "North", "South", "South"],
        "sales": [10, 20, 50, 60],
    })
    first = answer_dataset_question(df, "Which region has the highest sales?")
    follow_up = answer_dataset_question(df, "What about North?", context=first["context"])
    assert first["context"] == {"measure": "sales", "group": "region"}
    assert follow_up["intent"] == "group_follow_up"
    assert "North" in follow_up["answer"]
    assert follow_up["evidence"].iloc[0]["value"] == 15


def test_chatbot_follow_up_can_change_grouped_statistic():
    df = pd.DataFrame({
        "region": ["North", "North", "South", "South"],
        "sales": [10, 20, 50, 60],
    })
    first = answer_dataset_question(df, "Which region has the highest sales?")
    follow_up = answer_dataset_question(df, "What about median?", context=first["context"])
    assert follow_up["intent"] == "grouped"
    assert "median" in follow_up["answer"].lower()
    assert set(follow_up["evidence"]["median"]) == {15, 55}


def test_chatbot_can_recommend_next_analysis_steps():
    df = pd.DataFrame({
        "sales": [10, None, 30],
        "cost": [4, 5, 8],
        "region": ["North", "South", "North"],
    })
    result = answer_dataset_question(df, "What should I do next?")
    assert result["intent"] == "recommendations"
    assert not result["evidence"].empty
    assert "Review missing values" in result["evidence"]["title"].tolist()



def test_chatbot_answers_dataset_wide_quality_risks():
    df = pd.DataFrame({
        "year": [2022, 2023, 2023, 2024],
        "imports": [100.0, None, 120.0, 130.0],
        "category": ["Food", "Fuel", "Fuel", " "],
        "constant": ["same", "same", "same", "same"],
    })
    result = answer_dataset_question(df, "What are the main data-quality risks?")
    assert result["intent"] == "data_quality"
    assert not result["evidence"].empty
    findings = result["evidence"].set_index("check")
    assert findings.loc["Missing values", "status"] == "Review"
    assert findings.loc["Blank text values", "status"] == "Review"
    assert findings.loc["Constant or empty columns", "status"] == "Review"
    assert "screen" in result["answer"].lower()


def test_temporal_trend_uses_year_and_month_columns():
    df = pd.DataFrame({
        "Year": [2022, 2022, 2023, 2023, 2024, 2024],
        "Month": ["January", "February"] * 3,
        "Food": [100, 120, 130, 150, 160, 180],
        "Oil": [50, 55, 60, 65, 70, 75],
    })
    result = answer_dataset_question(df, "How have Food imports changed over time?")
    assert result["intent"] == "temporal"
    assert result["evidence"]["period"].tolist() == ["2022-01", "2022-02", "2023-01", "2023-02", "2024-01", "2024-02"]
    assert result["evidence"].iloc[0]["Food"] == 100
    assert result["evidence"].iloc[-1]["Food"] == 180
    assert "increased" in result["answer"]


def test_temporal_trend_handles_numeric_months_and_zero_baseline():
    df = pd.DataFrame({"year": [2024, 2024, 2025], "month": [1, 2, 1], "exports": [0, 10, 20]})
    result = answer_dataset_question(df, "Show the exports trend")
    assert result["intent"] == "temporal"
    assert "starting value is zero" in result["answer"]
    assert pd.isna(result["evidence"]["percentage_change"].iloc[1])


def test_correlation_excludes_detected_year_and_month_dimensions():
    df = pd.DataFrame({
        "Year": [2022, 2023, 2024, 2025], "Month": ["Jan", "Feb", "Mar", "Apr"],
        "Food": [10, 20, 30, 40], "Oil": [100, 80, 60, 40],
    })
    result = answer_dataset_question(df, "What are the strongest correlations?")
    assert result["intent"] == "correlation"
    variables = set(result["evidence"]["variable_1"]) | set(result["evidence"]["variable_2"])
    assert "Year" not in variables and "Month" not in variables


def test_year_detection_does_not_treat_arbitrary_numeric_id_as_time():
    df = pd.DataFrame({"record_id": [2022, 2023, 2024], "sales": [1, 2, 3]})
    result = answer_dataset_question(df, "Show the sales trend")
    assert result["intent"] == "unsupported"
