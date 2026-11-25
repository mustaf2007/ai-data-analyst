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
